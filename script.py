import argparse
import os
import retro
import numpy as np
import torch
import torch.nn as nn
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces, Wrapper
import optuna
import logging
import atexit
import signal
import sys

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(SimpleCNN, self).__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict"
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + enemy_vec_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
            
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
            
        if len(viewport.shape) == 4:
            if viewport.shape[3] == 4 or viewport.shape[3] == 1:
                viewport = viewport.permute(0, 3, 1, 2)
        
        if viewport.shape[1] != 4:
            raise ValueError(f"Expected 4 channels in viewport, got shape {viewport.shape}")
            
        cnn_output = self.cnn(viewport)
        
        if len(enemy_vector.shape) == 1:
            enemy_vector = enemy_vector.unsqueeze(0)
            
        combined = torch.cat([cnn_output, enemy_vector], dim=1)
        return self.linear(combined)

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (4,), np.float32)
        })
        self.actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0,0,0]
        ]
        self.projectiles = []
        self.last_hp = 0
        self.action_counts = np.zeros(11)
        self.last_frame = None
        self.success_dodges = 0
        self.failed_dodges = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.action_counts = np.zeros(11)
        self.projectiles = []
        self.last_frame = None
        self.success_dodges = 0
        self.failed_dodges = 0
        return self._get_obs(obs)

    def step(self, action):
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        score = sum(ram[0x0531:0x0536]) * 100
        scroll = ram[0x00E5]
        hp = ram[0x04A6]
        pos_x = ram[0x0094]
        
        prev_projectiles = self.projectiles.copy()
        self._update_projectiles(obs)
        
        hp_loss = max(0, int(self.last_hp) - int(hp))
        dodge_reward = self._calculate_dodge_reward(prev_projectiles, pos_x, hp_loss)
        reward = (score * 0.1 + scroll * 0.5 - hp_loss * 10 + dodge_reward)
        self.last_hp = hp
        
        info.update({
            "score": score, "hp": hp, "pos_x": pos_x,
            "action_percentages": self.action_counts / (sum(self.action_counts) + 1e-6),
            "projectiles": len(self.projectiles),
            "success_dodges": self.success_dodges,
            "failed_dodges": self.failed_dodges
        })
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        viewport = cv2.resize(gray, self.viewport_size)[..., np.newaxis]
        
        ram = self.env.get_ram()
        hero_x = int(ram[0x0094])
        enemy_xs = [
            int(ram[0x008E]), int(ram[0x008F]),
            int(ram[0x0090]), int(ram[0x0091])
        ]
        
        enemy_vector = np.array([e - hero_x if e != 0 else 0 for e in enemy_xs], dtype=np.float32)
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector
        }

    def _update_projectiles(self, obs):
        if self.last_frame is None:
            self.last_frame = obs
            self.projectiles = []
            return

        curr_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)
        
        frame_diff = cv2.absdiff(curr_gray, last_gray)
        thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        new_projectiles = []
        for contour in contours:
            if 10 < cv2.contourArea(contour) < 100:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                
                speed_x = 0
                for old_x, old_y, old_speed in self.projectiles:
                    if abs(old_y - cy) < 10:
                        speed_x = cx - old_x
                        break
                
                new_projectiles.append((cx, cy, speed_x))
        
        self.projectiles = new_projectiles
        self.last_frame = obs.copy()

    def _calculate_dodge_reward(self, prev_projectiles, hero_x, hp_loss):
        dodge_reward = 0
        for px, py, speed in prev_projectiles:
            if abs(px - hero_x) < 20 and speed != 0:
                new_dist = abs(px + speed - hero_x)
                if hp_loss == 0 and new_dist > 20:
                    dodge_reward += 5
                    self.success_dodges += 1
                elif hp_loss > 0:
                    dodge_reward -= 5
                    self.failed_dodges += 1
        return dodge_reward

def debug_network_dimensions():
    env = make_env()
    obs = env.reset()
    
    cnn = SimpleCNN(env.observation_space, features_dim=256)
    
    viewport_tensor = torch.from_numpy(obs['viewport']).unsqueeze(0).float()
    viewport_tensor = viewport_tensor.permute(0, 3, 1, 2)
    enemy_tensor = torch.from_numpy(obs['enemy_vector']).unsqueeze(0).float()
    
    viewport_stacked = torch.cat([viewport_tensor] * 4, dim=1)
    
    with torch.no_grad():
        cnn_output = cnn.cnn(viewport_stacked)
        combined = torch.cat([cnn_output, enemy_tensor], dim=1)
        final_output = cnn.linear(combined)
    
    env.close()
    return

def make_env(render=False):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    if render:
        env.render_mode = 'human'
    return KungFuWrapper(env)

def make_vec_env(num_envs, render=False):
    if num_envs > 1:
        return SubprocVecEnv([lambda: make_env(render) for _ in range(num_envs)])
    return DummyVecEnv([lambda: make_env(render)])

class TrainingCallback(BaseCallback):
    def __init__(self, progress_bar=False, logger=None):
        super().__init__()
        self.projectile_logs = []
        self.total_reward = 0
        self.progress_bar = progress_bar
        self.logger = logger
        if progress_bar:
            from tqdm import tqdm
            self.pbar = tqdm(total=10000)

    def _on_step(self):
        # Get the vectorized environment
        vec_env = self.training_env
        
        # Handle VecFrameStack unwrapping
        if isinstance(vec_env, VecFrameStack):
            vec_env = vec_env.venv
        
        # Check if we can access individual environments
        if hasattr(vec_env, 'envs'):
            # For DummyVecEnv
            env = vec_env.envs[0]
            while hasattr(env, 'env'):
                env = env.env
            ram = env.env.get_ram()
        else:
            # For SubprocVecEnv
            ram = None
        
        # Get info from the first environment
        info = self.locals["infos"][0]
        
        # Log data
        log_entry = {
            "action_percentages": info["action_percentages"],
            "projectiles": info["projectiles"],
            "success_dodges": info["success_dodges"],
            "failed_dodges": info["failed_dodges"]
        }
        
        if ram is not None:
            log_entry.update({
                "agent_x": ram[0x0094],
                "agent_y": ram[0x00B6],
                "hp_loss": max(0, env.last_hp - ram[0x04A6])
            })
        
        self.projectile_logs.append(log_entry)
        self.total_reward += self.locals["rewards"][0]
        
        if self.logger:
            self.logger.info(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][0]}, Score: {info['score']}")
            self.logger.record("rollout/ep_rew_mean", np.mean(self.locals["rewards"]))
            self.logger.record("metrics/score", info["score"])
            self.logger.record("metrics/hp", info["hp"])
            self.logger.record("metrics/projectiles", info["projectiles"])
            self.logger.record("metrics/success_dodges", info["success_dodges"])
            self.logger.record("metrics/failed_dodges", info["failed_dodges"])
        
        if self.progress_bar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.progress_bar:
            self.pbar.close()
                     
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 2048, 16384, step=2048)
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    env = make_vec_env(1)
    env = VecFrameStack(env, n_stack=4)
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs={"features_extractor_class": SimpleCNN},
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        tensorboard_log=args.log_dir,
        device="cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    callback = TrainingCallback(progress_bar=args.progress_bar)
    try:
        model.learn(
            total_timesteps=args.eval_timesteps,
            callback=callback,
            tb_log_name=f"PPO_KungFu_trial_{trial.number}"
        )
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')
    
    return callback.total_reward / args.eval_timesteps

# Global variable to store the model for emergency save
global_model = None
global_model_file = None

def save_model_on_exit():
    global global_model, global_model_file
    if global_model is not None and global_model_file is not None:
        try:
            global_model.save(global_model_file)
            print(f"\nEmergency save: Model saved to {global_model_file}.zip")
        except Exception as e:
            print(f"\nFailed to save model: {e}")
    sys.exit(0)

def signal_handler(sig, frame):
    print('\nReceived Ctrl+C, saving model...')
    save_model_on_exit()

def train(args):
    global global_model, global_model_file
    
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_model_on_exit)
    
    debug_network_dimensions()
    
    if args.enable_file_logging:
        logging.basicConfig(
            filename=os.path.join(args.log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    else:
        logger = None

    # Define model path and ensure directory exists
    model_path = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    model_file = os.path.join(model_path, "kungfu_ppo_optuna")
    global_model_file = model_file
    os.makedirs(model_path, exist_ok=True)
    
    # Check for existing model and optimization results
    model_exists = os.path.exists(model_file + ".zip")
    optuna_study_file = os.path.join(model_path, "optuna_study.pkl")
    optuna_completed = os.path.exists(optuna_study_file)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    env = make_vec_env(args.num_envs, render=args.render)
    env = VecFrameStack(env, n_stack=4)
    
    # Default best parameters
    best_params = {
        'learning_rate': 0.0007181399768445935, 
        'n_steps': 16384, 
        'batch_size': 256, 
        'n_epochs': 4, 
        'gamma': 0.9051620100876205, 
        'clip_range': 0.1003777999324603
    }
    
    # Load or create model
    if args.resume and model_exists:
        model = PPO.load(model_file, env=env, device=device)
        print(f"Resumed from {model_file}.zip")
    else:
        if not args.skip_optuna and not optuna_completed:
            print("Running Optuna optimization...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
            
            print("Best hyperparameters:", study.best_params)
            print("Best value:", study.best_value)
            best_params = study.best_params
            
            # Save Optuna study
            import pickle
            with open(optuna_study_file, 'wb') as f:
                pickle.dump(study, f)
            print(f"Optuna study saved to {optuna_study_file}")
        elif optuna_completed:
            print("Loading previous Optuna results...")
            import pickle
            with open(optuna_study_file, 'rb') as f:
                study = pickle.load(f)
            best_params = study.best_params
            print("Loaded best hyperparameters:", best_params)
        
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={"features_extractor_class": SimpleCNN},
            learning_rate=best_params["learning_rate"],
            n_steps=best_params["n_steps"],
            batch_size=best_params["batch_size"],
            n_epochs=best_params["n_epochs"],
            gamma=best_params["gamma"],
            clip_range=best_params["clip_range"],
            tensorboard_log=args.log_dir,
            device=device
        )

    global_model = model
    
    callback = TrainingCallback(progress_bar=args.progress_bar, logger=logger)
    if args.progress_bar:
        callback.pbar.total = args.timesteps
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO_KungFu_Best",
            reset_num_timesteps=not args.resume
        )
        
        # Save the final model
        model.save(model_file)
        print(f"Model saved to {model_file}.zip")
        
    except KeyboardInterrupt:
        save_model_on_exit()
    
    global_model = None
    global_model_file = None

def play(args):
    model_file = os.path.join(args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path), "kungfu_ppo_optuna")
    if not os.path.exists(model_file + ".zip"):
        print(f"Error: No trained model found at {model_file}.zip. Please train a model first.")
        sys.exit(1)

    env = make_vec_env(1, render=True)
    env = VecFrameStack(env, n_stack=4)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO.load(model_file, env=env, device=device)
    print(f"Loaded model from {model_file}.zip")

    obs = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    try:
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            if step_count % 100 == 0:
                print(f"Step: {step_count}, Reward: {reward[0]}, Total Reward: {total_reward}, Info: {info[0]}")
            
            env.render()

    except KeyboardInterrupt:
        print("\nPlay interrupted by user.")
    
    finally:
        print(f"Play ended. Total steps: {step_count}, Total reward: {total_reward}")
        env.close()  # Ensure environment is properly closed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Kung Fu with PPO.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--play", action="store_true", help="Play the game with a trained model")
    parser.add_argument("--model_path", default="models/kungfu_ppo_optuna", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps for training")
    parser.add_argument("--eval_timesteps", type=int, default=10_000, help="Timesteps per Optuna trial")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--timeout", type=int, default=3600, help="Optuna timeout in seconds")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--render", action="store_true", help="Render during training (slows down)")
    parser.add_argument("--enable_file_logging", action="store_true", help="Enable file logging")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--skip_optuna", action="store_true", help="Skip Optuna optimization")
    
    args = parser.parse_args()

    if args.train or args.play:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        if args.train:
            os.makedirs(args.log_dir, exist_ok=True)

    if args.train and args.play:
        print("Error: Cannot use --train and --play together. Choose one mode.")
        sys.exit(1)
    elif args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        print("Please specify a mode: --train or --play")
        parser.print_help()