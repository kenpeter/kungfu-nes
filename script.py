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
import time
from datetime import datetime
import pickle
from tqdm import tqdm

# Global variables for emergency saving
global_model = None
global_model_file = None
logger = None

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
        self.last_hp = 0
        self.action_counts = np.zeros(11)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]  # Initialize HP
        self.action_counts = np.zeros(11)
        return self._get_obs(obs)

    def step(self, action):
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        # Extract key variables from RAM
        score = sum(ram[0x0531:0x0536]) * 100  # Combine score digits (5 bytes)
        scroll = ram[0x00E5]                    # Screen Scroll 1 (primary scroll value)
        hp = ram[0x04A6]                        # Hero HP
        pos_x = ram[0x0094]                     # Hero X position
        stage = ram[0x0058]                     # Current stage
        boss_hp = ram[0x04A5]                   # Boss HP
        
        # Calculate HP loss
        hp_loss = max(0, int(self.last_hp) - int(hp))
        
        # Reward components
        raw_reward = (
            score * 0.01 +           # Small reward for accumulating score
            scroll * 0.5 +           # Reward for progressing through the level
            pos_x * 0.1 +            # Reward for moving right
            (255 - boss_hp) * 1.0 -  # Reward for damaging boss (255 is max HP)
            hp_loss * 10.0           # Penalty for losing HP
        )
        
        # Clip and normalize reward
        clipped_reward = np.clip(raw_reward, -10, 10)
        normalized_reward = clipped_reward / 10.0  # Range [-1, 1]
        
        # Update last HP for next step
        self.last_hp = hp
        
        # Info for logging/debugging
        info.update({
            "score": score,
            "hp": hp,
            "pos_x": pos_x,
            "scroll": scroll,
            "stage": stage,
            "boss_hp": boss_hp,
            "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "action_percentages": self.action_counts / (sum(self.action_counts) + 1e-6)
        })
        
        return self._get_obs(obs), normalized_reward, done, info
    def _get_obs(self, obs):
        # Convert to grayscale and resize
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        viewport = cv2.resize(gray, self.viewport_size)[..., np.newaxis]
        
        # Get enemy positions relative to hero
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
    
class TrainingCallback(BaseCallback):
    def __init__(self, progress_bar=False, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger()
        self.progress_bar = progress_bar
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        
        if progress_bar:
            self.pbar = tqdm(total=10000, desc="Training")

    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals["rewards"][0]
        
        # Log to TensorBoard
        if self.num_timesteps % 100 == 0:
            info = self.locals["infos"][0]
            self.logger.record("train/step_reward", self.locals["rewards"][0])
            self.logger.record("train/total_reward", self.current_episode_reward)
            self.logger.record("game/score", info.get("score", 0))
            self.logger.record("game/hp", info.get("hp", 0))
            self.logger.record("game/projectiles", info.get("projectiles", 0))
            self.logger.record("game/success_dodges", info.get("success_dodges", 0))
            self.logger.record("game/failed_dodges", info.get("failed_dodges", 0))
            self.logger.record("time/steps_per_second", self.total_steps / (time.time() - self.start_time))
        
        # Console logging
        current_time = time.time()
        if current_time - self.last_log_time > 30 or self.total_steps % 100 == 0:
            info = self.locals["infos"][0]
            self.logger.info(
                f"Step {self.total_steps}: "
                f"Reward={self.locals['rewards'][0]:.2f}, "
                f"Total Reward={self.current_episode_reward:.2f}, "
                f"Score={info.get('score', 0)}, "
                f"HP={info.get('hp', 0)}, "
                f"Projectiles={info.get('projectiles', 0)}, "
                f"Dodges={info.get('success_dodges', 0)}"
            )
            self.last_log_time = current_time
        
        if self.progress_bar:
            self.pbar.update(1)
        return True

    def _on_rollout_end(self):
        self.episode_count += 1
        self.episode_rewards.append(self.current_episode_reward)
        
        # Log episode metrics to TensorBoard
        self.logger.record("train/episode_reward", self.current_episode_reward)
        self.logger.record("train/avg_episode_reward", np.mean(self.episode_rewards))
        self.logger.record("train/episode_length", self.total_steps / self.episode_count)
        
        self.logger.info(
            f"Episode {self.episode_count} completed: "
            f"Total Reward={self.current_episode_reward:.2f}, "
            f"Avg Reward={np.mean(self.episode_rewards):.2f}"
        )
        self.current_episode_reward = 0
        
    def _on_training_end(self):
        if self.progress_bar:
            self.pbar.close()
        training_duration = time.time() - self.start_time
        self.logger.info(
            f"Training completed. Total steps: {self.total_steps}, "
            f"Episodes: {self.episode_count}, "
            f"Avg Reward: {np.mean(self.episode_rewards):.2f}, "
            f"Duration: {training_duration:.2f} seconds"
        )
        
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

def save_model_with_logging(model, path, logger):
    logger.info(f"Starting model save to {path}")
    start_time = time.time()
    
    logger.info("Saving model components:")
    logger.info(f"- Policy network: {model.policy}")
    optimizer_state = getattr(model.policy, 'optimizer', None)
    if optimizer_state:
        logger.info(f"- Optimizer state: {optimizer_state.state_dict()}")
    else:
        logger.info("- Optimizer state: Not initialized (training may not have started)")
    logger.info(f"- Features extractor: {model.policy.features_extractor}")
    
    try:
        save_start = time.time()
        model.save(path)
        save_time = time.time() - save_start
        file_size = os.path.getsize(path + '.zip') / (1024*1024)
        logger.info(f"Model successfully saved in {save_time:.2f} seconds")
        logger.info(f"Model size: {file_size:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

def signal_handler(sig, frame):
    logger.info('\nReceived interrupt signal, saving model...')
    save_model_on_exit()

def save_model_on_exit():
    global global_model, global_model_file, logger
    if global_model is not None and global_model_file is not None:
        try:
            logger.info("Emergency save initiated")
            save_model_with_logging(global_model, global_model_file, logger)
        except Exception as e:
            logger.error(f"Emergency save failed: {str(e)}")
    sys.exit(0)

def make_env(render=False):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    if render:
        env.render_mode = 'human'
    return KungFuWrapper(env)

def make_vec_env(num_envs, render=False):
    logger.info(f"Creating vectorized environment with {num_envs} subprocesses")
    start_time = time.time()
    
    if num_envs > 1:
        env = SubprocVecEnv([lambda: make_env(render) for _ in range(num_envs)])
    else:
        env = DummyVecEnv([lambda: make_env(render)])
    
    logger.info(f"Environment created in {time.time() - start_time:.2f} seconds")
    return env

def objective(trial):
    logger.info(f"Starting Optuna trial {trial.number}")
    
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

    callback = TrainingCallback(progress_bar=args.progress_bar, logger=logger)
    try:
        model.learn(
            total_timesteps=args.eval_timesteps,
            callback=callback,
            tb_log_name=f"PPO_KungFu_trial_{trial.number}"
        )
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return float('-inf')
    finally:
        env.close()
    
    avg_reward = np.mean(callback.episode_rewards)
    logger.info(f"Trial {trial.number} completed with average reward: {avg_reward:.2f}")
    return avg_reward

def train(args):
    global global_model, global_model_file, logger
    
    logger = setup_logging(args.log_dir)
    logger.info("Starting training session")
    logger.info(f"Command line arguments: {vars(args)}")
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_model_on_exit)
    
    # Environment setup
    logger.info("Creating environments...")
    env_start_time = time.time()
    env = make_vec_env(args.num_envs, render=args.render)
    env = VecFrameStack(env, n_stack=4)
    logger.info(f"Environments created in {time.time() - env_start_time:.2f} seconds")
    
    # Model setup
    model_path = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    model_file = os.path.join(model_path, "kungfu_ppo_optuna")
    global_model_file = model_file
    os.makedirs(model_path, exist_ok=True)
    
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
    if args.resume and os.path.exists(model_file + ".zip"):
        logger.info(f"Loading existing model from {model_file}.zip")
        load_start = time.time()
        model = PPO.load(model_file, env=env, device="cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"Model loaded in {time.time() - load_start:.2f} seconds")
    else:
        if not args.skip_optuna and not os.path.exists(os.path.join(model_path, "optuna_study.pkl")):
            logger.info("Running Optuna optimization...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
            
            logger.info("Best hyperparameters: %s", study.best_params)
            logger.info("Best value: %s", study.best_value)
            best_params = study.best_params
            
            # Save Optuna study
            study_file = os.path.join(model_path, "optuna_study.pkl")
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            logger.info(f"Optuna study saved to {study_file}")
        elif os.path.exists(os.path.join(model_path, "optuna_study.pkl")):
            logger.info("Loading previous Optuna results...")
            study_file = os.path.join(model_path, "optuna_study.pkl")
            with open(study_file, 'rb') as f:
                study = pickle.load(f)
            best_params = study.best_params
            logger.info("Loaded best hyperparameters: %s", best_params)
        
        logger.info("Creating new model with parameters: %s", best_params)
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
            device="cuda" if args.cuda and torch.cuda.is_available() else "cpu"
        )

    global_model = model
    
    callback = TrainingCallback(progress_bar=args.progress_bar, logger=logger)
    if args.progress_bar:
        callback.pbar.total = args.timesteps
    
    try:
        logger.info(f"Starting training for {args.timesteps} timesteps")
        train_start = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO_KungFu_Best",
            reset_num_timesteps=not args.resume
        )
        logger.info(f"Training completed in {time.time() - train_start:.2f} seconds")
        
        # Model saving
        save_model_with_logging(model, model_file, logger)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        save_model_on_exit()
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise
    finally:
        logger.info("Cleaning up resources...")
        env.close()
        global_model = None
        global_model_file = None
        logger.info("Training session ended")

def play(args):
    global logger
    logger = setup_logging(args.log_dir)
    logger.info("Starting play session")
    
    model_file = os.path.join(args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path), "kungfu_ppo_optuna")
    if not os.path.exists(model_file + ".zip"):
        logger.error(f"No trained model found at {model_file}.zip")
        sys.exit(1)

    logger.info("Creating environment...")
    env = make_vec_env(1, render=True)
    env = VecFrameStack(env, n_stack=4)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model from {model_file}.zip")
    model = PPO.load(model_file, env=env, device=device)

    obs = env.reset()
    total_reward = 0
    done = False
    step_count = 0

    try:
        logger.info("Starting gameplay...")
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            if step_count % 100 == 0:
                logger.info(
                    f"Step: {step_count}, Reward: {reward[0]:.2f}, "
                    f"Total Reward: {total_reward:.2f}, HP: {info[0].get('hp', 0)}"
                )
            
            env.render()

    except KeyboardInterrupt:
        logger.info("\nPlay interrupted by user.")
    finally:
        logger.info(f"Play ended. Total steps: {step_count}, Total reward: {total_reward:.2f}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Kung Fu with PPO.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--play", action="store_true", help="Play the game with a trained model")
    parser.add_argument("--model_path", default="models/kungfu_ppo_optuna", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total timesteps for training")
    parser.add_argument("--eval_timesteps", type=int, default=10_000, help="Timesteps per Optuna trial")
    parser.add_argument("--n_trials", type=int, default=5, help="Number of Optuna trials")
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