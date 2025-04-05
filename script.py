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
import logging
import atexit
import signal
import sys
import time
from datetime import datetime
from tqdm import tqdm
import optuna

# Global variables
global_model = None
global_model_file = None
logger = None

# [SimpleCNN, KungFuWrapper, CombatTrainingCallback classes remain unchanged]
# Skipping repetition for brevity
class SimpleCNN(BaseFeaturesExtractor):
    # ... (unchanged)
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
        combat_status_size = observation_space["combat_status"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + enemy_vec_size + combat_status_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        combat_status = observations["combat_status"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
        if isinstance(combat_status, np.ndarray):
            combat_status = torch.from_numpy(combat_status)
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and (viewport.shape[3] == 4 or viewport.shape[3] == 1):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        if len(enemy_vector.shape) == 1:
            enemy_vector = enemy_vector.unsqueeze(0)
        if len(combat_status.shape) == 1:
            combat_status = combat_status.unsqueeze(0)
            
        combined = torch.cat([cnn_output, enemy_vector, combat_status], dim=1)
        return self.linear(combined)

class KungFuWrapper(Wrapper):
    # ... (unchanged)
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        
        # Define actions with names
        self.actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No-op
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Punch
            [0,0,0,0,0,0,0,0,1,0,0,0],  # Kick
            [1,0,0,0,0,0,1,0,0,0,0,0],  # Right+Punch
            [0,1,0,0,0,0,1,0,0,0,0,0],  # Left+Punch
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Jump
            [0,0,1,0,0,0,0,0,0,0,0,0],  # Crouch
            [0,0,0,0,0,1,1,0,0,0,0,0],  # Jump+Punch
            [0,0,1,0,0,0,1,0,0,0,0,0]   # Crouch+Punch
        ]
        self.action_names = [
            "No-op",
            "Punch",
            "Kick",
            "Right+Punch", 
            "Left+Punch",
            "Jump",
            "Crouch",
            "Jump+Punch",
            "Crouch+Punch"
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (8,), np.float32),
            "combat_status": spaces.Box(0, 1, (1,), np.float32)
        })
        
        self.last_hp = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, **kwargs):
        self.current_step = 0
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4
        return self._get_obs(obs)

    def step(self, action):
        self.current_step += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = ram[0x04A6]
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        hp_loss = max(0, int(self.last_hp) - int(hp))
        
        raw_reward = (
            enemy_hit * 300.0 +
            -hp_loss * 100.0 + 
            (0.1 if enemy_hit > 0 else -0.1)
        )
        
        self.last_hp = hp
        self.last_enemies = curr_enemies
        
        if self.current_step > 50 and enemy_hit == 0:
            done = True
            raw_reward -= 50.0
        
        normalized_reward = np.clip(raw_reward / 300.0, -1, 1)
        
        info.update({
            "hp": hp,
            "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "action_percentages": self.action_counts / (sum(self.action_counts) + 1e-6),
            "action_names": self.action_names,
            "enemy_hit": enemy_hit
        })
        
        return self._get_obs(obs), normalized_reward, done, info

    def _get_obs(self, obs):
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        viewport = cv2.resize(gray, self.viewport_size)[..., np.newaxis]
        ram = self.env.get_ram()
        
        hero_x = int(ram[0x0094])
        enemy_info = []
        for addr in [0x008E, 0x008F, 0x0090, 0x0091]:
            enemy_x = int(ram[addr])
            if enemy_x != 0:
                distance = enemy_x - hero_x
                direction = 1 if distance > 0 else -1
                enemy_info.extend([direction, min(abs(distance), 255)])
            else:
                enemy_info.extend([0, 0])
        
        enemy_vector = np.array(enemy_info, dtype=np.float32)
        combat_status = np.array([self.last_hp/255.0], dtype=np.float32)
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector,
            "combat_status": combat_status
        }

class CombatTrainingCallback(BaseCallback):
    # ... (unchanged)
    def __init__(self, progress_bar=False, logger=None):
        super().__init__()
        self.logger = logger or logging.getLogger()
        self.progress_bar = progress_bar
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.total_steps = 0
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_hits = 0
        self.total_hits = 0
        
        if progress_bar:
            self.pbar = tqdm(total=10000, desc="Training")

    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals["rewards"][0]
        hit_count = self.locals["infos"][0]["enemy_hit"]
        self.episode_hits += hit_count
        self.total_hits += hit_count
        
        if self.num_timesteps % 100 == 0:
            info = self.locals["infos"][0]
            hit_rate = self.total_hits / (self.total_steps + 1e-6)
            
            self.logger.record("combat/step_reward", self.locals["rewards"][0])
            self.logger.record("combat/total_episode_reward", self.current_episode_reward)
            self.logger.record("combat/hit_count", hit_count)
            self.logger.record("combat/total_hits", self.total_hits)
            self.logger.record("combat/hit_rate", hit_rate)
            self.logger.record("game/hp", info.get("hp", 0))
            
            action_percentages = info.get("action_percentages", np.zeros(9))
            action_names = info.get("action_names", [])
            for name, percentage in zip(action_names, action_percentages):
                self.logger.record(f"actions/{name.replace('+', '_')}", percentage)
            
            steps_per_second = self.total_steps / (time.time() - self.start_time)
            self.logger.record("time/steps_per_second", steps_per_second)
        
        current_time = time.time()
        if current_time - self.last_log_time > 30 or self.total_steps % 100 == 0:
            info = self.locals["infos"][0]
            hit_rate = self.total_hits / (self.total_steps + 1e-6)
            self.logger.info(
                f"Step {self.total_steps}: "
                f"Reward={self.locals['rewards'][0]:.2f}, "
                f"Hits={self.total_hits}, "
                f"Hit Rate={hit_rate:.2%}, "
                f"HP={info.get('hp', 0)}"
            )
            self.last_log_time = current_time
        
        if self.progress_bar:
            self.pbar.update(1)
        return True

    def _on_rollout_end(self):
        self.episode_rewards.append(self.current_episode_reward)
        hit_rate = self.episode_hits / (self.total_steps + 1e-6)
        
        self.logger.record(" combat/episode_reward", self.current_episode_reward)
        self.logger.record("combat/episode_hits", self.episode_hits)
        self.logger.record("combat/episode_hit_rate", hit_rate)
        
        info = self.locals["infos"][0]
        action_percentages = info.get("action_percentages", np.zeros(9))
        action_names = info.get("action_names", [])
        for name, percentage in zip(action_names, action_percentages):
            self.logger.record(f"episode_actions/{name.replace('+', '_')}", percentage)
        
        self.logger.info(
            f"Episode completed: "
            f"Total Reward={self.current_episode_reward:.2f}, "
            f"Hits={self.episode_hits}, "
            f"Hit Rate={hit_rate:.2%}"
        )
        
        self.current_episode_reward = 0
        self.episode_hits = 0

    def _on_training_end(self):
        if self.progress_bar:
            self.pbar.close()
        training_duration = time.time() - self.start_time
        final_hit_rate = self.total_hits / (self.total_steps + 1e-6)
        self.logger.info(
            f"Training completed. Total steps: {self.total_steps}, "
            f"Total hits: {self.total_hits}, "
            f"Final hit rate: {final_hit_rate:.2%}, "
            f"Duration: {training_duration:.2f} seconds"
        )

def setup_logging(log_dir):
    # ... (unchanged)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()

def save_model_with_logging(model, path, logger):
    # ... (unchanged)
    logger.info(f"Starting model save to {path}")
    start_time = time.time()
    try:
        model.save(path)
        save_time = time.time() - start_time
        file_size = os.path.getsize(path + '.zip') / (1024*1024)
        logger.info(f"Model successfully saved in {save_time:.2f} seconds")
        logger.info(f"Model size: {file_size:.2f} MB")
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {str(e)}")
        raise

def signal_handler(sig, frame):
    # ... (unchanged)
    logger.info('\nReceived interrupt signal, saving model...')
    save_model_on_exit()

def save_model_on_exit():
    # ... (unchanged)
    global global_model, global_model_file, logger
    if global_model is not None and global_model_file is not None:
        try:
            logger.info("Emergency save initiated")
            save_model_with_logging(global_model, global_model_file, logger)
        except Exception as e:
            logger.error(f"Emergency save failed: {str(e)}")
    sys.exit(0)

def make_env(render=False):
    # ... (unchanged)
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    if render:
        env.render_mode = 'human'
    return KungFuWrapper(env)

def make_vec_env(num_envs, render=False):
    # ... (unchanged)
    logger.info(f"Creating vectorized environment with {num_envs} subprocesses")
    if num_envs > 1:
        env = SubprocVecEnv([lambda: make_env(render) for _ in range(num_envs)])
    else:
        env = DummyVecEnv([lambda: make_env(render)])
    return VecFrameStack(env, n_stack=4)

def objective(trial, args, env):
    global global_model, global_model_file, logger
    
    # Define hyperparameter search space
    combat_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 20),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-3, 0.1),
        'vf_coef': trial.suggest_uniform('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_uniform('max_grad_norm', 0.1, 1.0)
    }
    
    logger.info(f"Trial {trial.number} with params: {combat_params}")
    
    # Create or load model
    model_file = os.path.join(args.model_path, f"kungfu_ppo_trial_{trial.number}")
    if args.resume and os.path.exists(model_file + ".zip"):
        logger.info(f"Resuming model from {model_file}.zip")
        model = PPO.load(model_file, env=env, device="cuda" if args.cuda else "cpu")
    else:
        logger.info(f"Creating new model for trial {trial.number}")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={
                "features_extractor_class": SimpleCNN,
                "net_arch": [dict(pi=[128, 128], vf=[128, 128])]
            },
            **combat_params,
            tensorboard_log=args.log_dir,
            device="cuda" if args.cuda else "cpu"
        )
    
    global_model = model
    global_model_file = model_file
    
    # Callback for training
    callback = CombatTrainingCallback(progress_bar=args.progress_bar, logger=logger)
    if args.progress_bar:
        callback.pbar.total = args.timesteps
    
    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=f"PPO_KungFu_trial_{trial.number}",
        reset_num_timesteps=not args.resume  # Reset timesteps if not resuming
    )
    
    # Evaluate the model (total hits as the objective to maximize)
    total_hits = callback.total_hits
    logger.info(f"Trial {trial.number} completed with total_hits: {total_hits}")
    
    # Save the model for this trial
    save_model_with_logging(model, global_model_file, logger)
    
    return total_hits

def train(args):
    global logger
    logger = setup_logging(args.log_dir)
    logger.info("Starting training session with Optuna optimization")
    logger.info(f"Command line arguments: {vars(args)}")
    
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_model_on_exit)
    
    env = make_vec_env(args.num_envs, render=args.render)
    
    model_path = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    os.makedirs(model_path, exist_ok=True)
    
    # Optuna study storage
    study_name = "kungfu_ppo_study"
    storage_name = f"sqlite:///{os.path.join(args.log_dir, 'optuna_study.db')}"
    
    # Load or create study
    if args.resume and os.path.exists(os.path.join(args.log_dir, 'optuna_study.db')):
        logger.info(f"Resuming existing Optuna study from {storage_name}")
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    else:
        logger.info("Creating new Optuna study")
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
    
    n_trials = args.n_trials
    logger.info(f"Starting Optuna optimization with {n_trials} trials")
    study.optimize(lambda trial: objective(trial, args, env), n_trials=n_trials)
    
    # Log best trial
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value (total hits): {study.best_value}")
    
    # Save the best model
    best_model_file = os.path.join(model_path, "kungfu_ppo_best")
    global_model_file = best_model_file
    best_model = PPO.load(f"{model_path}/kungfu_ppo_trial_{study.best_trial.number}", env=env)
    save_model_with_logging(best_model, best_model_file, logger)
    
    env.close()
    logger.info("Training session ended")

def play(args):
    # ... (unchanged)
    global logger
    logger = setup_logging(args.log_dir)
    logger.info("Starting play session")
    
    model_file = os.path.join(args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path), "kungfu_ppo_best")
    if not os.path.exists(model_file + ".zip"):
        logger.error(f"No trained model found at {model_file}.zip")
        sys.exit(1)

    env = make_vec_env(1, render=True)
    model = PPO.load(model_file, env=env, device="cuda" if args.cuda else "cpu")

    obs = env.reset()
    total_reward = 0
    total_hits = 0
    done = False
    step_count = 0

    try:
        logger.info("Starting gameplay...")
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            total_hits += info[0]['enemy_hit']
            step_count += 1
            
            if step_count % 10 == 0 or info[0]['enemy_hit'] > 0:
                hit_rate = total_hits / step_count
                action_pct = env.envs[0].action_counts / (sum(env.envs[0].action_counts) + 1e-6)
                action_names = env.envs[0].action_names
                logger.info(
                    f"Step: {step_count}, "
                    f"Reward: {reward[0]:.2f}, "
                    f"Total: {total_reward:.2f}, "
                    f"Hits: {total_hits}, "
                    f"Hit Rate: {hit_rate:.2%}, "
                    f"HP: {info[0]['hp']}, "
                    f"Actions: {[f'{name}:{pct:.1%}' for name, pct in zip(action_names, action_pct)]}"
                )
                if info[0]['enemy_hit'] > 0:
                    print(">>> ENEMY HIT! <<<")
            
            if args.render:
                env.render()
    except KeyboardInterrupt:
        logger.info("\nPlay interrupted by user.")
    finally:
        hit_rate = total_hits / (step_count + 1e-6)
        logger.info(f"Play ended. Steps: {step_count}, Hits: {total_hits}, Hit Rate: {hit_rate:.2%}")
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Kung Fu with PPO and Optuna.")
    parser.add_argument("--train", action="store_true", help="Train the model with Optuna")
    parser.add_argument("--play", action="store_true", help="Play with trained model")
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps per trial")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--render", action="store_true", help="Render during training/play")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--n_trials", type=int, default=10, help="Number of Optuna trials")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved study and models")
    
    args = parser.parse_args()

    if args.train:
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        train(args)
    elif args.play:
        play(args)
    else:
        print("Please specify a mode: --train or --play")
        parser.print_help()