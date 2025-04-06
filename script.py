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
        combat_status_size = observation_space["combat_status"].shape[0]
        projectile_vec_size = observation_space["projectile_vector"].shape[0]  # Updated to projectile_vector
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + enemy_vec_size + combat_status_size + projectile_vec_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        combat_status = observations["combat_status"]
        projectile_vector = observations["projectile_vector"]  # Updated to projectile_vector
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
        if isinstance(combat_status, np.ndarray):
            combat_status = torch.from_numpy(combat_status)
        if isinstance(projectile_vector, np.ndarray):
            projectile_vector = torch.from_numpy(projectile_vector)
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and (viewport.shape[3] == 4 or viewport.shape[3] == 1):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        if len(enemy_vector.shape) == 1:
            enemy_vector = enemy_vector.unsqueeze(0)
        if len(combat_status.shape) == 1:
            combat_status = combat_status.unsqueeze(0)
        if len(projectile_vector.shape) == 1:
            projectile_vector = projectile_vector.unsqueeze(0)
            
        combined = torch.cat([cnn_output, enemy_vector, combat_status, projectile_vector], dim=1)
        return self.linear(combined)

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        
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
            "No-op", "Punch", "Kick", "Right+Punch", "Left+Punch",
            "Jump", "Crouch", "Jump+Punch", "Crouch+Punch"
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (8,), np.float32),
            "combat_status": spaces.Box(0, 1, (1,), np.float32),
            "projectile_vector": spaces.Box(-255, 255, (4,), np.float32)  # Updated to projectile_vector (x, y, dx, dy)
        })
        
        self.last_hp = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4
        self.max_steps = 1000
        self.current_step = 0
        self.total_steps = 0
        self.last_projectile_positions = []  # Track projectile positions for trajectory
        self.was_hit_by_projectile = False
        self.prev_frame = None  # For frame differencing

    def reset(self, **kwargs):
        self.current_step = 0
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4
        self.last_projectile_positions = []
        self.was_hit_by_projectile = False
        self.prev_frame = None
        return self._get_obs(obs)

    def step(self, action):
        self.current_step += 1
        self.total_steps += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = ram[0x04A6]
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        hp_loss = max(0, int(self.last_hp) - int(hp))
        
        # Detect projectiles using computer vision
        projectile_info = self._detect_projectiles(obs)
        projectile_hit = self._check_projectile_hit(hp_loss)
        projectile_avoided = len(projectile_info) > 0 and not projectile_hit
        
        # Adjusted reward structure
        raw_reward = (
            enemy_hit * 500.0 +
            -hp_loss * 50.0 +
            (1.0 if enemy_hit > 0 else -0.5) +
            (100.0 if projectile_avoided else 0.0) +  # Updated to projectile_avoided
            (-200.0 if projectile_hit else 0.0)  # Updated to projectile_hit
        )
        
        # Diversity penalty
        action_percentages = self.action_counts / (self.total_steps + 1e-6)
        dominant_action_percentage = np.max(action_percentages)
        if dominant_action_percentage > 0.4:
            diversity_penalty = -10.0 * (dominant_action_percentage - 0.4)
            raw_reward += diversity_penalty
        
        self.last_hp = hp
        self.last_enemies = curr_enemies
        self.was_hit_by_projectile = projectile_hit
        
        if self.current_step > 50 and enemy_hit == 0:
            done = True
            raw_reward -= 100.0
        
        normalized_reward = np.clip(raw_reward / 500.0, -1, 1)
        
        info.update({
            "hp": hp,
            "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "enemy_hit": enemy_hit,
            "projectile_hit": projectile_hit,  # Updated to projectile_hit
            "projectile_avoided": projectile_avoided,  # Updated to projectile_avoided
            "dominant_action_percentage": dominant_action_percentage
        })
        
        return self._get_obs(obs), normalized_reward, done, info

    def _detect_projectiles(self, obs):
        # Convert the observation (RGB) to grayscale
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        
        # Resize to match viewport size
        gray = cv2.resize(gray, self.viewport_size, interpolation=cv2.INTER_AREA)
        
        # Use frame differencing to detect moving objects
        if self.prev_frame is not None:
            # Compute the absolute difference between the current and previous frame
            frame_diff = cv2.absdiff(gray, self.prev_frame)
            
            # Apply a threshold to highlight significant changes (moving objects)
            _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)  # Adjust threshold as needed
            
            # Dilate to connect nearby changes (helps with small objects)
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours of moving objects
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            projectile_info = []
            current_projectile_positions = []
            hero_x = int(self.env.get_ram()[0x0094])  # Hero's X position from RAM
            
            for contour in contours:
                # Filter contours by size (projectiles are typically small)
                area = cv2.contourArea(contour)
                if 5 < area < 100:  # Adjusted range to account for various projectile sizes
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Calculate center of the projectile
                    proj_x = x + w // 2
                    proj_y = y + h // 2
                    
                    # Estimate trajectory by comparing with previous positions
                    dx, dy = 0, 0
                    for prev_pos in self.last_projectile_positions:
                        prev_x, prev_y = prev_pos
                        if abs(proj_x - prev_x) < 15 and abs(proj_y - prev_y) < 15:  # Likely the same projectile
                            dx = proj_x - prev_x
                            dy = proj_y - prev_y
                            # Filter by speed (projectiles move fast)
                            speed = np.sqrt(dx**2 + dy**2)
                            if speed < 3 or speed > 20:  # Adjust speed range for projectiles
                                dx, dy = 0, 0
                            break
                    
                    # Only consider projectiles with significant horizontal movement (typical for thrown objects)
                    if abs(dx) > 2:  # Projectiles typically move horizontally
                        # Calculate distance relative to hero
                        game_width = 256  # Typical NES game width
                        proj_x_game = (proj_x / self.viewport_size[0]) * game_width
                        distance = proj_x_game - hero_x
                        
                        projectile_info.extend([distance, proj_y, dx, dy])
                        current_projectile_positions.append((proj_x, proj_y))
            
            # Pad projectile_info to ensure consistent length (4 elements for 1 projectile: distance, y, dx, dy)
            while len(projectile_info) < 4:
                projectile_info.append(0)
            
            self.last_projectile_positions = current_projectile_positions[-2:]  # Track up to 2 projectiles
            self.prev_frame = gray  # Update previous frame
            return projectile_info[:4]
        
        self.prev_frame = gray  # Store current frame for next step
        return [0, 0, 0, 0]  # Return empty vector if no previous frame

    def _check_projectile_hit(self, hp_loss):
        if hp_loss > 0 and len(self.last_projectile_positions) > 0:
            return True
        return False

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
        projectile_vector = np.array(self._detect_projectiles(obs), dtype=np.float32)  # Updated to projectile_vector
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector,
            "combat_status": combat_status,
            "projectile_vector": projectile_vector  # Updated to projectile_vector
        }

class CombatTrainingCallback(BaseCallback):
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
        self.projectile_hits = 0  # Updated to projectile_hits
        self.projectile_avoids = 0  # Updated to projectile_avoids
        
        if progress_bar:
            self.pbar = tqdm(total=10000, desc="Training")

    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals["rewards"][0]
        hit_count = self.locals["infos"][0]["enemy_hit"]
        projectile_hit = self.locals["infos"][0]["projectile_hit"]  # Updated to projectile_hit
        projectile_avoided = self.locals["infos"][0]["projectile_avoided"]  # Updated to projectile_avoided
        self.episode_hits += hit_count
        self.total_hits += hit_count
        self.projectile_hits += projectile_hit
        self.projectile_avoids += projectile_avoided
        
        if self.num_timesteps % 100 == 0:
            info = self.locals["infos"][0]
            hit_rate = self.total_hits / (self.total_steps + 1e-6)
            
            self.logger.record("combat/step_reward", self.locals["rewards"][0])
            self.logger.record("combat/total_episode_reward", self.current_episode_reward)
            self.logger.record("combat/hit_count", hit_count)
            self.logger.record("combat/total_hits", self.total_hits)
            self.logger.record("combat/hit_rate", hit_rate)
            self.logger.record("game/hp", info.get("hp", 0))
            self.logger.record("game/dominant_action_percentage", info.get("dominant_action_percentage", 0))
            self.logger.record("combat/projectile_hits", self.projectile_hits)  # Updated to projectile_hits
            self.logger.record("combat/projectile_avoids", self.projectile_avoids)  # Updated to projectile_avoids
            
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
                f"HP={info.get('hp', 0)}, "
                f"Projectile Hits={self.projectile_hits}, "  # Updated to projectile_hits
                f"Projectile Avoids={self.projectile_avoids}, "  # Updated to projectile_avoids
                f"Dominant Action %={info.get('dominant_action_percentage', 0):.2%}"
            )
            self.last_log_time = current_time
        
        if self.progress_bar:
            self.pbar.update(1)
        return True

    def _on_rollout_end(self):
        self.episode_rewards.append(self.current_episode_reward)
        hit_rate = self.episode_hits / (self.total_steps + 1e-6)
        
        self.logger.record("combat/episode_reward", self.current_episode_reward)
        self.logger.record("combat/episode_hits", self.episode_hits)
        self.logger.record("combat/episode_hit_rate", hit_rate)
        self.logger.record("combat/episode_projectile_hits", self.projectile_hits)  # Updated to projectile_hits
        self.logger.record("combat/episode_projectile_avoids", self.projectile_avoids)  # Updated to projectile_avoids
        
        info = self.locals["infos"][0]
        action_percentages = info.get("action_percentages", np.zeros(9))
        action_names = info.get("action_names", [])
        for name, percentage in zip(action_names, action_percentages):
            self.logger.record(f"episode_actions/{name.replace('+', '_')}", percentage)
        
        self.logger.info(
            f"Episode completed: "
            f"Total Reward={self.current_episode_reward:.2f}, "
            f"Hits={self.episode_hits}, "
            f"Hit Rate={hit_rate:.2%}, "
            f"Projectile Hits={self.projectile_hits}, "  # Updated to projectile_hits
            f"Projectile Avoids={self.projectile_avoids}"  # Updated to projectile_avoids
        )
        
        self.current_episode_reward = 0
        self.episode_hits = 0
        self.projectile_hits = 0
        self.projectile_avoids = 0

    def _on_training_end(self):
        if self.progress_bar:
            self.pbar.close()
        training_duration = time.time() - self.start_time
        final_hit_rate = self.total_hits / (self.total_steps + 1e-6)
        self.logger.info(
            f"Training completed. Total steps: {self.total_steps}, "
            f"Total hits: {self.total_hits}, "
            f"Final hit rate: {final_hit_rate:.2%}, "
            f"Projectile Hits: {self.projectile_hits}, "  # Updated to projectile_hits
            f"Projectile Avoids: {self.projectile_avoids}, "  # Updated to projectile_avoids
            f"Duration: {training_duration:.2f} seconds"
        )

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger()

def save_model_with_logging(model, path, logger):
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
    if num_envs > 1:
        env = SubprocVecEnv([lambda: make_env(render) for _ in range(num_envs)])
    else:
        env = DummyVecEnv([lambda: make_env(render)])
    return VecFrameStack(env, n_stack=4)

def objective(trial, args, env):
    global global_model, global_model_file, logger
    
    combat_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 20),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-2, 1.0),
        'vf_coef': trial.suggest_uniform('vf_coef', 0.1, 1.0),
        'max_grad_norm': trial.suggest_uniform('max_grad_norm', 0.1, 1.0)
    }
    
    logger.info(f"Trial {trial.number} with params: {combat_params}")
    
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
    
    callback = CombatTrainingCallback(progress_bar=args.progress_bar, logger=logger)
    if args.progress_bar:
        callback.pbar.total = args.timesteps
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name=f"PPO_KungFu_trial_{trial.number}",
        reset_num_timesteps=not args.resume
    )
    
    total_hits = callback.total_hits
    logger.info(f"Trial {trial.number} completed with total_hits: {total_hits}")
    
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
    
    study_name = "kungfu_ppo_study"
    storage_name = f"sqlite:///{os.path.join(args.log_dir, 'optuna_study.db')}"
    
    if args.resume and os.path.exists(os.path.join(args.log_dir, 'optuna_study.db')):
        logger.info(f"Resuming existing Optuna study from {storage_name}")
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    else:
        logger.info("Creating new Optuna study")
        study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
    
    n_trials = args.n_trials
    logger.info(f"Starting Optuna optimization with {n_trials} trials")
    study.optimize(lambda trial: objective(trial, args, env), n_trials=n_trials)
    
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best parameters: {study.best_params}")
    logger.info(f"Best value (total hits): {study.best_value}")
    
    best_model_file = os.path.join(model_path, "kungfu_ppo_best")
    global_model_file = best_model_file
    best_model = PPO.load(f"{model_path}/kungfu_ppo_trial_{study.best_trial.number}", env=env)
    save_model_with_logging(best_model, best_model_file, logger)
    
    env.close()
    logger.info("Training session ended")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kung Fu with PPO and Optuna.")
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps per trial")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of Optuna trials")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved study and models")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    train(args)