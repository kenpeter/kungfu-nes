import argparse
import os
import retro
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import logging
import sys
from gym import spaces, Wrapper
import cv2
from tqdm import tqdm
import optuna

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super(SimpleCNN, self).__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict"
        
        self.cnn = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, 36, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        enemy_types_size = observation_space["enemy_types"].shape[0]
        enemy_history_size = observation_space["enemy_history"].shape[0]
        enemy_timers_size = observation_space["enemy_timers"].shape[0]
        enemy_patterns_size = observation_space["enemy_patterns"].shape[0]
        combat_status_size = observation_space["combat_status"].shape[0]
        projectile_vec_size = observation_space["projectile_vectors"].shape[0]
        enemy_proximity_size = observation_space["enemy_proximity"].shape[0]
        boss_info_size = observation_space["boss_info"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten + enemy_vec_size + enemy_types_size + enemy_history_size +
                enemy_timers_size + enemy_patterns_size + combat_status_size +
                projectile_vec_size + enemy_proximity_size + boss_info_size,
                512
            ),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        enemy_types = observations["enemy_types"]
        enemy_history = observations["enemy_history"]
        enemy_timers = observations["enemy_timers"]
        enemy_patterns = observations["enemy_patterns"]
        combat_status = observations["combat_status"]
        projectile_vectors = observations["projectile_vectors"]
        enemy_proximity = observations["enemy_proximity"]
        boss_info = observations["boss_info"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport).float()
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector).float()
        if isinstance(enemy_types, np.ndarray):
            enemy_types = torch.from_numpy(enemy_types).float()
        if isinstance(enemy_history, np.ndarray):
            enemy_history = torch.from_numpy(enemy_history).float()
        if isinstance(enemy_timers, np.ndarray):
            enemy_timers = torch.from_numpy(enemy_timers).float()
        if isinstance(enemy_patterns, np.ndarray):
            enemy_patterns = torch.from_numpy(enemy_patterns).float()
        if isinstance(combat_status, np.ndarray):
            combat_status = torch.from_numpy(combat_status).float()
        if isinstance(projectile_vectors, np.ndarray):
            projectile_vectors = torch.from_numpy(projectile_vectors).float()
        if isinstance(enemy_proximity, np.ndarray):
            enemy_proximity = torch.from_numpy(enemy_proximity).float()
        if isinstance(boss_info, np.ndarray):
            boss_info = torch.from_numpy(boss_info).float()
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and viewport.shape[-1] in (3, 36):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        for tensor in [enemy_vector, enemy_types, enemy_history, enemy_timers,
                       enemy_patterns, combat_status, projectile_vectors,
                       enemy_proximity, boss_info]:
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            
        combined = torch.cat([
            cnn_output, enemy_vector, enemy_types, enemy_history, enemy_timers,
            enemy_patterns, combat_status, projectile_vectors, enemy_proximity, boss_info
        ], dim=1)
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
            [0,0,1,0,0,0,0,0,0,0,0,0],  # Crouch
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Jump
            [0,0,0,0,0,1,1,0,0,0,0,0],  # Jump+Punch
            [0,0,1,0,0,0,1,0,0,0,0,0]   # Crouch+Punch
        ]
        self.action_names = [
            "No-op", "Punch", "Kick", "Right+Punch", "Left+Punch",
            "Crouch", "Jump", "Jump+Punch", "Crouch+Punch"
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_enemies = 4
        self.history_length = 5
        self.max_projectiles = 2
        self.patterns_length = 2
        
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 3), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (self.max_enemies * 2,), np.float32),
            "enemy_types": spaces.Box(0, 5, (self.max_enemies,), np.float32),
            "enemy_history": spaces.Box(-255, 255, (self.max_enemies * self.history_length * 2,), np.float32),
            "enemy_timers": spaces.Box(0, 1, (self.max_enemies,), np.float32),
            "enemy_patterns": spaces.Box(-255, 255, (self.max_enemies * self.patterns_length,), np.float32),
            "combat_status": spaces.Box(-1, 1, (2,), np.float32),
            "projectile_vectors": spaces.Box(-255, 255, (self.max_projectiles * 4,), np.float32),
            "enemy_proximity": spaces.Box(0, 1, (1,), np.float32),
            "boss_info": spaces.Box(-255, 255, (3,), np.float32)
        })
        
        self.last_hp = 0
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        self.last_projectile_positions = []
        self.last_projectile_distances = [float('inf')] * self.max_projectiles

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = float(self.env.get_ram()[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        self.last_projectile_positions = []
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        return self._get_obs(obs)

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = float(ram[0x04A6])
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        hp_change_rate = (hp - self.last_hp) / 255.0
        
        self._update_enemy_info(obs, ram)
        self._update_boss_info(ram)
        projectile_info = self._detect_projectiles(obs)
        
        reward = 0
        reward += enemy_hit * 10
        reward += hp_change_rate * 5
        if done:
            reward -= 50
        
        projectile_distances = [projectile_info[i] for i in range(0, len(projectile_info), 4)]
        dodge_reward = 0
        for i, (curr_dist, last_dist) in enumerate(zip(projectile_distances, self.last_projectile_distances)):
            if last_dist < 20 and curr_dist > last_dist:
                if action == 5:  # Crouch
                    dodge_reward += 3
                elif action == 6:  # Jump
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances
        
        action_entropy = -np.sum((self.action_counts / (self.total_steps + 1e-6)) * 
                                 np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6))
        reward += action_entropy * 0.1
        
        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward
        })
        
        return self._get_obs(obs), reward, done, info

    def _update_enemy_info(self, obs, ram):
        hero_x = int(ram[0x0094])
        new_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        new_types = np.zeros(self.max_enemies, dtype=np.float32)
        new_timers = np.zeros(self.max_enemies, dtype=np.float32)
        
        stage = int(ram[0x0058])
        for i, (pos_addr, action_addr, timer_addr) in enumerate([
            (0x008E, 0x0080, 0x002B),
            (0x008F, 0x0081, 0x002C),
            (0x0090, 0x0082, 0x002D),
            (0x0091, 0x0083, 0x002E)
        ]):
            enemy_x = int(ram[pos_addr])
            enemy_action = int(ram[action_addr])
            enemy_timer = int(ram[timer_addr])
            if enemy_x != 0:
                if stage == 1:
                    new_types[i] = 1 if enemy_action in [0x01, 0x02] else 2
                elif stage == 2:
                    new_types[i] = 3
                elif stage == 3:
                    new_types[i] = 4 if enemy_action in [0x03, 0x04] else 2
                elif stage in [4, 5]:
                    new_types[i] = 5 if stage == 5 else 0
                new_positions[i] = [enemy_x, 50]
                new_timers[i] = min(enemy_timer / 60.0, 1.0)
        
        for i in range(self.max_enemies):
            if new_types[i] != 0:
                dx = new_positions[i][0] - self.enemy_positions[i][0]
                dy = new_positions[i][1] - self.enemy_positions[i][1]
                self.enemy_history[i, :-1] = self.enemy_history[i, 1:]
                self.enemy_history[i, -1] = [dx, dy]
            else:
                self.enemy_history[i] = np.zeros((self.history_length, 2), dtype=np.float32)
        
        self.enemy_positions = new_positions
        self.enemy_types = new_types
        self.enemy_timers = new_timers

    def _update_boss_info(self, ram):
        stage = int(ram[0x0058])
        if stage == 5:
            boss_pos_x = int(ram[0x0093])
            boss_action = int(ram[0x004E])
            boss_hp = int(ram[0x04A5])
            self.boss_info = np.array([boss_pos_x - int(ram[0x0094]), boss_action / 255.0, boss_hp / 255.0], dtype=np.float32)
        else:
            self.boss_info = np.zeros(3, dtype=np.float32)

    def _detect_projectiles(self, obs):
        frame = cv2.resize(obs, self.viewport_size, interpolation=cv2.INTER_AREA)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(frame, self.prev_frame)
            diff_sum = np.sum(frame_diff, axis=2).astype(np.uint8)
            _, motion_mask = cv2.threshold(diff_sum, 20, 255, cv2.THRESH_BINARY)
            motion_mask = cv2.dilate(motion_mask, None, iterations=4)
            
            lower_white = np.array([180, 180, 180])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(frame, lower_white, upper_white)
            color_mask = cv2.dilate(white_mask, None, iterations=2)
            
            combined_mask = cv2.bitwise_and(motion_mask, color_mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            projectile_info = []
            current_projectile_positions = []
            hero_x = int(self.env.get_ram()[0x0094])
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    proj_x = x + w // 2
                    proj_y = y + h // 2
                    aspect_ratio = w / h if h > 0 else 1
                    if 0.5 < aspect_ratio < 3.0:
                        dx = 0
                        for prev_pos in self.last_projectile_positions:
                            prev_x, _ = prev_pos
                            if abs(proj_x - prev_x) < 20:
                                dx = proj_x - prev_x
                                break
                        if dx != 0:
                            game_width = 256
                            proj_x_game = (proj_x / self.viewport_size[0]) * game_width
                            distance = proj_x_game - hero_x
                            projectile_info.extend([distance, proj_y, dx, 0])
                            current_projectile_positions.append((proj_x, proj_y))
            
            projectile_info = projectile_info[:self.max_projectiles * 4]
            while len(projectile_info) < self.max_projectiles * 4:
                projectile_info.append(0)
            
            self.last_projectile_positions = current_projectile_positions[:self.max_projectiles]
            self.prev_frame = frame
            return projectile_info
        
        self.prev_frame = frame
        return [0] * (self.max_projectiles * 4)

    def _get_obs(self, obs):
        viewport = cv2.resize(obs, self.viewport_size)
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
        enemy_types = self.enemy_types
        enemy_history = self.enemy_history.reshape(-1)
        enemy_timers = self.enemy_timers
        enemy_patterns = self.enemy_patterns.reshape(-1)
        combat_status = np.array([self.last_hp/255.0, self.last_hp_change], dtype=np.float32)
        projectile_vectors = np.array(self._detect_projectiles(obs), dtype=np.float32)
        enemy_proximity = np.array([1.0 if any(abs(enemy_x - hero_x) <= 20 for enemy_x in [int(ram[addr]) for addr in [0x008E, 0x008F, 0x0090, 0x0091]] if enemy_x != 0) else 0.0], dtype=np.float32)
        boss_info = self.boss_info
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector,
            "enemy_types": enemy_types,
            "enemy_history": enemy_history,
            "enemy_timers": enemy_timers,
            "enemy_patterns": enemy_patterns,
            "combat_status": combat_status,
            "projectile_vectors": projectile_vectors,
            "enemy_proximity": enemy_proximity,
            "boss_info": boss_info
        }

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_hits = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        total_hits = sum([info.get('enemy_hit', 0) for info in infos])
        
        if total_hits > self.best_hits:
            self.best_hits = total_hits
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Saved best model with {self.best_hits} enemy hits at step {self.num_timesteps}")
        return True

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger()

def make_env():
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuWrapper(env)
    return env

def train(args):
    logger = setup_logging(args.log_dir)
    logger.info("Starting training session")

    # Common environment and policy setup
    env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
    env = VecFrameStack(env, n_stack=12)
    
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "net_arch": dict(
            pi=[256, 256, 128],
            vf=[512, 512, 256]
        )
    }

    # Optuna study configuration
    study_name = "kungfu_ppo_study"
    storage_name = f"sqlite:///{args.log_dir}/optuna_study.db"

    # Function to load or create study
    def get_study():
        try:
            study = optuna.load_study(study_name=study_name, storage=storage_name)
            logger.info(f"Loaded existing study '{study_name}' from database with {len(study.trials)} trials")
            return study
        except KeyError:
            logger.info(f"Study '{study_name}' not found. Creating new study.")
            return optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")

    # Function to get best hyperparameters with fallback to defaults
    def get_best_params(study=None):
        default_params = {
            "learning_rate": args.learning_rate,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10
        }
        if study and len(study.trials) > 0 and study.best_trial is not None:
            logger.info(f"Using best parameters from study with value {study.best_value}")
            return study.best_params
        logger.info("No previous trials found in study, using defaults")
        return default_params

    # Optuna objective function
    def objective(trial):
        params = {
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-2),
            "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.3),
            "ent_coef": trial.suggest_loguniform("ent_coef", 1e-8, 1e-1),
            "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
            "n_epochs": trial.suggest_int("n_epochs", 3, 10)
        }

        trial_env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
        trial_env = VecFrameStack(trial_env, n_stack=12)

        model = PPO(
            "MultiInputPolicy",
            trial_env,
            **params,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=0,
            tensorboard_log=args.tensorboard_log,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )

        callback = SaveBestModelCallback(save_path=f"{args.model_path}_trial_{trial.number}")
        model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)

        total_hits = callback.best_hits
        logger.info(f"Trial {trial.number} finished with {total_hits} enemy hits")
        trial_env.close()

        return total_hits

    # Load or create study
    study = get_study()
    best_params = get_best_params(study)

    # Check if resuming or starting fresh
    if args.resume and os.path.exists(args.model_path + ".zip"):
        logger.info(f"Resuming training from {args.model_path}")
        model = PPO.load(args.model_path, env=env, device="cuda" if args.cuda else "cpu")
        # Update model with best params from DB
        model.learning_rate = best_params["learning_rate"]
        model.clip_range = best_params["clip_range"]
        model.ent_coef = best_params["ent_coef"]
        model.n_steps = best_params["n_steps"]
        model.batch_size = best_params["batch_size"]
        model.n_epochs = best_params["n_epochs"]
        model.tensorboard_log = args.tensorboard_log
    else:
        logger.info("Starting new training session")
        # Run Optuna optimization with specified trials
        logger.info(f"Running Optuna optimization with {args.n_trials} trials")
        study.optimize(objective, n_trials=args.n_trials)
        best_params = get_best_params(study)  # Update best_params after optimization
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value (enemy hits): {study.best_value}")
        logger.info(f"Best hyperparameters: {study.best_params}")

        # Create new model with best params
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=best_params["learning_rate"],
            n_steps=best_params["n_steps"],
            batch_size=best_params["batch_size"],
            n_epochs=best_params["n_epochs"],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=best_params["clip_range"],
            ent_coef=best_params["ent_coef"],
            verbose=1,
            tensorboard_log=args.tensorboard_log,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )

    # Train the model
    callback = SaveBestModelCallback(save_path=args.model_path)
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)
    model.save(args.model_path)
    logger.info(f"Training completed. Model saved at {args.model_path}")

    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO model for Kung Fu with Optuna optimization")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo_best", help="Path to save the trained model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Default learning rate for PPO")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Default clip range for PPO")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Default entropy coefficient for PPO")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    parser.add_argument("--tensorboard_log", default="tensorboard_logs", help="Directory for TensorBoard logs")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs and Optuna database")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of Optuna trials for hyperparameter tuning")
    
    args = parser.parse_args()
    train(args)