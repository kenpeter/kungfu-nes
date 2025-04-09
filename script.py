import argparse
import os
import retro
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
import logging
import sys
from gym import spaces, Wrapper
import cv2
import signal
import time
import platform
import multiprocessing as mp
import zipfile
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Global variables for the model
current_model = None
global_logger = None
global_model_path = None
experience_data = []  # To store collected experience

def emergency_save_handler(signum, frame):
    global current_model, global_logger, global_model_path, experience_data
    if current_model is not None and global_model_path is not None:
        # Save model with experience data
        current_model.save(global_model_path)
        
        # Log the amount of experience collected
        experience_count = len(experience_data)
        if global_logger is not None:
            global_logger.info(f"Emergency save triggered by Ctrl+C. Model saved at {global_model_path}")
            global_logger.info(f"Collected experience: {experience_count} steps")
        else:
            print(f"Emergency save triggered by Ctrl+C. Model saved at {global_model_path}")
            print(f"Collected experience: {experience_count} steps")
        
        # Save experience data to file
        try:
            with open(f"{global_model_path}_experience_count.txt", "w") as f:
                f.write(f"Total experience collected: {experience_count} steps\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            if global_logger is not None:
                global_logger.error(f"Failed to save experience count: {e}")
            else:
                print(f"Failed to save experience count: {e}")
        
        # Clean up environment
        if hasattr(current_model, 'env'):
            current_model.env.close()
            if global_logger is not None:
                global_logger.info("Environment closed during emergency save.")
        
        # Clean up GPU
        if args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if global_logger is not None:
                global_logger.info("GPU cleaned up during emergency save.")
        
        del current_model
        current_model = None
    sys.exit(0)

signal.signal(signal.SIGINT, emergency_save_handler)


class ExperienceCollectionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(ExperienceCollectionCallback, self).__init__(verbose)
        self.experience_data = []
        
    def _on_step(self) -> bool:
        # Get the most recent observations, actions, rewards, etc.
        obs = self.locals.get('new_obs')
        actions = self.locals.get('actions')
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        if all(x is not None for x in [obs, actions, rewards, dones, infos]):
            # For vectorized environments, we need to handle the structure differently
            if isinstance(obs, dict):
                # For dict observations (like in your case)
                experience = {
                    "observation": obs,  # Store the entire observation dictionary
                    "action": actions,   # Store all actions
                    "reward": rewards,   # Store all rewards
                    "done": dones,       # Store all done flags
                    "info": infos        # Store all info dictionaries
                }
                self.experience_data.append(experience)
            else:
                # For array-like observations, we can iterate
                for i in range(len(rewards)):  # For each environment
                    if i < len(obs):  # Make sure the index is valid
                        experience = {
                            "observation": obs[i] if isinstance(obs, (list, np.ndarray)) else obs,
                            "action": actions[i] if isinstance(actions, (list, np.ndarray)) else actions,
                            "reward": rewards[i] if isinstance(rewards, (list, np.ndarray)) else rewards,
                            "done": dones[i] if isinstance(dones, (list, np.ndarray)) else dones,
                            "info": infos[i] if i < len(infos) else {}
                        }
                        self.experience_data.append(experience)
        
        return True
    
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
        enemy_attack_states_size = observation_space["enemy_attack_states"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten + enemy_vec_size + enemy_types_size + enemy_history_size +
                enemy_timers_size + enemy_patterns_size + combat_status_size +
                projectile_vec_size + enemy_proximity_size + boss_info_size +
                enemy_attack_states_size,
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
        enemy_attack_states = observations["enemy_attack_states"]
        
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
        if isinstance(enemy_attack_states, np.ndarray):
            enemy_attack_states = torch.from_numpy(enemy_attack_states).float()
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and viewport.shape[-1] in (3, 36):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        for tensor in [enemy_vector, enemy_types, enemy_history, enemy_timers,
                       enemy_patterns, combat_status, projectile_vectors,
                       enemy_proximity, boss_info, enemy_attack_states]:
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
            
        combined = torch.cat([
            cnn_output, enemy_vector, enemy_types, enemy_history, enemy_timers,
            enemy_patterns, combat_status, projectile_vectors, enemy_proximity,
            boss_info, enemy_attack_states
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
        self.history_length = 10
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
            "boss_info": spaces.Box(-255, 255, (3,), np.float32),
            "enemy_attack_states": spaces.Box(0, 1, (self.max_enemies,), np.float32)
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
        self.enemy_attack_states = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        self.last_projectile_positions = []
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        self.survival_reward_total = 0

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
        self.enemy_attack_states = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        self.last_projectile_positions = []
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        self.survival_reward_total = 0
        return self._get_obs(obs)

    def step(self, action):
        global experience_data
        self.total_steps += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = float(ram[0x04A6])
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)

        reward = 0
        hp_change_rate = (hp - self.last_hp) / 255.0
        
        if hp_change_rate < 0:
            reward += (hp_change_rate ** 2) * 500
        else:
            reward += hp_change_rate * 5

        reward += enemy_hit * 10
        reward += hp_change_rate * 5
        if done:
            reward -= 50

        self._update_enemy_info(obs, ram)
        self._update_boss_info(ram)
        projectile_info = self._detect_projectiles(obs)

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
        
        if hp_change_rate < 0 and action in [5, 6]:
            reward += 5

        action_entropy = -np.sum((self.action_counts / (self.total_steps + 1e-6)) * 
                                 np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6))
        reward += action_entropy * 1.0
        
        if not done and hp > 0:
            reward += 0.1
            self.survival_reward_total += 0.1

        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward,
            "survival_reward_total": self.survival_reward_total
        })
        
        # Collect experience
        experience = {
            "observation": self._get_obs(obs),
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        }
        experience_data.append(experience)
        
        return self._get_obs(obs), reward, done, info

    def _update_enemy_info(self, obs, ram):
        hero_x = int(ram[0x0094])
        new_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        new_types = np.zeros(self.max_enemies, dtype=np.float32)
        new_timers = np.zeros(self.max_enemies, dtype=np.float32)
        new_attack_states = np.zeros(self.max_enemies, dtype=np.float32)
        
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
                new_attack_states[i] = 1.0 if enemy_action in [0x01, 0x02] else 0.0
        
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
        self.enemy_attack_states = new_attack_states

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
        enemy_attack_states = self.enemy_attack_states
        
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
            "boss_info": boss_info,
            "enemy_attack_states": enemy_attack_states
        }

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_score = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        total_hits = sum([info.get('enemy_hit', 0) for info in infos])
        total_hp = sum([info.get('hp', 0) for info in infos])
        score = total_hits * 10 + total_hp / 255.0
        
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Saved best model with score {self.best_score} (hits: {total_hits}, HP: {total_hp}) at step {self.num_timesteps}")
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
    global current_model, global_logger, global_model_path, experience_data
    
    # Reset experience data
    experience_data = []
    
    # Setup logging
    global_logger = setup_logging(args.log_dir)
    global_logger.info(f"Starting training with {args.num_envs} envs and {args.timesteps} timesteps")
    global_model_path = args.model_path
    current_model = None

    # Validate num_envs
    if args.num_envs < 1:
        raise ValueError("Number of environments must be at least 1")
    
    # Create environments
    env_fns = [make_env for _ in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=12)
    
    # Policy setup
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "net_arch": dict(pi=[256, 256, 128], vf=[512, 512, 256])
    }
    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)

    # Default parameters
    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": 0.1,
        "n_steps": min(2048, args.timesteps // args.num_envs if args.num_envs > 0 else args.timesteps),
        "batch_size": 64,
        "n_epochs": 10
    }

    # Training logic
    if args.resume and os.path.exists(args.model_path + ".zip"):
        global_logger.info(f"Resuming training from {args.model_path}")
        # Load the old model just to get its weights
        old_model = PPO.load(args.model_path, device="cuda" if args.cuda else "cpu")
        
        # Create a new model with the desired parameters
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )
        
        # Copy the weights from the old model to the new one
        model.policy.load_state_dict(old_model.policy.state_dict())
        
        current_model = model
        global_logger.info(f"Created new model with n_steps={params['n_steps']} and copied weights from saved model")
    else:
        global_logger.info("Starting new training session")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=params["learning_rate"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )
        current_model = model

    # Train the model
    save_callback = SaveBestModelCallback(save_path=args.model_path)
    exp_callback = ExperienceCollectionCallback()
    # Use both callbacks
    callback = CallbackList([save_callback, exp_callback])
    
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)
    # Update the global experience_data from the callback
    experience_data = exp_callback.experience_data
    
    # Log the amount of experience collected at the end of training
    experience_count = len(experience_data)
    global_logger.info(f"Training completed. Total experience collected: {experience_count} steps")
    
    # Save experience count to file
    try:
        with open(f"{args.model_path}_experience_count.txt", "w") as f:
            f.write(f"Total experience collected: {experience_count} steps\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Training parameters: num_envs={args.num_envs}, timesteps={args.timesteps}\n")
    except Exception as e:
        global_logger.error(f"Failed to save experience count: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO model for Kung Fu")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo_best", help="Path to save the trained model")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="Default learning rate for PPO")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Default clip range for PPO")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Default entropy coefficient for PPO")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments (1 or more)")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    
    args = parser.parse_args()
    train(args)