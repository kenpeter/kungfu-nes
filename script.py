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
import sqlite3
from multiprocessing import set_start_method

if sys.platform == "win32":
    set_start_method("spawn", force=True)

# Global variables
global_model = None
global_model_file = None
logger = None

# Thresholds
GOOD_ENOUGH_THRESHOLD = 100
MAX_TIMESTEPS = 90000

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
        if len(viewport.shape) == 4 and viewport.shape[-1] == 3:
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
    def __init__(self, env, patterns_db="enemy_patterns.db"):
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
        self.enemy_types_map = {0: "None", 1: "Knife Thrower", 2: "Midget", 3: "Snake", 4: "Dragon", 5: "Boss"}
        self.max_enemy_types = len(self.enemy_types_map)
        self.max_enemies = 4
        self.history_length = 5
        self.max_projectiles = 2
        self.patterns_length = 2
        
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 3), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (self.max_enemies * 2,), np.float32),
            "enemy_types": spaces.Box(0, self.max_enemy_types - 1, (self.max_enemies,), np.float32),
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
        self.max_steps = 1000
        self.current_step = 0
        self.total_steps = 0
        self.last_projectile_positions = []
        self.was_hit_by_projectile = False
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        
        self.patterns_db = patterns_db
        self._init_db()
        self.stored_patterns = self._load_patterns()
        self.attack_intervals = {et: [] for et in range(self.max_enemy_types)}
        self.dx_history = {et: [] for et in range(self.max_enemy_types)}
        self.last_attack_steps = [-1] * self.max_enemies
        self.observation_counts = {et: 0 for et in range(self.max_enemy_types)}

    def _init_db(self):
        conn = sqlite3.connect(self.patterns_db)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS enemy_patterns (
                enemy_type INTEGER PRIMARY KEY,
                attack_interval REAL,
                avg_dx REAL,
                observations INTEGER
            )
        ''')
        for et in range(self.max_enemy_types):
            c.execute('INSERT OR IGNORE INTO enemy_patterns (enemy_type, attack_interval, avg_dx, observations) VALUES (?, 0, 0, 0)', (et,))
        conn.commit()
        conn.close()

    def _load_patterns(self):
        conn = sqlite3.connect(self.patterns_db)
        c = conn.cursor()
        c.execute('SELECT enemy_type, attack_interval, avg_dx, observations FROM enemy_patterns')
        patterns = {str(row[0]): {"attack_interval": row[1], "avg_dx": row[2], "observations": row[3]} for row in c.fetchall()}
        conn.close()
        return patterns

    def _save_patterns(self):
        conn = sqlite3.connect(self.patterns_db)
        c = conn.cursor()
        for et, data in self.stored_patterns.items():
            c.execute('''
                UPDATE enemy_patterns
                SET attack_interval = ?, avg_dx = ?, observations = ?
                WHERE enemy_type = ?
            ''', (data["attack_interval"], data["avg_dx"], data["observations"], int(et)))
        conn.commit()
        conn.close()

    def reset(self, **kwargs):
        self.current_step = 0
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.last_projectile_positions = []
        self.was_hit_by_projectile = False
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        self.last_attack_steps = [-1] * self.max_enemies
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
        hp_change_rate = (int(hp) - int(self.last_hp)) / 255.0
        
        hero_x = int(ram[0x0094])
        enemy_distances = [abs(enemy_x - hero_x) for enemy_x in curr_enemies if enemy_x != 0]
        enemy_very_close = 1.0 if any(dist <= 20 for dist in enemy_distances) else 0.0
        
        self._update_enemy_info(obs, ram)
        self._update_boss_info(ram)
        projectile_info = self._detect_projectiles(obs)
        projectile_hit = self._check_projectile_hit(hp_loss)
        projectile_avoided = len(projectile_info) > 0 and not projectile_hit
        
        raw_reward = (
            enemy_hit * 10.0 +
            -hp_loss * 2.0 +
            (1.0 if enemy_hit > 0 else -0.1) +
            (5.0 if projectile_avoided else 0.0) +
            (-10.0 if projectile_hit else 0.0)
        )
        
        if len(projectile_info) > 0:
            if action in [5, 6, 7, 8]:
                raw_reward += 1.0
            for i in range(0, len(projectile_info), 4):
                distance = projectile_info[i]
                proj_y = projectile_info[i + 1]
                if abs(distance) < 50 and 40 < proj_y < 60:
                    if action in [5, 7]:
                        raw_reward += 2.0
                    elif action in [6, 8]:
                        raw_reward += 2.0
        
        if action == 0:
            if hp_loss > 5:
                raw_reward -= 1.0
            if enemy_very_close:
                raw_reward -= 0.8
            if len(projectile_info) > 0:
                raw_reward -= 0.5
        
        action_percentages = self.action_counts / (self.total_steps + 1e-6)
        dominant_action_percentage = np.max(action_percentages)
        diversity_penalty = 0.0
        if self.current_step > 50 and hp_loss < 5:
            if dominant_action_percentage > 0.5:
                diversity_penalty = -0.5 * (dominant_action_percentage - 0.5)
                raw_reward += diversity_penalty
        
        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        self.was_hit_by_projectile = projectile_hit
        
        if self.current_step > 50 and enemy_hit == 0:
            done = True
            raw_reward -= 2.0
        
        normalized_reward = np.clip(raw_reward, -2, 2)
        
        info.update({
            "hp": hp,
            "hp_change_rate": hp_change_rate,
            "raw_reward": raw_reward,
            "normalized_reward": normalized_reward,
            "action_percentages": action_percentages,
            "action_names": self.action_names,
            "enemy_hit": enemy_hit,
            "projectile_hit": projectile_hit,
            "projectile_avoided": projectile_avoided,
            "dominant_action_percentage": dominant_action_percentage,
            "enemy_very_close": enemy_very_close,
            "enemy_types": self.enemy_types,
            "enemy_timers": self.enemy_timers,
            "enemy_patterns": self.enemy_patterns,
            "boss_info": self.boss_info
        })
        
        return self._get_obs(obs), normalized_reward, done, info

    def _update_enemy_info(self, obs, ram):
        hero_x = int(ram[0x0094])
        new_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        new_types = np.zeros(self.max_enemies, dtype=np.float32)
        new_timers = np.zeros(self.max_enemies, dtype=np.float32)
        
        stage = int(ram[0x0058])
        logger.debug(f"Current stage: {stage}")
        
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
                    if enemy_action in [0x01, 0x02]:
                        new_types[i] = 1
                    else:
                        new_types[i] = 2
                elif stage == 2:
                    new_types[i] = 3
                elif stage == 3:
                    if enemy_action in [0x03, 0x04]:
                        new_types[i] = 4
                    else:
                        new_types[i] = 2
                elif stage in [4, 5]:
                    new_types[i] = 5 if stage == 5 else 0
                new_positions[i] = [enemy_x, 50]
                new_timers[i] = min(enemy_timer / 60.0, 1.0)
                
                if enemy_action in [0x01, 0x02, 0x03, 0x04]:
                    if self.last_attack_steps[i] != -1:
                        interval = self.current_step - self.last_attack_steps[i]
                        if interval > 0:
                            self.attack_intervals[int(new_types[i])].append(interval)
                    self.last_attack_steps[i] = self.current_step
                else:
                    self.last_attack_steps[i] = -1
                
                dx = new_positions[i][0] - self.enemy_positions[i][0]
                if dx != 0:
                    self.dx_history[int(new_types[i])].append(dx)
                
                self.observation_counts[int(new_types[i])] += 1
            else:
                new_types[i] = 0
                new_timers[i] = 0
                self.last_attack_steps[i] = -1
        
        logger.debug(f"Enemy types: {new_types}")
        
        for i in range(self.max_enemies):
            if new_types[i] != 0:
                dx = new_positions[i][0] - self.enemy_positions[i][0]
                dy = new_positions[i][1] - self.enemy_positions[i][1]
                self.enemy_history[i, :-1] = self.enemy_history[i, 1:]
                self.enemy_history[i, -1] = [dx, dy]
            else:
                self.enemy_history[i] = np.zeros((self.history_length, 2), dtype=np.float32)
        
        if self.current_step % 50 == 0:
            for et in range(self.max_enemy_types):
                if et in [0, 5]:
                    continue
                current_data = self.stored_patterns.get(str(et), {"attack_interval": 0, "avg_dx": 0, "observations": 0})
                current_interval = current_data["attack_interval"]
                current_dx = current_data["avg_dx"]
                current_obs = current_data["observations"]
                
                if len(self.attack_intervals[et]) > 0:
                    new_intervals = [i for i in self.attack_intervals[et] if i > 0]
                    if new_intervals:
                        total_obs = current_obs + len(new_intervals)
                        avg_interval = np.mean(new_intervals)
                        self.stored_patterns[str(et)]["attack_interval"] = avg_interval
                        self.stored_patterns[str(et)]["observations"] = total_obs
                    self.attack_intervals[et] = []
                
                if len(self.dx_history[et]) > 0:
                    new_dx = [dx for dx in self.dx_history[et] if dx != 0]
                    if new_dx:
                        total_obs = current_obs + len(new_dx)
                        avg_dx = np.mean(new_dx)
                        self.stored_patterns[str(et)]["avg_dx"] = avg_dx
                        self.stored_patterns[str(et)]["observations"] = total_obs
                    self.dx_history[et] = []
                
                self.stored_patterns[str(et)]["observations"] = max(current_obs, self.observation_counts[et])
            
            self._save_patterns()
        
        for i in range(self.max_enemies):
            et = int(new_types[i])
            self.enemy_patterns[i] = [
                self.stored_patterns[str(et)]["attack_interval"],
                self.stored_patterns[str(et)]["avg_dx"]
            ]
        
        self.enemy_positions = new_positions
        self.enemy_types = new_types
        self.enemy_timers = new_timers

    def _update_boss_info(self, ram):
        stage = int(ram[0x0058])
        if stage == 5:
            boss_pos_x = int(ram[0x0093])
            boss_action = int(ram[0x004E])
            boss_hp = int(ram[0x04A5])
            self.boss_info = np.array([
                boss_pos_x - int(ram[0x0094]),
                boss_action / 255.0,
                boss_hp / 255.0
            ], dtype=np.float32)
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
            lower_green = np.array([0, 100, 0])
            upper_green = np.array([100, 255, 100])
            
            white_mask = cv2.inRange(frame, lower_white, upper_white)
            green_mask = cv2.inRange(frame, lower_green, upper_green)
            color_mask = cv2.bitwise_or(white_mask, green_mask)
            color_mask = cv2.dilate(color_mask, None, iterations=2)
            
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
                        dx, dy = 0, 0
                        for prev_pos in self.last_projectile_positions:
                            prev_x, prev_y = prev_pos
                            if abs(proj_x - prev_x) < 20 and abs(proj_y - prev_y) < 20:
                                dx = proj_x - prev_x
                                dy = proj_y - prev_y
                                speed = np.sqrt(dx**2 + dy**2)
                                if speed < 5 or speed > 25:
                                    dx, dy = 0, 0
                                if abs(dx) < 2:
                                    dx, dy = 0, 0
                                break
                        
                        if dx != 0 or dy != 0:
                            game_width = 256
                            proj_x_game = (proj_x / self.viewport_size[0]) * game_width
                            distance = proj_x_game - hero_x
                            projectile_info.extend([distance, proj_y, dx, dy])
                            current_projectile_positions.append((proj_x, proj_y))
            
            projectile_info = projectile_info[:self.max_projectiles * 4]
            while len(projectile_info) < self.max_projectiles * 4:
                projectile_info.append(0)
            
            self.last_projectile_positions = current_projectile_positions[:self.max_projectiles]
            self.prev_frame = frame
            return projectile_info
        
        self.prev_frame = frame
        return [0] * (self.max_projectiles * 4)

    def _check_projectile_hit(self, hp_loss):
        if hp_loss > 0 and len(self.last_projectile_positions) > 0:
            return True
        return False

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

    def close(self):
        try:
            self.env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

class CombatTrainingCallback(BaseCallback):
    def __init__(self, progress_bar=False, logger=None, total_timesteps=10000):
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
        self.projectile_hits = 0
        self.projectile_avoids = 0
        
        if progress_bar:
            self.pbar = tqdm(total=total_timesteps, desc="Training")

    def _on_step(self):
        self.total_steps += 1
        self.current_episode_reward += self.locals["rewards"][0]
        hit_count = self.locals["infos"][0]["enemy_hit"]
        projectile_hit = self.locals["infos"][0]["projectile_hit"]
        projectile_avoided = self.locals["infos"][0]["projectile_avoided"]
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
            self.logger.record("game/hp_change_rate", info.get("hp_change_rate", 0))
            self.logger.record("game/enemy_very_close", info.get("enemy_very_close", 0))
            self.logger.record("game/dominant_action_percentage", info.get("dominant_action_percentage", 0))
            self.logger.record("combat/projectile_hits", self.projectile_hits)
            self.logger.record("combat/projectile_avoids", self.projectile_avoids)
            
            enemy_types = info.get("enemy_types", [0] * 4)
            for i, et in enumerate(enemy_types):
                self.logger.record(f"enemy_types/enemy_{i}", et)
            
            enemy_timers = info.get("enemy_timers", [0] * 4)
            for i, timer in enumerate(enemy_timers):
                self.logger.record(f"enemy_timers/enemy_{i}", timer)
            
            enemy_patterns = info.get("enemy_patterns", np.zeros((4, 2))).reshape(-1)
            for i in range(len(enemy_types)):
                self.logger.record(f"enemy_patterns/enemy_{i}_attack_interval", enemy_patterns[i * 2])
                self.logger.record(f"enemy_patterns/enemy_{i}_avg_dx", enemy_patterns[i * 2 + 1])
            
            boss_info = info.get("boss_info", [0, 0, 0])
            self.logger.record("boss/pos_x", boss_info[0])
            self.logger.record("boss/action", boss_info[1])
            self.logger.record("boss/hp", boss_info[2])
            
            action_percentages = info.get("action_percentages", np.zeros(9))
            action_names = info.get("action_names", [])
            actions_log = [f"{name}:{percentage:.1%}" for name, percentage in zip(action_names, action_percentages)]
            self.logger.record("actions", actions_log)
            
            steps_per_second = self.total_steps / (time.time() - self.start_time)
            self.logger.record("time/steps_per_second", steps_per_second)
        
        current_time = time.time()
        if current_time - self.last_log_time > 30 or self.total_steps % 10 == 0:
            info = self.locals["infos"][0]
            hit_rate = self.total_hits / (self.total_steps + 1e-6)
            action_percentages = info.get("action_percentages", np.zeros(9))
            action_names = info.get("action_names", [])
            actions_log = [f"{name}:{percentage:.1%}" for name, percentage in zip(action_names, action_percentages)]
            self.logger.info(
                f"Step: {self.total_steps}, "
                f"Hits: {self.total_hits}, "
                f"Hit Rate: {hit_rate:.2%}, "
                f"HP: {info.get('hp', 0)}, "
                f"Actions: {actions_log}"
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
        self.logger.record("combat/episode_projectile_hits", self.projectile_hits)
        self.logger.record("combat/episode_projectile_avoids", self.projectile_avoids)
        
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
            f"Projectile Hits={self.projectile_hits}, "
            f"Projectile Avoids={self.projectile_avoids}"
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
            f"Projectile Hits: {self.projectile_hits}, "
            f"Projectile Avoids: {self.projectile_avoids}, "
            f"Duration: {training_duration:.2f} seconds"
        )

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
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

def make_env(render=False, patterns_db="enemy_patterns.db"):
    try:
        env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
        if render:
            env.render_mode = 'human'
        return KungFuWrapper(env, patterns_db=patterns_db)
    except Exception as e:
        logger.error(f"Failed to create environment: {e}")
        raise

def make_vec_env(num_envs, render=False, patterns_db="enemy_patterns.db"):
    logger.info(f"Creating vectorized environment with {num_envs} subprocesses")
    try:
        if num_envs > 1:
            env_fns = [lambda: make_env(render, patterns_db) for _ in range(num_envs)]
            env = SubprocVecEnv(env_fns, start_method="spawn")
        else:
            env = DummyVecEnv([lambda: make_env(render, patterns_db)])
        return VecFrameStack(env, n_stack=12)
    except Exception as e:
        logger.error(f"Failed to create vectorized environment: {e}")
        raise

def save_best_params_to_db(db_path, best_params, best_value):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS best_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learning_rate REAL,
            n_steps INTEGER,
            batch_size INTEGER,
            n_epochs INTEGER,
            gamma REAL,
            clip_range REAL,
            ent_coef REAL,
            vf_coef REAL,
            max_grad_norm REAL,
            best_value REAL,
            timestamp TEXT
        )
    ''')
    c.execute('''
        INSERT INTO best_params (
            learning_rate, n_steps, batch_size, n_epochs, gamma,
            clip_range, ent_coef, vf_coef, max_grad_norm, best_value, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        best_params['learning_rate'], best_params['n_steps'], best_params['batch_size'],
        best_params['n_epochs'], best_params['gamma'], best_params['clip_range'],
        best_params['ent_coef'], best_params['vf_coef'], best_params['max_grad_norm'],
        best_value, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))
    conn.commit()
    conn.close()
    logger.info(f"Saved best parameters to {db_path} with value {best_value}")

def load_best_params_from_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS best_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            learning_rate REAL,
            n_steps INTEGER,
            batch_size INTEGER,
            n_epochs INTEGER,
            gamma REAL,
            clip_range REAL,
            ent_coef REAL,
            vf_coef REAL,
            max_grad_norm REAL,
            best_value REAL,
            timestamp TEXT
        )
    ''')
    
    c.execute('''
        SELECT learning_rate, n_steps, batch_size, n_epochs, gamma,
               clip_range, ent_coef, vf_coef, max_grad_norm, best_value
        FROM best_params
        ORDER BY best_value DESC
        LIMIT 1
    ''')
    row = c.fetchone()
    conn.close()
    
    if row and row[9] >= GOOD_ENOUGH_THRESHOLD:
        params = {
            'learning_rate': row[0],
            'n_steps': row[1],
            'batch_size': row[2],
            'n_epochs': row[3],
            'gamma': row[4],
            'clip_range': row[5],
            'ent_coef': row[6],
            'vf_coef': row[7],
            'max_grad_norm': row[8]
        }
        logger.info(f"Loaded best parameters from {db_path} with value {row[9]}")
        return params, row[9]
    else:
        if row is None:
            logger.info(f"No parameters found in {db_path}. Proceeding with Optuna search.")
        else:
            logger.info(f"Best parameters found in {db_path} with value {row[9]}, below threshold {GOOD_ENOUGH_THRESHOLD}. Proceeding with Optuna search.")
        return None, None
    
def objective(trial, args, env):
    global global_model, global_model_file, logger
    
    combat_params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 20),
        'gamma': trial.suggest_uniform('gamma', 0.9, 0.999),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-2, 0.5),
        'vf_coef': trial.suggest_uniform('vf_coef', 0.5, 1.0),
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
                "net_arch": [dict(pi=[256, 256, 128], vf=[512, 512, 256])]
            },
            **combat_params,
            tensorboard_log=args.log_dir,
            device="cuda" if args.cuda else "cpu"
        )
    
    global_model = model
    global_model_file = model_file
    
    callback = CombatTrainingCallback(progress_bar=args.progress_bar, logger=logger, total_timesteps=args.timesteps)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=f"PPO_KungFu_trial_{trial.number}",
            reset_num_timesteps=not args.resume
        )
    except Exception as e:
        logger.error(f"Training failed for trial {trial.number}: {e}")
        raise
    
    total_hits = callback.total_hits
    logger.info(f"Trial {trial.number} completed with total_hits: {total_hits}")
    
    save_model_with_logging(model, global_model_file, logger)
    
    if total_hits >= GOOD_ENOUGH_THRESHOLD:
        logger.info(f"Trial {trial.number} exceeded threshold ({GOOD_ENOUGH_THRESHOLD} hits). Stopping Optuna optimization.")
        trial.study.stop()
        return total_hits
    
    return total_hits

def train_with_best_params(args, env, best_params):
    global global_model, global_model_file, logger
    
    logger.info(f"Starting full training with best parameters: {best_params}")
    
    model_file = os.path.join(args.model_path, "kungfu_ppo_best")
    if args.resume and os.path.exists(model_file + ".zip"):
        logger.info(f"Resuming best model from {model_file}.zip")
        model = PPO.load(model_file, env=env, device="cuda" if args.cuda else "cpu")
    else:
        logger.info("Creating new model with best parameters")
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={
                "features_extractor_class": SimpleCNN,
                "net_arch": [dict(pi=[256, 256, 128], vf=[512, 512, 256])]
            },
            **best_params,
            tensorboard_log=args.log_dir,
            device="cuda" if args.cuda else "cpu"
        )
    
    global_model = model
    global_model_file = model_file
    
    callback = CombatTrainingCallback(progress_bar=args.progress_bar, logger=logger, total_timesteps=args.timesteps)
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name="PPO_KungFu_best",
            reset_num_timesteps=not args.resume
        )
    except Exception as e:
        logger.error(f"Full training failed: {e}")
        raise
    
    save_model_with_logging(model, global_model_file, logger)
    logger.info(f"Full training completed with total hits: {callback.total_hits}")

def train(args):
    global logger
    logger = setup_logging(args.log_dir)
    logger.info("Starting training session")
    logger.info(f"Command line arguments: {vars(args)}")
    
    signal.signal(signal.SIGINT, signal_handler)
    atexit.register(save_model_on_exit)
    
    patterns_db = os.path.join(args.log_dir, "enemy_patterns.db")
    best_params_db = os.path.join(args.log_dir, "best_params.db")
    env = None
    
    try:
        env = make_vec_env(args.num_envs, render=args.render, patterns_db=patterns_db)
        
        model_path = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
        os.makedirs(model_path, exist_ok=True)
        
        default_params = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'clip_range': 0.3,
            'ent_coef': 0.1,
            'vf_coef': 0.7,
            'max_grad_norm': 0.5
        }
        
        if args.no_optuna:
            logger.info("Optuna optimization disabled via --no-optuna flag")
            best_params, best_value = load_best_params_from_db(best_params_db)
            if best_params is not None:
                logger.info(f"Using pre-loaded best parameters from database with value {best_value}")
                train_with_best_params(args, env, best_params)
            else:
                logger.info("No best parameters found in database. Using default parameters.")
                train_with_best_params(args, env, default_params)
        else:
            best_params, best_value = load_best_params_from_db(best_params_db)
            
            if best_params is not None and best_value >= GOOD_ENOUGH_THRESHOLD:
                logger.info(f"Found good enough parameters (value: {best_value}) in database. Skipping Optuna search.")
                train_with_best_params(args, env, best_params)
            else:
                logger.info("Starting or resuming Optuna optimization.")
                study_name = "kungfu_ppo_study"
                storage_name = f"sqlite:///{os.path.join(args.log_dir, 'optuna_study.db')}"
                
                if args.resume and os.path.exists(os.path.join(args.log_dir, 'optuna_study.db')):
                    logger.info(f"Resuming existing Optuna study from {storage_name}")
                    study = optuna.load_study(study_name=study_name, storage=storage_name)
                else:
                    logger.info("Creating new Optuna study")
                    study = optuna.create_study(study_name=study_name, storage=storage_name, direction="maximize")
                
                n_trials = args.n_trials
                logger.info(f"Running Optuna optimization with {n_trials} trials or until threshold ({GOOD_ENOUGH_THRESHOLD}) is reached")
                study.optimize(lambda trial: objective(trial, args, env), n_trials=n_trials)
                
                best_trial = study.best_trial
                logger.info(f"Best trial: {best_trial.number}")
                logger.info(f"Best parameters: {best_trial.params}")
                logger.info(f"Best value (total hits): {best_trial.value}")
                
                # Correct the path for loading the best model
                best_model_file = os.path.join(model_path, f"kungfu_ppo_trial_{best_trial.number}")
                best_model_file = os.path.normpath(best_model_file)  # Normalize the path
                global_model_file = os.path.join(model_path, "kungfu_ppo_best")
                global_model_file = os.path.normpath(global_model_file)
                
                # Check if the file exists before loading
                if not os.path.exists(best_model_file + ".zip"):
                    logger.error(f"Best model file {best_model_file}.zip does not exist. Cannot proceed with loading.")
                    raise FileNotFoundError(f"Best model file {best_model_file}.zip not found.")
                
                logger.info(f"Loading best model from {best_model_file}")
                best_model = PPO.load(best_model_file, env=env, device="cuda" if args.cuda else "cpu")
                save_model_with_logging(best_model, global_model_file, logger)
                
                save_best_params_to_db(best_params_db, best_trial.params, best_trial.value)
                
                if best_trial.value >= GOOD_ENOUGH_THRESHOLD:
                    logger.info(f"Best trial ({best_trial.value} hits) meets threshold. Proceeding to full training.")
                    train_with_best_params(args, env, best_trial.params)
                else:
                    logger.info(f"No trial met the threshold ({GOOD_ENOUGH_THRESHOLD} hits). Saved parameters for future improvement.")
    
    except Exception as e:
        logger.error(f"Training session failed: {e}")
        raise
    finally:
        if env is not None:
            logger.info("Closing environment")
            env.close()
    
    logger.info("Training session ended")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Kung Fu with PPO and Optuna.")
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps per trial")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--n_trials", type=int, default=3, help="Number of Optuna trials")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved study and models")
    parser.add_argument("--no-optuna", action="store_true", help="Disable Optuna optimization and use default or loaded parameters")

    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    train(args)