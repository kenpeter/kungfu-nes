import retro
import numpy as np
from gym import spaces, Wrapper
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

KUNGFU_MAX_ENEMIES = 5  # Single source of truth

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        
        self.actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No-op (0)
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Punch (1)
            [0,0,0,0,0,0,0,0,1,0,0,0],  # Kick (2)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # Right+Punch (3)
            [0,1,0,0,0,0,1,0,0,0,0,0],  # Left+Punch (4)
            [0,0,1,0,0,0,0,0,0,0,0,0],  # Crouch (5)
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Jump (6)
            [0,0,0,0,0,1,1,0,0,0,0,0],  # Jump+Punch (7)
            [0,0,1,0,0,0,1,0,0,0,0,0]   # Crouch+Punch (8)
        ]
        self.action_names = [
            "No-op", "Punch", "Kick", "Right+Punch", "Left+Punch",
            "Crouch", "Jump", "Jump+Punch", "Crouch+Punch"
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_enemies = KUNGFU_MAX_ENEMIES
        self.max_projectiles = 2
        
        # Removed is_distance_enemy from observation space
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 3), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (self.max_enemies * 2,), np.float32),
            "projectile_vectors": spaces.Box(-255, 255, (self.max_projectiles * 4,), np.float32),
            "combat_status": spaces.Box(-1, 1, (2,), np.float32),
            "enemy_proximity": spaces.Box(0, 1, (1,), np.float32),
            "boss_info": spaces.Box(-255, 255, (3,), np.float32),
        })
        
        self.last_hp = 0
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        self.survival_reward_total = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.prev_min_enemy_dist = 255

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = float(self.env.get_ram()[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        self.survival_reward_total = 0
        self.prev_min_enemy_dist = 255
        return self._get_obs(obs)

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = float(ram[0x04A6])
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091]), int(ram[0x0092])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)

        reward = 0
        hp_change_rate = (hp - self.last_hp) / 255.0
        
        if hp_change_rate < 0:
            reward += (hp_change_rate ** 2) * 50
        else:
            reward += hp_change_rate * 5

        reward += enemy_hit * 10
        if done:
            reward -= 50

        projectile_info = self._detect_projectiles(obs)
        projectile_distances = [projectile_info[i] for i in range(0, len(projectile_info), 4)]
        dodge_reward = 0
        for i, (curr_dist, last_dist) in enumerate(zip(projectile_distances, self.last_projectile_distances)):
            if last_dist < 20 and curr_dist > last_dist:
                if action in [5, 6]:  # Crouch or Jump
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances
        
        if hp_change_rate < 0 and action in [5, 6]:
            reward += 5

        # Simplified distance logic for close combat
        hero_x = int(ram[0x0094])
        min_enemy_dist = min([abs(enemy_x - hero_x) for enemy_x in curr_enemies if enemy_x != 0] or [255])
        distance_change = self.prev_min_enemy_dist - min_enemy_dist
        
        # Simplified close-combat focused rewards
        close_range_threshold = 30
        if min_enemy_dist <= close_range_threshold:
            if action in [1, 2, 8]:  # Punch, Kick, Crouch+Punch
                reward += 1.0
        else:
            distance_reward = distance_change * 0.05  # Encourage closing distance
            reward += distance_reward

        self.prev_min_enemy_dist = min_enemy_dist

        action_entropy = -np.sum((self.action_counts / (self.total_steps + 1e-6)) * 
                                np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6))
        reward += action_entropy * 3.0
        
        if not done and hp > 0:
            reward += 0.05
            self.survival_reward_total += 0.05

        # Reward normalization
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        normalized_reward = (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-6)
        normalized_reward = np.clip(normalized_reward, -10, 10)

        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward,
            "distance_reward": distance_reward if 'distance_reward' in locals() else 0,
            "survival_reward_total": self.survival_reward_total,
            "raw_reward": reward,
            "normalized_reward": normalized_reward
        })
        
        return self._get_obs(obs), normalized_reward, done, info

    def _update_boss_info(self, ram):
        stage = int(ram[0x0058])
        if stage == 5:
            boss_pos_x = int(ram[0x0093])
            boss_action = int(ram[0x004E])
            boss_hp = int(ram[0x04A5])
            return np.array([boss_pos_x - int(ram[0x0094]), boss_action / 255.0, boss_hp / 255.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _detect_projectiles(self, obs):
        frame = cv2.resize(obs, self.viewport_size, interpolation=cv2.INTER_AREA)
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(frame, self.prev_frame)
            diff_sum = np.sum(frame_diff, axis=2).astype(np.uint8)
            _, motion_mask = cv2.threshold(diff_sum, 20, 255, cv2.THRESH_BINARY)
            
            lower_white = np.array([180, 180, 180])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(frame, lower_white, upper_white)
            combined_mask = cv2.bitwise_and(motion_mask, white_mask)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            projectile_info = []
            hero_x = int(self.env.get_ram()[0x0094])
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 5 < area < 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    proj_x = x + w // 2
                    proj_y = y + h // 2
                    game_width = 256
                    proj_x_game = (proj_x / self.viewport_size[0]) * game_width
                    distance = proj_x_game - hero_x
                    projectile_info.extend([distance, proj_y, 0, 0])
            
            projectile_info = projectile_info[:self.max_projectiles * 4]
            while len(projectile_info) < self.max_projectiles * 4:
                projectile_info.append(0)
            self.prev_frame = frame
            return projectile_info
        
        self.prev_frame = frame
        return [0] * (self.max_projectiles * 4)

    def _get_obs(self, obs):
        viewport = cv2.resize(obs, self.viewport_size)
        ram = self.env.get_ram()
        
        hero_x = int(ram[0x0094])
        enemy_info = []
        for addr in [0x008E, 0x008F, 0x0090, 0x0091, 0x0092]:
            enemy_x = int(ram[addr])
            if enemy_x != 0:
                distance = enemy_x - hero_x
                direction = 1 if distance > 0 else -1
                enemy_info.extend([direction, min(abs(distance), 255)])
            else:
                enemy_info.extend([0, 0])
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": np.array(enemy_info, dtype=np.float32),
            "combat_status": np.array([self.last_hp/255.0, self.last_hp_change], dtype=np.float32),
            "projectile_vectors": np.array(self._detect_projectiles(obs), dtype=np.float32),
            "enemy_proximity": np.array([1.0 if any(abs(enemy_x - hero_x) <= 20 for enemy_x in [int(ram[addr]) for addr in [0x008E, 0x008F, 0x0090, 0x0091, 0x0092]] if enemy_x != 0) else 0.0], dtype=np.float32),
            "boss_info": self._update_boss_info(ram),
        }

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(36, 64, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample_input = torch.zeros(1, 36, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        combat_status_size = observation_space["combat_status"].shape[0]
        projectile_vec_size = observation_space["projectile_vectors"].shape[0]
        enemy_proximity_size = observation_space["enemy_proximity"].shape[0]
        boss_info_size = observation_space["boss_info"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten + enemy_vec_size + combat_status_size +
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
        combat_status = observations["combat_status"]
        projectile_vectors = observations["projectile_vectors"]
        enemy_proximity = observations["enemy_proximity"]
        boss_info = observations["boss_info"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport).float()
        for tensor in [enemy_vector, combat_status, projectile_vectors, 
                      enemy_proximity, boss_info]:
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor).float()
                
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and viewport.shape[-1] in (3,):
            viewport = viewport.permute(0, 3, 1, 2)
            
        cnn_output = self.cnn(viewport)
        
        for tensor in [enemy_vector, combat_status, projectile_vectors,
                      enemy_proximity, boss_info]:
            if len(tensor.shape) == 1:
                tensor = tensor.unsqueeze(0)
                
        combined = torch.cat([
            cnn_output, enemy_vector, combat_status, projectile_vectors,
            enemy_proximity, boss_info
        ], dim=1)
        return self.linear(combined)
         
def make_env():
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    return KungFuWrapper(env)