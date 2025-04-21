import gymnasium as gym
import numpy as np
from gymnasium import spaces, Wrapper
import cv2
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import retro

KUNGFU_MAX_ENEMIES = 5
MAX_PROJECTILES = 2

# Define actions with 9 buttons to match retro environment
KUNGFU_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No-op
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Punch)
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # SELECT
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # START
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # UP (Jump)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN (Crouch)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (Kick)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # B + A (Punch + Kick)
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # UP + RIGHT (Jump + Right)
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # DOWN + A (Crouch Kick)
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN + B (Crouch Punch)
]
KUNGFU_ACTION_NAMES = [
    "No-op", "Punch", "Select", "Start", "Jump",
    "Crouch", "Left", "Right", "Kick", "Punch + Kick",
    "Jump + Right", "Crouch Kick", "Crouch Punch"
]

# Define observation space with 160x160 viewport
KUNGFU_OBSERVATION_SPACE = spaces.Dict({
    "viewport": spaces.Box(0, 255, (160, 160, 3), np.uint8),
    "enemy_vector": spaces.Box(-255, 255, (KUNGFU_MAX_ENEMIES * 2,), np.float32),
    "projectile_vectors": spaces.Box(-255, 255, (MAX_PROJECTILES * 4,), np.float32),
    "combat_status": spaces.Box(-1, 1, (2,), np.float32),
    "enemy_proximity": spaces.Box(0, 1, (1,), np.float32),
    "boss_info": spaces.Box(-255, 255, (3,), np.float32),
    "closest_enemy_direction": spaces.Box(-1, 1, (1,), np.float32)
})

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Get initial observation and handle tuple return
        result = env.reset()
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result
            
        self.true_height, self.true_width = obs.shape[:2]
        self.viewport_size = (160, 160)
        
        self.actions = KUNGFU_ACTIONS
        self.action_names = KUNGFU_ACTION_NAMES
        self.action_space = spaces.Discrete(len(self.actions))
        self.max_enemies = KUNGFU_MAX_ENEMIES
        self.max_projectiles = MAX_PROJECTILES
        self.observation_space = KUNGFU_OBSERVATION_SPACE
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
        self.last_movement = None

    def reset(self, seed=None, options=None, **kwargs):
        result = self.env.reset(seed=seed, options=options, **kwargs)
        
        # Handle both old and new API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        self.last_hp = float(self.env.get_ram()[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * self.max_projectiles
        self.survival_reward_total = 0
        self.prev_min_enemy_dist = 255
        self.last_movement = None
        return self._get_obs(obs), info

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1
        
        # Execute action and handle return value formats
        result = self.env.step(self.actions[action])
        
        # Handle both old and new API
        if len(result) == 5:  # New API (obs, rew, terminated, truncated, info)
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old API (obs, rew, done, info)
            obs, reward, done, info = result
            terminated = done
            truncated = False
            
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
                if action in [5, 6]:  # Crouch or Left dodge
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances
        
        if hp_change_rate < 0 and action in [5, 6]:
            reward += 5

        hero_x = int(ram[0x0094])
        enemy_distances = [(enemy_x - hero_x) for enemy_x in curr_enemies if enemy_x != 0]
        min_enemy_dist = min([abs(d) for d in enemy_distances] or [255])
        
        closest_enemy_dir = 0
        if enemy_distances:
            closest_dist = min([abs(d) for d in enemy_distances])
            closest_enemy_dir = 1 if [d for d in enemy_distances if abs(d) == closest_dist][0] > 0 else -1
        
        close_range_threshold = 30
        
        if min_enemy_dist > close_range_threshold:
            movement_reward = 8.0
            valid_actions = [9, 10]  # Punch + Kick, Jump + Right
            if action in valid_actions:
                if (action == 9 and closest_enemy_dir == 1) or (action == 10 and closest_enemy_dir == -1):
                    reward += movement_reward * (1 + min_enemy_dist/255) * 1.5
                    if hasattr(self, 'last_movement') and self.last_movement == action:
                        reward += 3.0
                    self.last_movement = action
                else:
                    reward -= 10.0
            else:
                reward -= 50.0
                print(f"Invalid action {self.action_names[action]} taken when far (dist={min_enemy_dist}), reward={reward}")
        else:
            if action in [1, 8, 11, 12]:  # Punch, Kick, Crouch Kick, Crouch Punch
                reward += 2.0
            elif action in [3, 4]:  # Start, Jump
                if (action == 3 and closest_enemy_dir == 1) or (action == 4 and closest_enemy_dir == -1):
                    reward += 2.5
                else:
                    reward += 0.5
            elif action in [9, 10]:  # Punch + Kick, Jump + Right
                reward -= 0.5
            elif action in [0, 5, 6]:  # No-op, Crouch, Left
                reward -= 0.5

        self.prev_min_enemy_dist = min_enemy_dist

        action_entropy = -np.sum((self.action_counts / (self.total_steps + 1e-6)) * 
                                np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6))
        reward += action_entropy * 0.1
        
        if not done and hp > 0:
            reward += 0.05
            self.survival_reward_total += 0.05

        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        normalized_reward = (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-6)
        normalized_reward = np.clip(normalized_reward, -10, 10)

        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        
        if min_enemy_dist > close_range_threshold:
            print(f"Step {self.total_steps}: Action={self.action_names[action]}, MinDist={min_enemy_dist}, ClosestDir={closest_enemy_dir}, Reward={reward}")
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward,
            "survival_reward_total": self.survival_reward_total,
            "raw_reward": reward,
            "normalized_reward": normalized_reward,
            "min_enemy_dist": min_enemy_dist,
            "closest_enemy_direction": closest_enemy_dir
        })
        
        # Return format compatible with new Gymnasium API
        return self._get_obs(obs), normalized_reward, terminated, truncated, info
    
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
                    distance = proj_x - hero_x
                    projectile_info.extend([distance, proj_y, 0, 0])
            projectile_info = projectile_info[:self.max_projectiles * 4]
            while len(projectile_info) < self.max_projectiles * 4:
                projectile_info.append(0)
            self.prev_frame = frame.copy()
            return projectile_info
        self.prev_frame = frame.copy()
        return [0] * (self.max_projectiles * 4)

    def _get_obs(self, obs):
        viewport = cv2.resize(obs, self.viewport_size, interpolation=cv2.INTER_AREA)
        ram = self.env.get_ram()
        hero_x = int(ram[0x0094])
        enemy_info = []
        distances = []
        for addr in [0x008E, 0x008F, 0x0090, 0x0091, 0x0092]:
            enemy_x = int(ram[addr])
            if enemy_x != 0:
                distance = enemy_x - hero_x
                direction = 1 if distance > 0 else -1
                enemy_info.extend([direction, min(abs(distance), 255)])
                distances.append((abs(distance), direction))
            else:
                enemy_info.extend([0, 0])
                distances.append((float('inf'), 0))
        closest_enemy_direction = 0
        if distances:
            closest_dist, closest_dir = min(distances, key=lambda x: x[0])
            closest_enemy_direction = closest_dir if closest_dist != float('inf') else 0
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": np.array(enemy_info, dtype=np.float32),
            "combat_status": np.array([self.last_hp/255.0, self.last_hp_change], dtype=np.float32),
            "projectile_vectors": np.array(self._detect_projectiles(obs), dtype=np.float32),
            "enemy_proximity": np.array([1.0 if any(abs(enemy_x - hero_x) <= 20 for enemy_x in [int(ram[addr]) for addr in [0x008E, 0x008F, 0x0090, 0x0091, 0x0092]] if enemy_x != 0) else 0.0], dtype=np.float32),
            "boss_info": self._update_boss_info(ram),
            "closest_enemy_direction": np.array([closest_enemy_direction], dtype=np.float32)
        }

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_stack=4):
        super().__init__(observation_space, features_dim)
        viewport_shape = observation_space["viewport"].shape
        height, width = viewport_shape[0], viewport_shape[1]
        input_channels = 3 * n_stack

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            sample_input = torch.zeros(1, input_channels, height, width)
            n_flatten = self.cnn(sample_input).shape[1]

        non_visual_size = sum(observation_space[k].shape[0] for k in [
            "enemy_vector", "combat_status", "projectile_vectors",
            "enemy_proximity", "boss_info", "closest_enemy_direction"
        ])

        self.non_visual = nn.Sequential(
            nn.Linear(non_visual_size, 256),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 256, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        viewport = observations["viewport"].float() / 255.0
        viewport = viewport.permute(0, 3, 1, 2)
        cnn_output = self.cnn(viewport)

        non_visual = torch.cat([
            observations["enemy_vector"].float(),
            observations["combat_status"].float(),
            observations["projectile_vectors"].float(),
            observations["enemy_proximity"].float(),
            observations["boss_info"].float(),
            observations["closest_enemy_direction"].float()
        ], dim=-1)
        
        non_visual_output = self.non_visual(non_visual)
        
        combined = torch.cat([cnn_output, non_visual_output], dim=1)
        return self.linear(combined)

def make_env():
    env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
    env = Monitor(KungFuWrapper(env))
    return env