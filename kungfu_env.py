import gymnasium as gym
import numpy as np
from gymnasium import spaces, Wrapper
import cv2
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import retro
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Constants
KUNGFU_MAX_ENEMIES = 4
MAX_PROJECTILES = 2
N_STACK = 4  # Match VecFrameStack in train.py

# Define actions
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

KUNGFU_OBSERVATION_SPACE = spaces.Dict({
    "viewport": spaces.Box(low=0, high=255, shape=(160, 160, 3), dtype=np.uint8),
    "projectile_vectors": spaces.Box(low=-255, high=255, shape=(MAX_PROJECTILES * 4,), dtype=np.float32)
})

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
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
        self.observation_space = KUNGFU_OBSERVATION_SPACE
        
        # State tracking
        self.last_hp = 0
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * KUNGFU_MAX_ENEMIES
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * MAX_PROJECTILES
        self.survival_reward_total = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.player_x = 0
        self.last_player_x = 0
        self.timer = 0
        self.last_timer = 0
        self.stage = 0
        self.last_stage = 0
        self.boss_hp = 0
        self.last_boss_hp = 0
        self.boss_pos_x = 0
        self.boss_action = 0
        self.enemy_action_timers = [0] * KUNGFU_MAX_ENEMIES
        self.enemy_actions = [0] * KUNGFU_MAX_ENEMIES

    def reset(self, seed=None, options=None, **kwargs):
        result = self.env.reset(seed=seed, options=options, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        ram = self.env.get_ram()
        self.last_hp = float(ram[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * KUNGFU_MAX_ENEMIES
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * MAX_PROJECTILES
        self.survival_reward_total = 0
        self.player_x = float(ram[0x0094])
        self.last_player_x = self.player_x
        self.timer = float(ram[0x0391])
        self.last_timer = self.timer
        self.stage = int(ram[0x0058])
        self.last_stage = self.stage
        self.boss_hp = float(ram[0x04A5])
        self.last_boss_hp = self.boss_hp
        self.boss_pos_x = float(ram[0x0093])
        self.boss_action = int(ram[0x004E])
        self.enemy_action_timers = [int(ram[0x002B]), int(ram[0x002C]), int(ram[0x002D]), int(ram[0x002E])]
        self.enemy_actions = [int(ram[0x0080]), int(ram[0x0081]), int(ram[0x0082]), int(ram[0x0083])]
        return self._get_obs(obs), info

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1

        result = self.env.step(self.actions[action])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        ram = self.env.get_ram()
        hp = float(ram[0x04A6])
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        self.player_x = float(ram[0x0094])
        self.timer = float(ram[0x0391])
        self.stage = int(ram[0x0058])
        self.boss_hp = float(ram[0x04A5])
        self.boss_pos_x = float(ram[0x0093])
        self.boss_action = int(ram[0x004E])
        self.enemy_action_timers = [int(ram[0x002B]), int(ram[0x002C]), int(ram[0x002D]), int(ram[0x002E])]
        self.enemy_actions = [int(ram[0x0080]), int(ram[0x0081]), int(ram[0x0082]), int(ram[0x0083])]

        # Reward structure
        reward = 0
        hp_change_rate = (hp - self.last_hp) / 255.0
        if hp_change_rate < 0:
            reward += (hp_change_rate ** 2) * 50
        else:
            reward += hp_change_rate * 5

        reward += enemy_hit * 10
        if done:
            reward -= 50

        # Progression reward based on stage
        x_change = self.player_x - self.last_player_x
        desired_direction = -1 if self.stage in [1, 3, 5] else 1
        progression_reward = x_change * desired_direction * 0.2
        if action in [7, 10] and desired_direction > 0:
            progression_reward += 3.0
        elif action == 6 and desired_direction < 0:
            progression_reward += 3.0
        reward += progression_reward

        # Stage transition reward
        if self.stage > self.last_stage:
            reward += 50
            self.last_stage = self.stage

        # Timer penalty
        timer_change = self.last_timer - self.timer
        if timer_change > 0:
            reward -= timer_change * 0.5

        # Boss HP reward
        boss_hp_decrease = self.last_boss_hp - self.boss_hp
        if boss_hp_decrease > 0:
            reward += boss_hp_decrease * 10

        # Boss distance reward
        boss_distance = abs(self.player_x - self.boss_pos_x)
        last_boss_distance = abs(self.last_player_x - self.boss_pos_x)
        distance_change = last_boss_distance - boss_distance
        boss_distance_reward = distance_change * 0.1
        reward += boss_distance_reward

        # Projectile-based dodge reward
        projectile_info = self._detect_projectiles(obs)
        projectile_distances = [projectile_info[i] for i in range(0, len(projectile_info), 4)]
        dodge_reward = 0
        for i, (curr_dist, last_dist) in enumerate(zip(projectile_distances, self.last_projectile_distances)):
            if last_dist < 20 and curr_dist > last_dist:
                if action in [4, 5]:
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances

        # Action-based rewards
        if hp_change_rate < 0 and action in [5, 6]:
            reward += 5
        if action in [1, 8, 11, 12]:
            reward += 2.0
        elif action in [9, 10]:
            reward -= 0.5

        # Action entropy
        action_entropy = -np.sum((self.action_counts / (self.total_steps + 1e-6)) * 
                                 np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6))
        reward += action_entropy * 0.1

        # Survival reward
        if not done and hp > 0:
            reward += 0.05
            self.survival_reward_total += 0.05

        # Normalize reward
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = 0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        normalized_reward = (reward - self.reward_mean) / (np.sqrt(self.reward_std) + 1e-6)
        normalized_reward = np.clip(normalized_reward, -10, 10)

        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        self.last_player_x = self.player_x
        self.last_timer = self.timer
        self.last_boss_hp = self.boss_hp

        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward,
            "survival_reward_total": self.survival_reward_total,
            "raw_reward": reward,
            "normalized_reward": normalized_reward,
            "progression_reward": progression_reward,
            "stage": self.stage,
            "player_x": self.player_x,
            "timer": self.timer,
            "boss_hp": self.boss_hp,
            "boss_hp_decrease": boss_hp_decrease,
            "boss_pos_x": self.boss_pos_x,
            "boss_action": self.boss_action,
            "enemy_action_timers": self.enemy_action_timers,
            "enemy_actions": self.enemy_actions
        })

        return self._get_obs(obs), normalized_reward, terminated, truncated, info

    def _get_obs(self, obs):
        viewport = cv2.resize(obs, self.viewport_size, interpolation=cv2.INTER_AREA)
        return {
            "viewport": viewport.astype(np.uint8),
            "projectile_vectors": np.array(self._detect_projectiles(obs), dtype=np.float32)
        }

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
                    velocity = (proj_x - hero_x) - (self.last_player_x - hero_x)
                    projectile_info.extend([distance, proj_y, velocity, 0])
            projectile_info = projectile_info[:MAX_PROJECTILES * 4]
            while len(projectile_info) < MAX_PROJECTILES * 4:
                projectile_info.append(0)
            self.prev_frame = frame.copy()
            return projectile_info
        self.prev_frame = frame.copy()
        return [0] * (MAX_PROJECTILES * 4)

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        # Infer n_stack from viewport channels
        viewport_shape = observation_space["viewport"].shape
        
        # Handle the case where shape is (H, W, C) instead of (C, H, W)
        if len(viewport_shape) == 3 and viewport_shape[2] < viewport_shape[0]:
            # Shape is (H, W, C)
            C = viewport_shape[2]
        else:
            # Shape is (C, H, W)
            C = viewport_shape[0]
            
        n_stack = C // 3  # Assuming 3 channels per frame
        logger.info(f"Inferred n_stack={n_stack} from viewport shape={viewport_shape}")

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Get correct shape for sample input based on viewport shape
        if len(viewport_shape) == 3 and viewport_shape[2] < viewport_shape[0]:
            # For (H, W, C) format - need to transpose
            sample_shape = (1, C, viewport_shape[0], viewport_shape[1])
        else:
            # For (C, H, W) format
            sample_shape = (1,) + viewport_shape
            
        with torch.no_grad():
            sample_input = torch.zeros(sample_shape)
            n_flatten = self.cnn(sample_input).shape[1]
        
        proj_vectors_size = observation_space["projectile_vectors"].shape[0]
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + proj_vectors_size, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU()
        )
        logger.info(f"SimpleCNN initialized: n_flatten={n_flatten}, proj_vectors_size={proj_vectors_size}, total_input={n_flatten + proj_vectors_size}")

    def forward(self, observations):
        viewport = observations["viewport"]
        viewport_shape = viewport.shape
        
        # Check if we need to transpose
        if len(viewport_shape) == 4 and viewport_shape[1] > viewport_shape[3]:
            # Input is (batch, H, W, C) - need to transpose to (batch, C, H, W)
            viewport = viewport.permute(0, 3, 1, 2)
        
        viewport = viewport.float() / 255.0
        proj_vectors = observations["projectile_vectors"].float()
        
        cnn_features = self.cnn(viewport)
        combined = torch.cat([cnn_features, proj_vectors], dim=1)
        return self.linear(combined)
    

def make_env():
    env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
    env = Monitor(KungFuWrapper(env))
    return env