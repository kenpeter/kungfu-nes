import gymnasium as gym
import numpy as np
from gymnasium import spaces, Wrapper
import cv2
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import retro

# Constants for future scalability
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

# Define minimal observation space as an empty-like spaces.Dict
KUNGFU_OBSERVATION_SPACE = spaces.Dict({
    "placeholder": spaces.Box(low=0, high=0, shape=(), dtype=np.float32)
})
# Note: Add observation components (e.g., viewport, enemy_vector) here for scaling

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Get initial observation for viewport size
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
        
        # Use minimal observation space
        self.observation_space = KUNGFU_OBSERVATION_SPACE
        
        # State tracking for rewards and projectile detection
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

    def reset(self, seed=None, options=None, **kwargs):
        result = self.env.reset(seed=seed, options=options, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self.last_hp = float(self.env.get_ram()[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * KUNGFU_MAX_ENEMIES
        self.total_steps = 0
        self.prev_frame = None
        self.last_projectile_distances = [float('inf')] * MAX_PROJECTILES
        self.survival_reward_total = 0
        return self._get_obs(obs), info

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1

        # Execute action
        result = self.env.step(self.actions[action])
        if len(result) == 5:  # New API
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:  # Old API
            obs, reward, done, info = result
            terminated = done
            truncated = False

        ram = self.env.get_ram()
        hp = float(ram[0x04A6])
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091]), int(ram[0x0092])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)

        # Simplified reward based on RAM values and projectile detection
        reward = 0
        hp_change_rate = (hp - self.last_hp) / 255.0
        if hp_change_rate < 0:
            reward += (hp_change_rate ** 2) * 50
        else:
            reward += hp_change_rate * 5

        reward += enemy_hit * 10
        if done:
            reward -= 50

        # Projectile-based dodge reward
        projectile_info = self._detect_projectiles(obs)
        projectile_distances = [projectile_info[i] for i in range(0, len(projectile_info), 4)]
        dodge_reward = 0
        for i, (curr_dist, last_dist) in enumerate(zip(projectile_distances, self.last_projectile_distances)):
            if last_dist < 20 and curr_dist > last_dist:
                if action in [5, 6]:  # Crouch or Left dodge
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances

        # Action-based rewards
        if hp_change_rate < 0 and action in [5, 6]:  # Defensive actions
            reward += 5
        if action in [1, 8, 11, 12]:  # Punch, Kick, Crouch Kick, Crouch Punch
            reward += 2.0
        elif action in [9, 10]:  # Punch + Kick, Jump + Right
            reward -= 0.5

        # Action entropy for exploration
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

        # Info dictionary for debugging
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names,
            "dodge_reward": dodge_reward,
            "survival_reward_total": self.survival_reward_total,
            "raw_reward": reward,
            "normalized_reward": normalized_reward
        })

        return self._get_obs(obs), normalized_reward, terminated, truncated, info

    def _get_obs(self, obs):
        # Return minimal observation
        # Note: Add observation processing (e.g., viewport, enemy_vector) here for scaling
        return {"placeholder": np.array(0.0, dtype=np.float32)}

    def _detect_projectiles(self, obs):
        # Note: Currently used for rewards only; can be integrated into observation space later
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
            projectile_info = projectile_info[:MAX_PROJECTILES * 4]
            while len(projectile_info) < MAX_PROJECTILES * 4:
                projectile_info.append(0)
            self.prev_frame = frame.copy()
            return projectile_info
        self.prev_frame = frame.copy()
        return [0] * (MAX_PROJECTILES * 4)

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, n_stack=4):
        super().__init__(observation_space, features_dim)
        # Note: Currently processes minimal placeholder; update for scaled observation space
        self.linear = nn.Sequential(
            nn.Linear(1, 128),  # Input is single placeholder value
            nn.ReLU(),
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # Process minimal placeholder
        # Note: Add viewport and vector processing here when scaling
        placeholder = observations["placeholder"].float().unsqueeze(-1)
        return self.linear(placeholder)

def make_env():
    env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
    env = Monitor(KungFuWrapper(env))
    return env