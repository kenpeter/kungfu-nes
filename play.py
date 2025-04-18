import os
import retro
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from kungfu_env import SimpleCNN, KUNGFU_MAX_ENEMIES, MAX_PROJECTILES
from gymnasium import spaces

class PlaybackWrapper(retro.RetroEnv):
    """Wrapper for playback with MultiBinary action space and dictionary observation space."""
    def __init__(self):
        super().__init__(game='KungFu-Nes', render_mode='human')
        # Define MultiBinary action space to match the trained model
        self.action_space = spaces.MultiBinary(9)
        # Define dictionary observation space to match training
        self.observation_space = spaces.Dict({
            'viewport': spaces.Box(low=0, high=255, shape=(224, 240, 3), dtype=np.uint8),
            'enemy_vector': spaces.Box(low=-255, high=255, shape=(KUNGFU_MAX_ENEMIES * 2,), dtype=np.float32),
            'combat_status': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'projectile_vectors': spaces.Box(low=-255, high=255, shape=(MAX_PROJECTILES * 4,), dtype=np.float32),
            'enemy_proximity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'boss_info': spaces.Box(low=-255, high=255, shape=(3,), dtype=np.float32),
            'closest_enemy_direction': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })
        self.max_enemies = KUNGFU_MAX_ENEMIES
        self.max_projectiles = MAX_PROJECTILES
        self.prev_frame = None
        self.last_hp = 0
        self.last_hp_change = 0

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.prev_frame = None
        self.last_hp = float(self.get_ram()[0x04A6])
        self.last_hp_change = 0
        return self._get_obs(obs)

    def step(self, action):
        # Convert MultiBinary(9) action to 12-element retro action
        action = np.array(action, dtype=np.uint8)
        if action.shape != (9,):
            raise ValueError(f"Expected action shape (9,), got {action.shape}")
        # Pad 9-element action to 12-element array (fill unused buttons with 0)
        retro_action = np.zeros(12, dtype=np.uint8)
        retro_action[:9] = action
        obs, reward, terminated, truncated, info = super().step(retro_action)
        return self._get_obs(obs), reward, terminated, truncated, info

    def _get_obs(self, obs):
        # Resize observation to (224, 240) to match training
        obs = cv2.resize(obs, (240, 224), interpolation=cv2.INTER_AREA)
        ram = self.get_ram()
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
        hp = float(ram[0x04A6])
        hp_change_rate = (hp - self.last_hp) / 255.0
        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        return {
            "viewport": obs.astype(np.uint8),
            "enemy_vector": np.array(enemy_info, dtype=np.float32),
            "combat_status": np.array([hp / 255.0, hp_change_rate], dtype=np.float32),
            "projectile_vectors": np.array(self._detect_projectiles(obs), dtype=np.float32),
            "enemy_proximity": np.array([1.0 if any(abs(enemy_x - hero_x) <= 20 for enemy_x in [int(ram[addr]) for addr in [0x008E, 0x008F, 0x0090, 0x0091, 0x0092]] if enemy_x != 0) else 0.0], dtype=np.float32),
            "boss_info": self._update_boss_info(ram),
            "closest_enemy_direction": np.array([closest_enemy_direction], dtype=np.float32)
        }

    def _update_boss_info(self, ram):
        stage = int(ram[0x0058])
        if stage == 5:
            boss_pos_x = int(ram[0x0093])
            boss_action = int(ram[0x004E])
            boss_hp = int(ram[0x04A5])
            return np.array([boss_pos_x - int(ram[0x0094]), boss_action / 255.0, boss_hp / 255.0], dtype=np.float32)
        return np.zeros(3, dtype=np.float32)

    def _detect_projectiles(self, obs):
        frame = obs
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
            hero_x = int(self.get_ram()[0x0094])
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

def play():
    # Initialize the environment
    env = PlaybackWrapper()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env, skip=True)

    model_path = 'models/kungfu_ppo/kungfu_ppo.zip'

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    # Match the policy_kwargs from training
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 256, "n_stack": 4},
        "net_arch": dict(pi=[128, 128], vf=[256, 256])
    }
    custom_objects = {"policy_kwargs": policy_kwargs}

    try:
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)
    except Exception as e:
        print(f"Failed to load model: {e}")
        env.close()
        return

    obs = env.reset()

    print("Playing with trained model. Press Ctrl+C to stop.")
    try:
        while True:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            env.render()
            if done:
                obs = env.reset()
    except KeyboardInterrupt:
        print("Stopping playback.")
    except Exception as e:
        print(f"Error during playback: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    play()