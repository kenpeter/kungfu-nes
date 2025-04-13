import os
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from kungfu_env import KungFuWrapper, SimpleCNN, KUNGFU_MAX_ENEMIES, MAX_PROJECTILES
from gym import spaces

class PlayKungFuWrapper(KungFuWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Define the same observation space as in train.py
        self.observation_space = spaces.Dict({
            'viewport': spaces.Box(low=0, high=255, shape=(224, 240, 3), dtype=np.uint8),
            'enemy_vector': spaces.Box(low=-255, high=255, shape=(KUNGFU_MAX_ENEMIES * 2,), dtype=np.float32),
            'combat_status': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'projectile_vectors': spaces.Box(low=-255, high=255, shape=(MAX_PROJECTILES * 4,), dtype=np.float32),
            'enemy_proximity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'boss_info': spaces.Box(low=-255, high=255, shape=(3,), dtype=np.float32),
            'closest_enemy_direction': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })

    def reset(self):
        obs = super().reset()
        # Ensure reset returns a dictionary matching the observation space
        return {
            'viewport': obs,  # Assuming KungFuWrapper returns the frame
            'enemy_vector': np.zeros(KUNGFU_MAX_ENEMIES * 2, dtype=np.float32),
            'combat_status': np.zeros(2, dtype=np.float32),
            'projectile_vectors': np.zeros(MAX_PROJECTILES * 4, dtype=np.float32),
            'enemy_proximity': np.zeros(1, dtype=np.float32),
            'boss_info': np.zeros(3, dtype=np.float32),
            'closest_enemy_direction': np.zeros(1, dtype=np.float32),
        }

    def step(self, action):
        obs, reward, done, info = super().step(action)
        # Ensure step returns a dictionary matching the observation space
        return (
            {
                'viewport': obs,
                'enemy_vector': np.zeros(KUNGFU_MAX_ENEMIES * 2, dtype=np.float32),
                'combat_status': np.zeros(2, dtype=np.float32),
                'projectile_vectors': np.zeros(MAX_PROJECTILES * 4, dtype=np.float32),
                'enemy_proximity': np.zeros(1, dtype=np.float32),
                'boss_info': np.zeros(3, dtype=np.float32),
                'closest_enemy_direction': np.zeros(1, dtype=np.float32),
            },
            reward,
            done,
            info
        )

def play():
    # Initialize the base environment
    base_env = retro.make('KungFu-Nes')
    env = PlayKungFuWrapper(base_env)
    env = DummyVecEnv([lambda: env])  # Vectorize for compatibility with PPO
    env = VecFrameStack(env, n_stack=4, channels_order='last')  # Match training setup
    env = VecTransposeImage(env, skip=True)  # Match training's VecTransposeImage

    model_path = 'models/kungfu_ppo/kungfu_ppo_mixed_bak.zip'  # Updated to match train.py's default

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

    # Match the policy_kwargs from training
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 512, "n_stack": 4},
        "net_arch": dict(pi=[256, 256, 128], vf=[512, 512, 256])
    }
    custom_objects = {"policy_kwargs": policy_kwargs}
    model = PPO.load(model_path, env=env, custom_objects=custom_objects)

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
    finally:
        env.close()

if __name__ == "__main__":
    play()