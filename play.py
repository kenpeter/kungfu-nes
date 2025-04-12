# play.py
import os  # Added for os.path.exists
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from kungfu_env import KungFuWrapper, SimpleCNN

def play():
    base_env = retro.make('KungFu-Nes')
    env = KungFuWrapper(base_env)
    env = DummyVecEnv([lambda: env])  # Vectorize for compatibility with PPO
    env = VecFrameStack(env, n_stack=4)  # Match training setup with 12 stacked frames
    model_path = 'models/kungfu_ppo/kungfu_ppo_best.zip'
    
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return
    
    # Match the policy_kwargs from training
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "net_arch": dict(pi=[256, 256, 128], vf=[512, 512, 256])  # Same as train.py
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