# play.py
import retro
import numpy as np
from stable_baselines3 import PPO
from kungfu_env import KungFuWrapper, SimpleCNN

def play():
    base_env = retro.make('KungFu-Nes')
    env = KungFuWrapper(base_env)
    env = DummyVecEnv([lambda: env])  # Vectorize
    env = VecFrameStack(env, n_stack=12)  # Match training
    model_path = 'models/kungfu_ppo/kungfu_ppo_best.zip'
    
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return
    
    policy_kwargs = {"features_extractor_class": SimpleCNN}
    model = PPO.load(model_path, env=env, custom_objects={"policy_kwargs": policy_kwargs})
    obs = env.reset()
if __name__ == "__main__":
    play()