import os
import numpy as np
import gymnasium as gym
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.vec_env import VecTransposeImage
from kungfu_env import KungFuWrapper, SimpleCNN

def play():
    # Create the environment
    env = retro.make('KungFu-Nes', render_mode='human')
    # Use the existing KungFuWrapper instead of creating a new wrapper
    env = KungFuWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env, skip=True)

    model_path = 'models/kungfu_ppo/kungfu_ppo.zip'

    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return

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

    # Handle the reset operation
    result = env.reset()
    
    # Handle various return formats from vectorized env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs = result[0]  # (obs, info) format
    else:
        obs = result
    
    print("Playing with trained model. Press Ctrl+C to stop.")
    try:
        while True:
            action, _ = model.predict(obs)
            result = env.step(action)
            
            # Handle different return formats
            if len(result) == 5:  # Newer Gymnasium format
                obs, rewards, terminations, truncations, infos = result
                terminated = terminations[0] if hasattr(terminations, "__getitem__") else terminations
                truncated = truncations[0] if hasattr(truncations, "__getitem__") else truncations
            else:  # Older Gym format
                obs, rewards, dones, infos = result
                terminated = dones[0] if hasattr(dones, "__getitem__") else dones
                truncated = False
                
            env.render()
            
            if terminated or truncated:
                result = env.reset()
                # Handle various return formats
                if isinstance(result, tuple) and len(result) == 2:
                    obs = result[0]  # (obs, info) format
                else:
                    obs = result
    except KeyboardInterrupt:
        print("Stopping playback.")
    except Exception as e:
        print(f"Error during playback: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    play()