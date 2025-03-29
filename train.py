import argparse
import os
import time
import gym
import retro
import numpy as np
import torch
import signal
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit

# Global variable to store the model for Ctrl+C handling
global_model = None
global_model_path = None

class FrameStackWrapper(Wrapper):
    """Custom frame stack wrapper for play mode"""
    def __init__(self, env, n_stack=4):
        super().__init__(env)
        self.n_stack = n_stack
        self.frames = []
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(n_stack, *env.observation_space.shape[:-1]),  # Remove channel dim
            dtype=np.uint8
        )

    def reset(self):
        obs = self.env.reset()
        self.frames = [obs[..., 0]] * self.n_stack  # Use only first channel
        return np.stack(self.frames)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.frames.pop(0)
        self.frames.append(obs[..., 0])  # Use only first channel
        return np.stack(self.frames), reward, done, info

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(14)
        self._actions = [
            [0]*12, [1,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0],
            [1,1,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0], [1,0,0,0,0,0,0,1,0,0,0,0],
            [0,1,0,0,0,0,1,0,0,0,0,0], [1,0,0,0,1,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,1,0,0,0,0,0], [1,0,0,0,0,0,1,0,0,0,0,0],
            [0,1,0,0,0,0,0,1,0,0,0,0]
        ]

    def action(self, action):
        return self._actions[action]

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        self.last_score = 0
        self.last_x = 0
        self.last_health = 3
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        current_score = info.get('score', 0)
        current_x = info.get('x_pos', 0)
        health = info.get('health', 3)
        
        reward = (
            (current_score - self.last_score) * 0.1 +
            max(0, current_x - self.last_x) * 0.3 +
            (health - self.last_health) * 2.0 -
            0.01  # Time penalty
        )
        
        self.last_score = current_score
        self.last_x = current_x
        self.last_health = health
        
        done = done or health <= 0
        return obs, reward, done, info

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps)
    
    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

class SaveCheckpointCallback(BaseCallback):
    """
    Callback for saving model checkpoints at regular intervals.
    """
    def __init__(self, save_freq, save_path, name_prefix="model"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.timesteps_elapsed = 0
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self):
        self.timesteps_elapsed += self.training_env.num_envs
        
        if self.timesteps_elapsed >= self.save_freq:
            self.timesteps_elapsed = 0
            timesteps = self.num_timesteps
            
            # Create checkpoint filename with timesteps
            checkpoint_path = f"{self.save_path}_{timesteps}.zip"
            
            # Save the model
            self.model.save(checkpoint_path)
            print(f"\nCheckpoint saved: {checkpoint_path}")
            
        return True

def signal_handler(sig, frame):
    """Handle Ctrl+C by saving the model before exiting"""
    if global_model is not None and global_model_path is not None:
        print("\nCaught Ctrl+C! Saving model before exiting...")
        
        # Create emergency save filename with timestamp
        emergency_path = f"{global_model_path}_emergency_{int(time.time())}.zip"
        global_model.save(emergency_path)
        print(f"Emergency save completed: {emergency_path}")
    
    print("Exiting...")
    sys.exit(0)

def make_env(render=False, test=False):
    env = retro.make(
        "KungFu-Nes",
        use_restricted_actions=retro.Actions.ALL
    )
    if render:
        env.render()
    env = KungFuDiscreteWrapper(env)
    env = KungFuRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=5000)
    if test:  # Only apply frame stack in test mode
        env = FrameStackWrapper(env, n_stack=4)
    return env

def train(args):
    global global_model, global_model_path
    
    env = make_vec_env(
        lambda: make_env(args.render),
        n_envs=4 if not args.render else 1,
        vec_env_cls=SubprocVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    model_path = args.model_path
    base_model_path = model_path + ".zip" if not model_path.endswith(".zip") else model_path
    
    # Set global model path for signal handler
    global_model_path = model_path
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(os.path.dirname(model_path) or ".", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Set checkpoint base path
    checkpoint_base = os.path.join(checkpoints_dir, os.path.basename(model_path))
    
    if args.resume and os.path.exists(base_model_path):
        model = PPO.load(base_model_path, env=env, device=device)
        print(f"Resuming training from {base_model_path}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            learning_rate=2.5e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log="./logs"
        )
        print("Starting new training session")
    
    # Set global model for signal handler
    global_model = model

    # Create callbacks
    callbacks = [
        ProgressBarCallback(args.timesteps),
        SaveCheckpointCallback(
            save_freq=args.save_freq,
            save_path=checkpoint_base
        )
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=os.path.basename(args.model_path)
        )
    finally:
        model.save(base_model_path)
        env.close()
        print(f"Final model saved to {base_model_path}")

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")
    
    # Use test mode to apply frame stacking
    env = make_env(render=args.render, test=True)
    
    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if args.render:
                env.render()
                time.sleep(0.01)  # Small delay to make visualization smoother
            if done:
                print(f"Episode finished. Score: {info.get('score', 0)}")
                obs = env.reset()
    except KeyboardInterrupt:
        print("\nPlay session ended")
    finally:
        env.close()

def list_checkpoints(args):
    # Get the directory containing checkpoints
    model_path = args.model_path
    checkpoints_dir = os.path.join(os.path.dirname(model_path) or ".", "checkpoints")
    
    if not os.path.exists(checkpoints_dir):
        print(f"No checkpoints directory found at {checkpoints_dir}")
        return
    
    # Get base name of the model
    base_name = os.path.basename(model_path)
    
    # List all files matching the pattern
    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith(base_name) and f.endswith(".zip")]
    emergency_saves = [f for f in os.listdir(".") if f.startswith(base_name) and "emergency" in f and f.endswith(".zip")]
    
    if not checkpoints and not emergency_saves:
        print(f"No checkpoints or emergency saves found for {base_name}")
        return
    
    if checkpoints:
        # Sort by timesteps (numerical part after the last underscore before .zip)
        checkpoints.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        print(f"Available checkpoints for {base_name}:")
        for i, checkpoint in enumerate(checkpoints):
            timesteps = checkpoint.split("_")[-1].split(".")[0]
            print(f"  {i+1}. {checkpoint} ({int(timesteps):,} timesteps)")
    
    if emergency_saves:
        # Sort by timestamp
        emergency_saves.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        
        print(f"\nEmergency saves for {base_name}:")
        for i, save in enumerate(emergency_saves):
            timestamp = save.split("_")[-1].split(".")[0]
            save_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(timestamp)))
            print(f"  {i+1}. {save} (saved on {save_time})")

if __name__ == "__main__":
    # Register the signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true", help="Run in play mode")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Number of training timesteps")
    parser.add_argument("--model_path", default="kungfu_ppo", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--save_freq", type=int, default=4000, help="Frequency for saving checkpoints (in timesteps)")
    parser.add_argument("--list", action="store_true", help="List available checkpoints for the model")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args)
    elif args.play:
        play(args)
    else:
        train(args)