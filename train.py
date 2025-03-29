import argparse
import os
import time
import gym
import retro
import numpy as np
import torch
import signal
import sys
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for Ctrl+C handling
global_model = None
global_model_path = None

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(14)
        # NES Kung Fu button mapping: [B, A, Select, Start, Up, Down, Left, Right, B+A, etc.]
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No action
            [1,0,0,0,0,0,0,0,0,0,0,0],  # B (Punch)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # A (Kick)
            [1,1,0,0,0,0,0,0,0,0,0,0],  # B+A
            [0,0,0,0,1,0,0,0,0,0,0,0],  # Up
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Down
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Left
            [0,0,0,0,0,0,0,1,0,0,0,0],  # Right
            [1,0,0,0,0,0,0,1,0,0,0,0],  # B+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # A+Left
            [1,0,0,0,1,0,0,0,0,0,0,0],  # B+Up
            [0,0,0,0,1,0,1,0,0,0,0,0],  # Up+Left
            [1,0,0,0,0,0,1,0,0,0,0,0],  # B+Left
            [0,1,0,0,0,0,0,1,0,0,0,0]   # A+Right
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
        
        # Reward moving left (decreasing x), penalize moving right or staying still
        x_delta = self.last_x - current_x  # Positive if moving left
        reward = (
            (current_score - self.last_score) * 0.1 +  # Reward score increase
            max(0, x_delta) * 0.5 +                    # Reward moving left
            (health - self.last_health) * 2.0 -        # Health changes
            0.01                                       # Time penalty
        )
        
        # Debug logging
        logging.debug(f"x_pos: {current_x}, x_delta: {x_delta}, reward: {reward}")
        
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
    def __init__(self, save_freq, save_path, name_prefix="model"):
        super().__init__()
        self.save_freq = save_freq
        # Ensure save_path has a directory; default to current dir if none specified
        if os.path.dirname(save_path) == '':
            self.save_path = os.path.join(".", save_path)
        else:
            self.save_path = save_path
        self.name_prefix = name_prefix
        self.timesteps_elapsed = 0
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self):
        self.timesteps_elapsed += self.training_env.num_envs
        
        if self.timesteps_elapsed >= self.save_freq:
            self.timesteps_elapsed = 0
            checkpoint_path = f"{self.save_path}.zip"
            self.model.save(checkpoint_path)
            logging.info(f"Checkpoint saved (overwritten): {checkpoint_path} at {self.num_timesteps:,} timesteps")
            
        return True

def signal_handler(sig, frame):
    if global_model is not None and global_model_path is not None:
        logging.info("Caught Ctrl+C! Saving model before exiting...")
        emergency_path = f"{global_model_path}_emergency_{int(time.time())}.zip"
        global_model.save(emergency_path)
        logging.info(f"Emergency save completed: {emergency_path}")
    logging.info("Exiting...")
    sys.exit(0)

def make_env(render=False):
    env = retro.make("KungFu-Nes", use_restricted_actions=retro.Actions.ALL)
    if render:
        env.render()
    env = KungFuDiscreteWrapper(env)
    env = KungFuRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=5000)
    return env

def train(args):
    global global_model, global_model_path
    
    logging.info("Starting training")
    
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
    global_model_path = model_path
    
    if args.resume and os.path.exists(base_model_path):
        model = PPO.load(base_model_path, env=env, device=device)
        logging.info(f"Resuming training from {base_model_path}")
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
            ent_coef=0.01,  # Increase exploration
            tensorboard_log="./logs"
        )
        logging.info("Starting new training session")
    
    global_model = model

    callbacks = [
        ProgressBarCallback(args.timesteps),
        SaveCheckpointCallback(save_freq=args.save_freq, save_path=model_path)
    ]

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=os.path.basename(args.model_path)
        )
        logging.info("Training completed successfully")
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    finally:
        logging.info("Saving final model")
        model.save(base_model_path)
        try:
            env.close()
            logging.info("Environment closed successfully")
        except BrokenPipeError:
            logging.warning("Could not cleanly close subprocesses (BrokenPipeError)")
        logging.info(f"Final model saved to {base_model_path}")
        sys.exit(0)

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = args.model_path
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    logging.info(f"Loaded model from {model_path}")
    
    env = make_vec_env(
        lambda: make_env(render=args.render),
        n_envs=1,
        vec_env_cls=DummyVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    
    obs = env.reset()
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if args.render:
                env.render()
                time.sleep(0.01)
            if done:
                logging.info(f"Episode finished. Score: {info[0].get('score', 0)}")
                obs = env.reset()
    except KeyboardInterrupt:
        logging.info("Play session ended by user")
    finally:
        env.close()
        logging.info("Play environment closed")

def list_checkpoints(args):
    model_path = args.model_path
    checkpoint_path = model_path + ".zip" if not model_path.endswith(".zip") else model_path
    
    emergency_saves = [f for f in os.listdir(".") if f.startswith(os.path.basename(model_path)) and "emergency" in f and f.endswith(".zip")]
    
    if os.path.exists(checkpoint_path):
        print(f"Current checkpoint exists: {checkpoint_path}")
        print(f"Last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(checkpoint_path)))}")
    else:
        print(f"No checkpoint found at {checkpoint_path}")
    
    if emergency_saves:
        emergency_saves.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
        print(f"\nEmergency saves for {os.path.basename(model_path)}:")
        for i, save in enumerate(emergency_saves):
            timestamp = save.split("_")[-1].split(".")[0]
            save_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(int(timestamp)))
            print(f"  {i+1}. {save} (saved on {save_time})")
    else:
        print(f"\nNo emergency saves found for {os.path.basename(model_path)}")

if __name__ == "__main__":
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