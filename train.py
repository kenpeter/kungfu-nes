import argparse
import os
import time
import gym
import retro
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit

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

def make_env(render=False, test=False):
    # Remove render_mode for retro compatibility
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
    env = make_vec_env(
        lambda: make_env(args.render),
        n_envs=4 if not args.render else 1,
        vec_env_cls=DummyVecEnv
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    
    model_path = args.model_path + ".zip" if not args.model_path.endswith(".zip") else args.model_path
    
    if args.resume and os.path.exists(model_path):
        model = PPO.load(model_path, env=env, device=device)
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

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[ProgressBarCallback(args.timesteps)],
            tb_log_name=os.path.basename(args.model_path)
        )
    finally:
        model.save(model_path)
        env.close()

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = args.model_path + ".zip" if not args.model_path.endswith(".zip") else args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    
    # Use test mode to apply frame stacking
    env = make_env(render=args.render, test=True)
    
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        if args.render:
            env.render()
        if done:
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--model_path", default="kungfu_ppo")
    parser.add_argument("--cuda", action="store_true")
    
    args = parser.parse_args()
    
    if args.play:
        play(args)
    else:
        train(args)