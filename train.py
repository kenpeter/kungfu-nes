import argparse
import time
import os
import gym
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(8)
        self._actions = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # NOOP
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A (Punch)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Jump)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # UP
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # DOWN
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # LEFT
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # RIGHT
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # A+B
        ]

    def action(self, action):
        return self._actions[action]

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = time.time()
        
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")
        
    def _on_step(self):
        self.pbar.update(1)
        elapsed = time.time() - self.start_time
        remaining = (elapsed / max(1, self.num_timesteps)) * (self.total_timesteps - self.num_timesteps)
        self.pbar.set_postfix({
            "Time Left": f"{remaining/60:.1f}min",
            "Reward": f"{np.mean(self.model.ep_info_buffer[-10:]) if self.model.ep_info_buffer else 0:.1f}"
        })
        return True
        
    def _on_training_end(self):
        self.pbar.close()

def make_env(render=False):
    env = retro.make("KungFu-Nes", use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    if render:
        env.render()
    return env

def train(args):
    env = DummyVecEnv([lambda: make_env(args.render)])
    
    if args.resume and os.path.exists(args.model_path):
        model = PPO.load(args.model_path, env=env)
        print(f"Resuming training from {args.model_path}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            tensorboard_log="./logs",
        )

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[ProgressBarCallback(args.timesteps)],
            tb_log_name="kungfu_ppo"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving model...")
    finally:
        model.save(args.model_path)
        env.close()
        print(f"Model saved to {args.model_path}")

def play(args):
    model = PPO.load(args.model_path)
    env = make_env(render=True)
    
    obs = env.reset()
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
        if args.debug:
            print(f"Reward: {reward:.2f} | Info: {info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play Kung Fu Master")
    parser.add_argument("--play", action="store_true", help="Run in play mode")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total timesteps")
    parser.add_argument("--model_path", default="kungfu_ppo", help="Model path")
    
    args = parser.parse_args()
    
    if args.play:
        play(args)
    else:
        train(args)