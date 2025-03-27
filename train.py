import argparse
import os
import time
import gym
import retro
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # 12 essential actions for combat + movement
        self.action_space = spaces.Discrete(12)
        self._actions = [
            [0]*12,                          # 0: NOOP
            [1,0,0,0,0,0,0,0,0,0,0,0],       # 1: PUNCH (A)
            [0,1,0,0,0,0,0,0,0,0,0,0],       # 2: JUMP (B)
            [1,1,0,0,0,0,0,0,0,0,0,0],       # 3: JUMP+KICK
            [0,0,0,0,1,0,0,0,0,0,0,0],       # 4: UP (duck)
            [0,0,0,0,0,1,0,0,0,0,0,0],       # 5: DOWN
            [0,0,0,0,0,0,1,0,0,0,0,0],       # 6: LEFT (backstep)
            [0,0,0,0,0,0,0,1,0,0,0,0],       # 7: RIGHT (advance)
            [1,0,0,0,0,0,0,1,0,0,0,0],       # 8: ADVANCE+PUNCH
            [0,1,0,0,0,0,1,0,0,0,0,0],       # 9: BACKSTEP+JUMP
            [1,0,0,0,1,0,0,0,0,0,0,0],       # 10: DUCK+PUNCH
            [0,0,0,0,1,0,1,0,0,0,0,0]        # 11: DUCK+BACK (dodge)
        ]

    def action(self, action):
        return self._actions[action]

class KungFuRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_x = 0
        self.last_health = 3
        self.combo_counter = 0
        self.last_enemy_count = 2  # Starting enemies
        
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 1. Combat rewards
        current_score = info.get('score', 0)
        score_reward = (current_score - self.last_score) * 0.1
        self.last_score = current_score
        
        # 2. Movement rewards
        current_x = info.get('x_pos', 0)
        progress_reward = max(0, current_x - self.last_x) * 0.5  # Only reward right movement
        self.last_x = current_x
        
        # 3. Health rewards
        health = info.get('health', 3)
        health_reward = (health - self.last_health) * 2.0  # Big reward for health gain
        self.last_health = health
        
        # 4. Combat effectiveness
        current_enemies = info.get('enemies', 0)
        if current_enemies < self.last_enemy_count:
            reward += 5.0  # Big reward for defeating enemies
        self.last_enemy_count = current_enemies
        
        # 5. Action-specific rewards
        if action in [1, 3, 8, 10]:  # Punching actions
            reward += 0.2  # Encourage attacking
        
        # 6. Survival penalty
        reward -= 0.01  # Small time penalty to encourage progress
        
        # Combine all rewards
        reward += score_reward + progress_reward + health_reward
        
        # Check for game over
        if health <= 0:
            done = True
            reward -= 10.0  # Big penalty for dying
            
        return obs, reward, done, info

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
        fps = self.num_timesteps / max(1, elapsed)
        remaining = (elapsed / max(1, self.num_timesteps)) * (self.total_timesteps - self.num_timesteps)
        self.pbar.set_postfix({
            "Time Left": f"{remaining/60:.1f}min",
            "FPS": f"{fps:.1f}",
            "Avg Reward": f"{np.mean(self.model.ep_info_buffer[-10:]) if self.model.ep_info_buffer else 0:.1f}"
        })
        return True
        
    def _on_training_end(self):
        self.pbar.close()

def make_env(render=False):
    env = retro.make("KungFu-Nes", use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    env = KungFuRewardWrapper(env)
    if render:
        env.render()
    return env

def train(args):
    env = DummyVecEnv([lambda: make_env(args.render)])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames for temporal info
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device.upper()}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if args.resume and os.path.exists(args.model_path):
        model = PPO.load(args.model_path, env=env, device=device)
        print(f"Resuming training from {args.model_path}")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            learning_rate=3e-4,
            n_steps=4096,  # Larger rollout buffer
            batch_size=128,
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
            tb_log_name="kungfu_ppo_enhanced"
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        model.save(args.model_path)
        env.close()
        print(f"\nModel saved to {args.model_path}")

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = PPO.load(args.model_path, device=device)
    env = make_env(render=True)
    
    print(f"\nPlaying with {device.upper()}")
    obs = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        
        if args.debug:
            print(f"Action: {action} | Reward: {reward:.2f} | Total: {total_reward:.1f}")
            print(f"Position: {info.get('x_pos', 0)} | Health: {info.get('health', 3)}")
            
        if done:
            print(f"Episode ended! Total reward: {total_reward:.1f}")
            total_reward = 0
            obs = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kung Fu Master RL Agent")
    parser.add_argument("--play", action="store_true", help="Run in play mode")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--debug", action="store_true", help="Show debug info")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--timesteps", type=int, default=30_000, help="Total timesteps")
    parser.add_argument("--model_path", default="kungfu_ppo_enhanced", help="Model path")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration")
    
    args = parser.parse_args()
    
    if args.play:
        play(args)
    else:
        train(args)