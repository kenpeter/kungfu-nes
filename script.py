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
from stable_baselines3.common.utils import set_random_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log')]
)

# Global variables for signal handler
global_model = None
global_model_path = None

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(9)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # B (kick)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # A (punch)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # B+Left
            [1,0,0,0,0,0,0,0,1,0,0,0],  # B+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # A+Left
            [0,1,0,0,0,0,0,0,1,0,0,0]   # A+Right
        ]

    def action(self, action):
        return self._actions[action]

class FrameSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset()
        self.enemy_positions = {
            'player_x': 0x0050,  # Player X position
            'player_y': 0x0053,  # Player Y position
            'enemy1_x': 0x036E,  # Enemy 1 X position
            'enemy1_y': 0x0371,  # Enemy 1 Y position
            'enemy2_x': 0x0372,  # Enemy 2 X position
            'enemy2_y': 0x0375   # Enemy 2 Y position
        }
        self.enemy_proximity_threshold = 50  # Increased from 30 to detect enemies from further away
        self.no_attack_penalty_cooldown = 0
        self.consecutive_attacks = 0
        self.frames_since_last_enemy_hit = 0
        self.enemy_health_previous = {}
        self.successful_hits = 0

    def reset(self):
        self.last_score = 0
        self.last_x = 0
        self.last_health = 46
        self.time_without_progress = 0
        self.death_penalty_applied = False
        self.frames_since_last_attack = 0
        self.enemy_nearby_frames = 0
        self.consecutive_attacks = 0
        self.frames_since_last_enemy_hit = 0
        self.enemy_health_previous = {}
        self.successful_hits = 0
        return self.env.reset()

    def _is_enemy_nearby(self, ram):
        player_x = ram[self.enemy_positions['player_x']]
        player_y = ram[self.enemy_positions['player_y']]
        for i in range(1, 3):
            enemy_x = ram[self.enemy_positions[f'enemy{i}_x']]
            enemy_y = ram[self.enemy_positions[f'enemy{i}_y']]
            if enemy_x > 0 and enemy_y > 0:  # Check if enemy is active
                x_dist = abs(player_x - enemy_x)
                y_dist = abs(player_y - enemy_y)
                if x_dist < self.enemy_proximity_threshold and y_dist < 25:  # Increased from 20
                    return True
        return False

    def _detect_enemy_hit(self, ram, info):
        # Check if score increased, which often indicates a successful hit
        if info.get('score', 0) > self.last_score:
            self.successful_hits += 1
            return True
        
        # Try to detect hits by monitoring enemy state changes (simplified)
        # In a real implementation, you'd track specific memory locations for enemy health/state
        for i in range(1, 3):
            enemy_x_addr = self.enemy_positions[f'enemy{i}_x']
            enemy_y_addr = self.enemy_positions[f'enemy{i}_y']
            
            # Check if enemy was present before but now disappeared
            if (enemy_x_addr in self.enemy_health_previous and 
                self.enemy_health_previous[enemy_x_addr] > 0 and
                ram[enemy_x_addr] == 0):
                self.successful_hits += 1
                return True
                
        return False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        ram = self.env.get_ram()
        
        current_score = info.get('score', 0)
        current_x = info.get('x_pos', 0)
        health = info.get('health', 46)
        enemy_nearby = self._is_enemy_nearby(ram)
        enemy_hit = self._detect_enemy_hit(ram, info)
        
        # Track enemy positions for hit detection in the next frame
        for i in range(1, 3):
            self.enemy_health_previous[self.enemy_positions[f'enemy{i}_x']] = ram[self.enemy_positions[f'enemy{i}_x']]
        
        if enemy_nearby:
            self.enemy_nearby_frames += 1
        else:
            self.enemy_nearby_frames = 0
        
        score_delta = current_score - self.last_score
        x_delta = current_x - self.last_x
        health_delta = health - self.last_health
        
        # Initialize reward
        reward = 0
        
        # Reward for score increases (defeating enemies)
        if score_delta > 0:
            reward += score_delta * 200.0  # Increased from 150
            reward += 150.0  # Bonus for defeating enemies, increased from 100
        
        # Reward for forward progress
        if x_delta > 0:
            reward += x_delta * 10.0  # Reduced from 20 to prioritize combat over movement
        if x_delta < 0:
            reward += x_delta * 2.0  # Penalty for moving backward remains the same
        
        # Penalty for taking damage
        if health_delta < 0:
            reward += health_delta * 5.0
        
        # Small time penalty to encourage faster completion
        reward -= 0.05
        
        # Death penalty
        if health <= 0 and not self.death_penalty_applied:
            reward -= 100  # Increased from 50
            self.death_penalty_applied = True
        
        # Penalty for being stuck
        if abs(x_delta) < 1:
            self.time_without_progress += 1
            if self.time_without_progress > 100:
                reward -= 0.8  # Increased from 0.5
        else:
            self.time_without_progress = 0
        
        # Enhanced rewards for attacking, especially when enemies are nearby
        is_attack_action = action in [3, 4, 5, 6, 7, 8]
        
        if is_attack_action:
            # Base reward for attacking
            reward += 5.0  # Small reward just for attacking
            self.frames_since_last_attack = 0
            self.consecutive_attacks += 1
            
            # Extra reward for consecutive attacks (combo bonus)
            combo_bonus = min(self.consecutive_attacks * 2, 20)  # Cap at 20
            reward += combo_bonus
            
            if enemy_nearby:
                # Much higher reward for attacking when enemies are nearby
                reward += 50.0  # Increased from 25
                
                # Extra reward for successful hits
                if enemy_hit:
                    reward += 100.0  # Big reward for landing hits
                    self.frames_since_last_enemy_hit = 0
                else:
                    self.frames_since_last_enemy_hit += 1
                    # Gradually decrease reward if attacks aren't landing
                    if self.frames_since_last_enemy_hit > 20:
                        penalty_factor = min(self.frames_since_last_enemy_hit - 20, 30) / 30
                        reward -= 10.0 * penalty_factor
            
            # Reset no-attack penalty cooldown
            self.no_attack_penalty_cooldown = 5  # Reduced from 10 to encourage more frequent attacks
        else:
            # Reset consecutive attack counter if not attacking
            self.consecutive_attacks = 0
            self.frames_since_last_attack += 1
            
            # Increased penalty for not attacking when enemies are nearby
            if enemy_nearby and self.no_attack_penalty_cooldown <= 0:
                # Escalating penalty based on how long enemies have been nearby without attacking
                penalty = min(self.enemy_nearby_frames, 60) / 10
                reward -= 5.0 + penalty  # Increased from 2.0
            elif not enemy_nearby:
                # Small reward for movement/exploration when no enemies around
                reward += 0.5  # Reduced from 1.0 to prioritize combat
        
        if self.no_attack_penalty_cooldown > 0:
            self.no_attack_penalty_cooldown -= 1
        
        self.last_score = current_score
        self.last_x = current_x
        self.last_health = health
        
        done = done or health <= 0
        info['custom_reward'] = reward
        info['successful_hits'] = self.successful_hits
        
        return obs, reward, done, info

class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps, num_envs):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.num_envs = num_envs
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training", unit="step")
    
    def _on_step(self):
        # Update by the number of environments to reflect parallel steps
        self.pbar.update(self.num_envs)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

class RewardLoggerCallback(BaseCallback):
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self):
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-10:])
            avg_length = np.mean(self.episode_lengths[-10:])
            print(f"Timestep: {self.num_timesteps}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
        return True

class ActionDistributionCallback(BaseCallback):
    def __init__(self, log_freq=5000):
        super().__init__()
        self.log_freq = log_freq
        self.actions_taken = []
    
    def _on_step(self):
        if 'actions' in self.locals:
            self.actions_taken.extend(self.locals['actions'])
        
        if self.n_calls % self.log_freq == 0 and len(self.actions_taken) > 0:
            action_counts = np.bincount(self.actions_taken, minlength=9)
            total_actions = len(self.actions_taken)
            action_percentages = (action_counts / total_actions) * 100
            print("\n===== Action Distribution =====")
            action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
            for i, (name, percentage) in enumerate(zip(action_names, action_percentages)):
                print(f"{name}: {percentage:.2f}%")
            print(f"Total Attack Actions: {sum(action_percentages[3:]):.2f}%")
            self.actions_taken = []
        return True

def signal_handler(sig, frame):
    if global_model is not None and global_model_path is not None:
        print("Interrupt received! Saving model...")
        emergency_path = f"{global_model_path}_emergency_{int(time.time())}"
        global_model.save(emergency_path)
        print(f"Emergency save completed: {emergency_path}")
    sys.exit(0)

def make_env(render=False, seed=None):
    env = retro.make(
        game="KungFu-Nes",
        use_restricted_actions=retro.Actions.ALL,
        obs_type=retro.Observations.IMAGE
    )
    env = KungFuDiscreteWrapper(env)
    env = FrameSkipWrapper(env, skip=4)
    env = KungFuRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=5000)
    if seed is not None:
        env.seed(seed)
    return env

def train(args):
    global global_model, global_model_path
    
    set_random_seed(args.seed)
    env = make_vec_env(
        lambda: make_env(args.render, args.seed),
        n_envs=args.num_envs,
        vec_env_cls=SubprocVecEnv if args.num_envs > 1 else DummyVecEnv,
        monitor_dir="./monitor_logs"
    )
    env = VecFrameStack(env, n_stack=4)
    env = VecMonitor(env)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = os.path.abspath(args.model_path)
    global_model_path = model_path
    
    if args.resume and os.path.exists(f"{model_path}.zip"):
        model = PPO.load(f"{model_path}.zip", env=env, device=device)
        print(f"Resumed from {model_path}.zip")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=4096,
            batch_size=256,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=0.15,  # Reduced from 0.2 to make the policy more aggressive/focused
            tensorboard_log=args.log_dir
        )
    
    global_model = model
    callbacks = [
        RewardLoggerCallback(log_freq=1000),
        ActionDistributionCallback(log_freq=5000)
    ]
    
    # Add progress bar callback if requested
    if args.progress_bar:
        callbacks.append(ProgressBarCallback(args.timesteps, args.num_envs))
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=not args.resume,
        progress_bar=False  # Set to True if you prefer the built-in progress bar
    )
    model.save(model_path)
    env.close()
    print(f"Model saved to {model_path}.zip")

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = f"{args.model_path}.zip"
    model = PPO.load(model_path, device=device)
    
    env = make_vec_env(lambda: make_env(args.render), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    raw_env = env.envs[0].env.env
    
    obs = env.reset()
    total_reward = 0
    episode_count = 0
    action_counts = [0] * 9
    action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
    
    while True:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        action_counts[action[0]] += 1
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        
        if args.render:
            raw_env.render()
            time.sleep(0.01)
        
        if done:
            episode_count += 1
            print(f"Episode {episode_count} - Score: {info[0].get('score', 0)}, Reward: {total_reward:.2f}")
            total_actions = sum(action_counts)
            print("\n===== Action Distribution =====")
            for i, (name, count) in enumerate(zip(action_names, action_counts)):
                print(f"{name}: {count/total_actions*100:.2f}% ({count})")
            print(f"Total Attack Actions: {sum(action_counts[3:])/total_actions*100:.2f}%")
            print(f"Successful Hits: {info[0].get('successful_hits', 0)}")
            total_reward = 0
            action_counts = [0] * 9
            obs = env.reset()
            if args.episodes > 0 and episode_count >= args.episodes:
                break
    env.close()

def evaluate(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = f"{args.model_path}.zip"
    model = PPO.load(model_path, device=device)
    
    env = make_vec_env(lambda: make_env(args.render), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    n_episodes = args.eval_episodes
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    action_counts = [0] * 9
    action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[action[0]] += 1
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            if args.render:
                env.envs[0].env.env.render()
                time.sleep(0.01)
            if done:
                episode_rewards.append(total_reward)
                episode_scores.append(info[0].get('score', 0))
                episode_lengths.append(step_count)
                print(f"Episode {episode+1}/{n_episodes} - Score: {episode_scores[-1]}, Reward: {total_reward:.2f}")
                break
    
    avg_reward = np.mean(episode_rewards)
    avg_score = np.mean(episode_scores)
    avg_length = np.mean(episode_lengths)
    total_actions = sum(action_counts)
    print("===== Evaluation Results =====")
    print(f"Avg Score: {avg_score:.2f}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
    print("\n===== Action Distribution =====")
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        print(f"{name}: {count/total_actions*100:.2f}% ({count})")
    print(f"Total Attack Actions: {sum(action_counts[3:])/total_actions*100:.2f}%")
    env.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Train or play KungFu Master using PPO")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--play", action="store_true")
    mode_group.add_argument("--evaluate", action="store_true")
    
    parser.add_argument("--model_path", default="models/kungfu_ppo")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timesteps", type=int, default=90_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--log_dir", default="logs")
    
    args = parser.parse_args()
    if not any([args.train, args.play, args.evaluate]):
        args.train = True
    
    if args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        evaluate(args)