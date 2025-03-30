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
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('training.log')]
)

global_model = None
global_model_path = None
terminate_flag = False

def signal_handler(sig, frame):
    global terminate_flag, global_model, global_model_path
    print(f"Signal {sig} received! Preparing to terminate...")
    if global_model is not None and global_model_path is not None:
        global_model.save(f"{global_model_path}/kungfu_ppo")
        print(f"Emergency save completed: {global_model_path}/kungfu_ppo.zip")
    terminate_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
        if isinstance(action, (list, np.ndarray)):
            action = int(action.item() if isinstance(action, np.ndarray) else action[0])
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

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
    def observation(self, obs):
        obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114])  # Convert to grayscale
        obs = np.array(Image.fromarray(obs).resize((84, 84), Image.BILINEAR))  # Resize
        obs = np.expand_dims(obs, axis=-1)  # Add channel dimension
        return obs.astype(np.uint8)

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        if not hasattr(self.env, 'get_ram'):
            raise ValueError("Environment must support get_ram() method")
        self.enemy_positions = {
            'player_x': 0x0050, 'player_y': 0x0053,
            'enemy1_x': 0x036E, 'enemy1_y': 0x0371,
            'enemy2_x': 0x0372, 'enemy2_y': 0x0375
        }
        self.enemy_proximity_threshold = 50
        self.reset_state()

    def reset_state(self):
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

    def reset(self):
        self.reset_state()
        return self.env.reset()

    def is_enemy_nearby(self, ram):
        try:
            player_x = ram[self.enemy_positions['player_x']]
            player_y = ram[self.enemy_positions['player_y']]
            for i in range(1, 3):
                enemy_x = ram[self.enemy_positions[f'enemy{i}_x']]
                enemy_y = ram[self.enemy_positions[f'enemy{i}_y']]
                if enemy_x > 0 and enemy_y > 0:
                    x_dist = abs(player_x - enemy_x)
                    y_dist = abs(player_y - enemy_y)
                    if x_dist < self.enemy_proximity_threshold and y_dist < 25:
                        return True
            return False
        except (IndexError, KeyError):
            logging.warning("Failed to read RAM for enemy proximity check")
            return False

    def detect_enemy_hit(self, ram, info):
        if info.get('score', 0) > self.last_score:
            self.successful_hits += 1
            return True
        for i in range(1, 3):
            enemy_x_addr = self.enemy_positions[f'enemy{i}_x']
            if (enemy_x_addr in self.enemy_health_previous and 
                self.enemy_health_previous[enemy_x_addr] > 0 and
                ram[enemy_x_addr] == 0):
                self.successful_hits += 1
                return True
        return False

    def step(self, action):
        if isinstance(action, (list, np.ndarray)):
            action = int(action.item() if isinstance(action, np.ndarray) else action[0])
            
        obs, reward, done, info = self.env.step(action)
        try:
            ram = self.env.get_ram()
        except Exception as e:
            logging.error(f"Failed to get RAM: {e}")
            return obs, reward, done, info

        current_score = info.get('score', 0)
        current_x = info.get('x_pos', 0)
        health = info.get('health', 46)
        enemy_nearby = self.is_enemy_nearby(ram)
        enemy_hit = self.detect_enemy_hit(ram, info)
        
        for i in range(1, 3):
            self.enemy_health_previous[self.enemy_positions[f'enemy{i}_x']] = ram[self.enemy_positions[f'enemy{i}_x']]
        
        if enemy_nearby:
            self.enemy_nearby_frames += 1
        else:
            self.enemy_nearby_frames = 0
        
        score_delta = current_score - self.last_score
        x_delta = current_x - self.last_x
        health_delta = health - self.last_health
        
        reward = 0
        if score_delta > 0:
            reward += score_delta * 200.0 + 150.0
        if x_delta > 0:
            reward += x_delta * 10.0
        if x_delta < 0:
            reward += x_delta * 2.0
        if health_delta < 0:
            reward += health_delta * 5.0
        reward -= 0.05
        
        if health <= 0 and not self.death_penalty_applied:
            reward -= 100
            self.death_penalty_applied = True
        
        if abs(x_delta) < 1:
            self.time_without_progress += 1
            if self.time_without_progress > 100:
                reward -= 5
        else:
            self.time_without_progress = 0
        
        if enemy_nearby:
            self.frames_since_last_attack += 1
            if self.frames_since_last_attack > 30 and action < 3:
                reward -= 1
        
        if action >= 3:
            self.frames_since_last_attack = 0
            if enemy_nearby:
                reward += 0.5
        
        if enemy_hit:
            reward += 20
            self.frames_since_last_enemy_hit = 0
        else:
            self.frames_since_last_enemy_hit += 1
            if self.frames_since_last_enemy_hit > 100 and enemy_nearby:
                reward -= 0.5
        
        self.last_score = current_score
        self.last_x = current_x
        self.last_health = health
        
        return obs, reward, done, info

def make_kungfu_env(render=False, seed=None):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    env = FrameSkipWrapper(env, skip=4)
    env = PreprocessFrame(env)
    env = KungFuRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=5000)
    if seed is not None:
        env.seed(seed)
    return env

def make_env(rank, seed=0, render=False):
    def _init():
        env = make_kungfu_env(render=(rank == 0 and render), seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.update_interval = max(self.total_timesteps // 1000, 1)
        self.n_calls = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.n_calls += 1
        if self.n_calls % self.update_interval == 0 or self.n_calls == self.total_timesteps:
            self.pbar.update(self.n_calls - self.pbar.n)
        return not terminate_flag

    def _on_training_end(self):
        self.pbar.n = self.total_timesteps
        self.pbar.close()

def train(args):
    global global_model, global_model_path
    global_model_path = args.model_path
    
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    set_random_seed(args.seed)
    
    # Create vectorized environment with new shape
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed, args.render) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.render)])
    
    env = VecFrameStack(env, n_stack=4)  # New shape: (4, 84, 84)
    env = VecMonitor(env, os.path.join(args.log_dir, 'monitor.csv'))
    
    # Verify observation space
    expected_obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    print(f"Current environment observation space: {env.observation_space}")
    if env.observation_space != expected_obs_space:
        print(f"Warning: Observation space mismatch. Expected {expected_obs_space}, got {env.observation_space}")

    callbacks = []
    if args.progress_bar:
        callbacks.append(TqdmCallback(total_timesteps=args.timesteps))
    
    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if args.resume and os.path.exists(model_file):
        print(f"Resuming training from {model_file}")
        try:
            # Load the model without immediately passing the env to bypass the check
            loaded_model = PPO.load(model_file, device=device, print_system_info=True)
            old_obs_space = loaded_model.env.observation_space
            print(f"Saved model observation space: {old_obs_space}")
            
            # Check if observation spaces differ
            if old_obs_space != env.observation_space:
                print("Observation space mismatch detected. Adapting model to new observation space...")
                
                # Create a new PPO model with the new environment
                model = PPO(
                    "CnnPolicy",
                    env,
                    learning_rate=args.learning_rate,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=args.clip_range,
                    tensorboard_log=args.log_dir,
                    verbose=1,
                    device=device
                )
                
                # Transfer weights from the old model to the new one where possible
                old_policy_state_dict = loaded_model.policy.state_dict()
                new_policy_state_dict = model.policy.state_dict()
                
                # Copy weights for layers that match in shape
                for key in old_policy_state_dict:
                    if key in new_policy_state_dict and old_policy_state_dict[key].shape == new_policy_state_dict[key].shape:
                        new_policy_state_dict[key] = old_policy_state_dict[key]
                    else:
                        print(f"Skipping weight transfer for {key} due to shape mismatch.")
                
                # Load the adapted weights into the new model
                model.policy.load_state_dict(new_policy_state_dict)
                print("Weights transferred successfully where compatible.")
            else:
                # If observation spaces match, load normally
                model = PPO.load(
                    model_file,
                    env=env,
                    custom_objects={
                        "learning_rate": args.learning_rate,
                        "n_steps": 2048,
                        "batch_size": 64,
                        "n_epochs": args.n_epochs,
                        "gamma": args.gamma,
                        "gae_lambda": args.gae_lambda,
                        "clip_range": args.clip_range,
                        "tensorboard_log": args.log_dir
                    },
                    device=device
                )
        except Exception as e:
            print(f"Error during model loading/adaptation: {e}")
            print("Starting new training instead.")
            model = PPO(
                "CnnPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=args.clip_range,
                tensorboard_log=args.log_dir,
                verbose=1,
                device=device
            )
    else:
        print("Starting new training.")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            tensorboard_log=args.log_dir,
            verbose=1,
            device=device
        )
    
    global_model = model
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=False if args.resume and os.path.exists(model_file) else True
        )
        model.save(model_file)
        print(f"Training completed. Model saved to {model_file}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        logging.error(f"Error during training: {str(e)}")
        model.save(model_file)
        print(f"Model saved to {model_file} due to error")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Error closing environment: {str(e)}")
        if terminate_flag:
            print("Training terminated by signal.")

def play(args):
    env = make_kungfu_env(render=args.render)
    env = DummyVecEnv([lambda: env])  # Wrap in DummyVecEnv for consistency
    env = VecFrameStack(env, n_stack=4)  # Match training setup
    
    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if not os.path.exists(model_file):
        print(f"No model found at {model_file}. Please train a model first.")
        return
    
    print(f"Loading model from {model_file}")
    model = PPO.load(model_file, env=env)
    
    episode_count = 0
    try:
        while not terminate_flag:
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_count += 1
            print(f"Starting episode {episode_count}")
            
            while not done and not terminate_flag:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if args.render:
                    env.render()
                    time.sleep(0.01)
                
                if terminate_flag:
                    break
            
            print(f"Episode {episode_count} - Total reward: {total_reward}, Steps: {steps}")
            if args.episodes > 0 and episode_count >= args.episodes:
                break
    
    except Exception as e:
        print(f"Error during play: {str(e)}")
        logging.error(f"Error during play: {str(e)}")
    finally:
        env.close()

def evaluate(args):
    pass  # Placeholder for evaluate function

if __name__ == "__main__":
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
    parser.add_argument("--timesteps", type=int, default=10_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--progress_bar", action="store_true")
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