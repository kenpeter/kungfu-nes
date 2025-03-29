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

# Configure logging with minimal output
logging.basicConfig(
    level=logging.WARNING,  # Changed from INFO to WARNING to reduce verbosity
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log')]
)

# Global variables for the signal handler
global_model = None
global_model_path = None

class KungFuDiscreteWrapper(gym.ActionWrapper):
    """
    Converts the continuous action space to a discrete set of actions
    suitable for Kung Fu.
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(9)  # 9 discrete actions
        # Map discrete actions to controller buttons
        # Format: [B, A, Select, Start, Up, Down, Left, Right, ...]
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
    """
    Repeats the same action for `skip` frames and returns
    the maximum of the last two observations.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        
        # Repeat action for skip frames
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, info

class KungFuRewardWrapper(Wrapper):
    """
    Enhanced reward function for Kung Fu, focusing on score, forward progress,
    health preservation, and encouraging attacking when enemies are nearby.
    """
    def __init__(self, env):
        super().__init__(env)
        self.reset()
        # Define memory regions for enemy detection
        # These values need to be adjusted based on the game's RAM layout
        self.enemy_positions = {
            'enemy1_x': 0x036E,  # Example address - needs to be verified
            'enemy1_y': 0x0371,  # Example address - needs to be verified
            'enemy2_x': 0x0372,  # Example address - needs to be verified
            'enemy2_y': 0x0375,  # Example address - needs to be verified
            'player_x': 0x0050,  # Example address - needs to be verified
            'player_y': 0x0053,  # Example address - needs to be verified
        }
        self.enemy_proximity_threshold = 30  # Pixel distance threshold to consider an enemy "nearby"
        self.no_attack_penalty_cooldown = 0

    def reset(self):
        self.last_score = 0
        self.last_x = 0
        self.last_health = 46  # Initial health
        self.time_without_progress = 0
        self.last_action_time = time.time()
        self.death_penalty_applied = False
        self.last_enemy_count = 0
        self.last_enemy_health = 0
        self.frames_since_last_attack = 0
        self.enemy_nearby_frames = 0
        return self.env.reset()

    def _is_enemy_nearby(self, ram):
        """
        Check if any enemy is within attacking distance of the player.
        Uses RAM values to determine positions.
        """
        # Get player position
        try:
            player_x = ram[self.enemy_positions['player_x']]
            player_y = ram[self.enemy_positions['player_y']]
            
            # Check each potential enemy
            for i in range(1, 3):  # Check first two potential enemies
                enemy_x = ram[self.enemy_positions[f'enemy{i}_x']]
                enemy_y = ram[self.enemy_positions[f'enemy{i}_y']]
                
                # Calculate distance
                x_distance = abs(player_x - enemy_x)
                y_distance = abs(player_y - enemy_y)
                
                # If enemy is within attacking range
                if x_distance < self.enemy_proximity_threshold and y_distance < 10:
                    return True
            
            # Alternative method: Check known memory locations that indicate enemy presence
            # This is a fallback mechanism if the position-based check doesn't work
            enemies_present = False
            for addr in range(0x0390, 0x03A0):  # Example range to check enemy presence flags
                if ram[addr] != 0:
                    enemies_present = True
                    break
                    
            return enemies_present
        except:
            # Fallback mechanism if RAM access fails
            # Use a time-based heuristic: if the agent hasn't attacked in a while,
            # assume enemies are nearby to encourage periodic attacking
            self.frames_since_last_attack += 1
            if self.frames_since_last_attack > 60:  # Every ~1 second (assuming 60 fps)
                self.frames_since_last_attack = 0
                return True
            return False

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # Get RAM state for enemy detection (if available)
        ram = self.env.get_ram() if hasattr(self.env, 'get_ram') else None
        
        # Extract game state information
        current_score = info.get('score', 0)
        current_x = info.get('x_pos', 0)
        health = info.get('health', 46)
        
        # Check if enemies are nearby using RAM state
        enemy_nearby = False
        if ram is not None:
            enemy_nearby = self._is_enemy_nearby(ram)
        else:
            # Fallback: use a heuristic - encourage attacks every ~2 seconds
            self.frames_since_last_attack += 1
            if self.frames_since_last_attack > 120:
                enemy_nearby = True
                # Reset after encouraging an attack
                if action in [3, 4, 5, 6, 7, 8]:
                    self.frames_since_last_attack = 0
        
        # Update enemy nearby tracking
        if enemy_nearby:
            self.enemy_nearby_frames += 1
        else:
            self.enemy_nearby_frames = 0
        
        # Calculate changes
        score_delta = current_score - self.last_score
        x_delta = current_x - self.last_x
        health_delta = health - self.last_health
        
        # Initialize reward
        reward = 0
        
        # Increased reward for increasing score (defeating enemies)
        if score_delta > 0:
            reward += score_delta * 100.0  # Increased reward for score increase
            reward += 50.0  # Additional flat reward for defeating an enemy
        
        # Reward for forward progress (moving right in this side-scroller)
        if x_delta > 0:
            reward += x_delta * 15.0
        
        # Small penalty for moving backward
        if x_delta < 0:
            reward += x_delta * 1.0
        
        # Reduced penalty for losing health to encourage combat
        if health_delta < 0:
            reward += health_delta * 10.0
        
        # Time penalty to encourage faster completion
        reward -= 0.1
        
        # Large penalty if health reaches zero (death)
        if health <= 0 and not self.death_penalty_applied:
            reward -= 100
            self.death_penalty_applied = True
        
        # Detect if agent is stuck
        if abs(x_delta) < 1:
            self.time_without_progress += 1
            if self.time_without_progress > 100:
                reward -= 1.0  # Penalty for being stuck
                if self.time_without_progress > 200:
                    reward -= 2.0  # Increased penalty for being stuck for a long time
        else:
            self.time_without_progress = 0
        
        # Strongly encourage attacking when enemies are nearby
        if enemy_nearby:
            if action in [3, 4, 5, 6, 7, 8]:  # Attack actions
                reward += 15.0  # Significantly increased reward for appropriate attacks
                self.no_attack_penalty_cooldown = 5  # Set cooldown to prevent immediate penalties
            elif self.no_attack_penalty_cooldown <= 0 and self.enemy_nearby_frames > 10:
                # Penalize not attacking when enemies have been nearby for a while
                reward -= 5.0
                self.enemy_nearby_frames = 0  # Reset counter after applying penalty
        else:
            # Standard reward for any action when no enemies nearby
            reward += 0.5
            
        # Reduce cooldown counter
        if self.no_attack_penalty_cooldown > 0:
            self.no_attack_penalty_cooldown -= 1
        
        # Update state tracking variables
        self.last_score = current_score
        self.last_x = current_x
        self.last_health = health
        
        # Mark episode as done if health reaches zero
        done = done or health <= 0
        return obs, reward, done, info

class ProgressBarCallback(BaseCallback):
    """
    Displays a progress bar during training.
    """
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training")
    
    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

class RewardLoggerCallback(BaseCallback):
    """
    Logs reward and episode statistics during training.
    """
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = [0] * 9  # Track action distribution
    
    def _on_step(self):
        # Track action distribution
        if hasattr(self.model, 'actions'):
            for action in self.model.actions:
                self.action_counts[action] += 1
        
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                if 'r' in info and 'l' in info:
                    self.episode_rewards.append(info['r'])
                    self.episode_lengths.append(info['l'])
        
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else np.mean(self.episode_rewards)
            avg_length = np.mean(self.episode_lengths[-10:]) if len(self.episode_lengths) >= 10 else np.mean(self.episode_lengths)
            print(f"Timestep: {self.num_timesteps}, Avg Reward (last 10): {avg_reward:.2f}, Avg Episode Length: {avg_length:.2f}")
            
            # Print action distribution every 5000 steps
            if self.n_calls % 5000 == 0 and sum(self.action_counts) > 0:
                total_actions = sum(self.action_counts)
                attack_percentage = sum(self.action_counts[3:9]) / total_actions * 100
                print(f"Action distribution: {[count/total_actions*100 for count in self.action_counts]}")
                print(f"Attack actions: {attack_percentage:.2f}%")
        
        return True

class ActionDistributionCallback(BaseCallback):
    """
    Tracks and logs the distribution of actions taken by the agent.
    """
    def __init__(self, log_freq=5000):
        super().__init__()
        self.log_freq = log_freq
        self.actions_taken = []
    
    def _on_step(self):
        # Store actions taken by the agent
        if hasattr(self, 'locals') and 'actions' in self.locals:
            self.actions_taken.extend(self.locals['actions'])
        
        if self.n_calls % self.log_freq == 0 and len(self.actions_taken) > 0:
            # Count action frequencies
            action_counts = np.bincount(self.actions_taken, minlength=9)
            total_actions = len(self.actions_taken)
            
            # Calculate percentages
            action_percentages = (action_counts / total_actions) * 100
            
            # Log the distribution
            print("\n===== Action Distribution =====")
            action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
            for i, (name, percentage) in enumerate(zip(action_names, action_percentages)):
                print(f"{name}: {percentage:.2f}%")
            
            attack_percentage = sum(action_percentages[3:]) 
            print(f"Total Attack Actions: {attack_percentage:.2f}%")
            print("==============================\n")
            
            # Reset action tracking to avoid memory issues
            self.actions_taken = []
        
        return True

def signal_handler(sig, frame):
    """
    Handle Ctrl+C by saving the model before exiting.
    """
    if global_model is not None and global_model_path is not None:
        print("Interrupt received! Saving model...")
        emergency_path = f"{global_model_path}_emergency_{int(time.time())}"
        global_model.save(emergency_path)
        print(f"Emergency save completed: {emergency_path}")
    sys.exit(0)

def make_env(render=False):
    """
    Create and configure the Kung Fu environment with all necessary wrappers.
    """
    try:
        env = retro.make(
            game="KungFu-Nes",
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.IMAGE
        )
        env = KungFuDiscreteWrapper(env)
        env = FrameSkipWrapper(env, skip=4)
        env = KungFuRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=5000)
        return env
    except Exception as e:
        print(f"Failed to create environment: {e}")
        raise

def train(args):
    """
    Train the agent using PPO algorithm.
    """
    global global_model, global_model_path
    
    print("Initializing training...")
    
    try:
        env = make_vec_env(
            lambda: make_env(args.render),
            n_envs=args.num_envs,
            vec_env_cls=SubprocVecEnv if args.num_envs > 1 else DummyVecEnv,
            monitor_dir="./monitor_logs",
            seed=args.seed
        )
        env = VecFrameStack(env, n_stack=4)
        env = VecMonitor(env)
        print("Environment created successfully")
    except Exception as e:
        print(f"Environment creation failed: {e}")
        raise

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' and torch.cuda.is_available() else 'CPU'})")
    
    model_path = os.path.abspath(args.model_path)
    global_model_path = model_path
    
    if args.resume and os.path.exists(f"{model_path}.zip"):
        model = PPO.load(f"{model_path}.zip", env=env, device=device, 
                         custom_objects={"learning_rate": args.learning_rate})
        print(f"Resumed training from {model_path}.zip")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=0,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,  # Increased entropy coefficient for more exploration
            tensorboard_log=args.log_dir
        )
        print("Created new model")
    
    global_model = model

    # Set up callbacks
    callbacks = []
    if args.progress_bar:
        callbacks.append(ProgressBarCallback(args.timesteps))
    callbacks.append(RewardLoggerCallback(log_freq=1000))
    callbacks.append(ActionDistributionCallback(log_freq=5000))

    try:
        print(f"Starting training for {args.timesteps} timesteps")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=os.path.basename(model_path),
            reset_num_timesteps=not args.resume,
            progress_bar=False
        )
        print("Training completed successfully")
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    finally:
        print("Saving final model...")
        model.save(model_path)
        env.close()
        print(f"Final model saved to {model_path}.zip")

def play(args):
    """
    Use a trained model to play the game.
    """
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = f"{args.model_path}.zip" if not args.model_path.endswith(".zip") else args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")

    env = make_vec_env(lambda: make_env(render=args.render), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    raw_env = env.envs[0].env.env if args.render else None

    obs = env.reset()
    total_reward = 0
    episode_count = 0
    
    # For action tracking
    action_counts = [0] * 9
    action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            
            # Track action distribution
            action_counts[action[0]] += 1
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            
            if args.render and raw_env:
                raw_env.render()
                time.sleep(0.01)
                
                # Periodically print current action
                if sum(action_counts) % 100 == 0:
                    print(f"Current action: {action_names[action[0]]}")
            
            if done:
                episode_count += 1
                print(f"Episode {episode_count} finished. Score: {info[0].get('score', 0)}, Total Reward: {total_reward:.2f}")
                
                # Print action distribution at the end of episode
                total_actions = sum(action_counts)
                print("\n===== Action Distribution =====")
                for i, (name, count) in enumerate(zip(action_names, action_counts)):
                    print(f"{name}: {count/total_actions*100:.2f}% ({count} times)")
                
                attack_count = sum(action_counts[3:])
                print(f"Total Attack Actions: {attack_count/total_actions*100:.2f}% ({attack_count} times)")
                print("==============================\n")
                
                total_reward = 0
                action_counts = [0] * 9  # Reset action tracking for next episode
                obs = env.reset()
                
                # Optional stop after a certain number of episodes
                if args.episodes > 0 and episode_count >= args.episodes:
                    break
    except KeyboardInterrupt:
        print("Play session ended by user")
    finally:
        env.close()
        print("Play environment closed")

def evaluate(args):
    """
    Evaluate a trained model over multiple episodes.
    """
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = f"{args.model_path}.zip" if not args.model_path.endswith(".zip") else args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    print(f"Loaded model from {model_path}")

    env = make_vec_env(lambda: make_env(render=args.render), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    
    # Run evaluation for specified number of episodes
    n_episodes = args.eval_episodes if hasattr(args, 'eval_episodes') else 10
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    # For action tracking
    action_counts = [0] * 9
    
    for episode in range(n_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[action[0]] += 1
            
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            step_count += 1
            
            if args.render:
                env.envs[0].env.env.render()
                time.sleep(0.01)
            
            if done[0]:
                episode_score = info[0].get('score', 0)
                episode_rewards.append(total_reward)
                episode_scores.append(episode_score)
                episode_lengths.append(step_count)
                print(f"Episode {episode+1}/{n_episodes} - Score: {episode_score}, Reward: {total_reward:.2f}, Length: {step_count}")
                break
    
    # Print evaluation summary
    avg_reward = np.mean(episode_rewards)
    avg_score = np.mean(episode_scores)
    avg_length = np.mean(episode_lengths)
    
    print("===== Evaluation Results =====")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Average Score: {avg_score:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Episode Length: {avg_length:.2f}")
    
    # Print action distribution
    total_actions = sum(action_counts)
    print("\n===== Action Distribution =====")
    action_names = ["None", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right"]
    for i, (name, count) in enumerate(zip(action_names, action_counts)):
        print(f"{name}: {count/total_actions*100:.2f}% ({count} times)")
    
    attack_count = sum(action_counts[3:])
    print(f"Total Attack Actions: {attack_count/total_actions*100:.2f}% ({attack_count} times)")
    print("============================")
    
    env.close()

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Train or play KungFu Master using PPO")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Run in training mode")
    mode_group.add_argument("--play", action="store_true", help="Run in play mode")
    mode_group.add_argument("--evaluate", action="store_true", help="Run in evaluation mode")
    
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    
    # Training parameters
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Number of training timesteps")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    
    # Evaluation and play parameters
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to play (0 for infinite)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions in play mode")
    
    # PPO hyperparameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--n_epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.1)  # Increased from 0.05 to encourage exploration
    
    parser.add_argument("--log_dir", default="logs", help="Directory for TensorBoard logs")
    
    args = parser.parse_args()
    
    # Set default mode to train if none specified
    if not any([args.train, args.play, args.evaluate]):
        args.train = True
    
    try:
        if args.play:
            play(args)
        elif args.evaluate:
            evaluate(args)
        else:
            train(args)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)