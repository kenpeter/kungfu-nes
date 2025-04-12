import argparse
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
import logging
import sys
import signal
import time
import glob
from kungfu_env import make_env, SimpleCNN, KUNGFU_MAX_ENEMIES  # Import KUNGFU_MAX_ENEMIES

# Global variables
current_model = None
global_logger = None
global_model_path = None
experience_data = []

def emergency_save_handler(signum, frame):
    global current_model, global_logger, global_model_path, experience_data
    if current_model is not None and global_model_path is not None:
        current_model.save(global_model_path)
        experience_count = len(experience_data)
        if global_logger:
            global_logger.info(f"Emergency save triggered. Model saved at {global_model_path}")
            global_logger.info(f"Collected experience: {experience_count} steps")
        else:
            print(f"Emergency save triggered. Model saved at {global_model_path}")
            print(f"Collected experience: {experience_count} steps")
        
        with open(f"{global_model_path}_experience_count.txt", "w") as f:
            f.write(f"Total experience collected: {experience_count} steps\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if hasattr(current_model, 'env'):
            current_model.env.close()
        if args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del current_model
        current_model = None
    sys.exit(0)

signal.signal(signal.SIGINT, emergency_save_handler)

class ExperienceCollectionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.experience_data = []
        
    def _on_step(self) -> bool:
        obs = self.locals.get('new_obs')
        actions = self.locals.get('actions')
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        infos = self.locals.get('infos')
        
        if all(x is not None for x in [obs, actions, rewards, dones, infos]):
            experience = {
                "observation": obs,
                "action": actions,
                "reward": rewards,
                "done": dones,
                "info": infos
            }
            self.experience_data.append(experience)
        return True

class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_score = 0

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        
        # Aggregate metrics from infos
        total_hits = sum([info.get('enemy_hit', 0) for info in infos])
        total_hp = sum([info.get('hp', 0) for info in infos])
        total_dodge_reward = sum([info.get('dodge_reward', 0) for info in infos])
        total_distance_reward = sum([info.get('distance_reward', 0) for info in infos])
        total_survival_reward = sum([info.get('survival_reward_total', 0) for info in infos])
        avg_normalized_reward = sum([info.get('normalized_reward', 0) for info in infos]) / max(1, len(infos))
        avg_min_enemy_dist = sum([info.get('min_enemy_dist', 255) for info in infos]) / max(1, len(infos))
        
        # Compute action diversity score to encourage exploration
        action_diversity = 0
        if infos and 'action_percentages' in infos[0]:
            action_percentages = infos[0].get('action_percentages', [])
            if len(action_percentages) > 1:
                # Negative entropy to reward balanced action usage
                action_diversity = -sum(p * np.log(p + 1e-6) for p in action_percentages if p > 0)
                action_diversity = action_diversity / np.log(len(action_percentages))  # Normalize by max entropy
        
        # Compute progress-focused score
        close_combat_bonus = 0
        if avg_min_enemy_dist <= 30:  # Align with close_range_threshold in KungFuWrapper
            close_combat_bonus = 10.0  # Bonus for staying in effective combat range
        
        score = (
            total_hits * 10 +                   # Reward for defeating enemies
            total_hp / 255.0 * 20 +             # Strong incentive for survival
            total_dodge_reward * 15 +           # Heavily reward dodging projectiles
            total_distance_reward * 8 +         # Encourage closing the gap when far
            total_survival_reward * 12 +        # Emphasize staying alive longer
            avg_normalized_reward * 200 +       # Prioritize consistent rewards
            action_diversity * 25 +             # Promote diverse actions
            close_combat_bonus * (1 + total_hits)  # Amplify combat effectiveness when close
        )
        
        # Update best score and save model if improved
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Saved best model with score {self.best_score:.2f} at step {self.num_timesteps}")
                print(f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Dodge: {total_dodge_reward:.2f}, "
                    f"Distance Reward: {total_distance_reward:.2f}, Survival: {total_survival_reward:.2f}, "
                    f"Norm. Reward: {avg_normalized_reward:.2f}, Action Diversity: {action_diversity:.2f}, "
                    f"Min Enemy Dist: {avg_min_enemy_dist:.1f}")
                
                # Log action percentages to monitor exploration
                if infos and 'action_percentages' in infos[0] and 'action_names' in infos[0]:
                    action_percentages = infos[0].get('action_percentages', [])
                    action_names = infos[0].get('action_names', [])
                    if len(action_percentages) == len(action_names):
                        print("  Action Percentages:")
                        for name, perc in zip(action_names, action_percentages):
                            print(f"    {name}: {perc * 100:.1f}%")
        
        # Periodic logging to track progress
        if self.num_timesteps % 5000 == 0 and self.verbose > 0:  # Reduced frequency to align with game pace
            print(f"Step {self.num_timesteps} Progress:")
            print(f"  Current Score: {score:.2f}, Best Score: {self.best_score:.2f}")
            print(f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Norm. Reward: {avg_normalized_reward:.2f}, "
                f"Min Enemy Dist: {avg_min_enemy_dist:.1f}, Survival: {total_survival_reward:.2f}")
        
        return True

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger()

class NPZReplayEnvironment:
    """Custom environment for training from NPZ files"""
    def __init__(self, npz_directory):
        self.npz_files = glob.glob(os.path.join(npz_directory, '*.npz'))
        if not self.npz_files:
            raise ValueError(f"No NPZ files found in directory: {npz_directory}")
        
        self.current_file_idx = 0
        self.current_frame_idx = 0
        self.load_current_npz()
        
        self.observation_space = None
        self.action_space = None
        self.num_envs = 1
        
        self._setup_spaces()
    
    def load_current_npz(self):
        """Load the current NPZ file data"""
        npz_data = np.load(self.npz_files[self.current_file_idx])
        self.frames = npz_data['frames']
        self.actions = npz_data['actions']
        self.rewards = npz_data['rewards']
        self.current_frame_idx = 0
    
    def _setup_spaces(self):
        # Import spaces within the method to ensure availability in subprocesses
        from gym import spaces
        import numpy as np  # Also ensure numpy is available if not already imported

        first_frame = self.frames[0]
        
        # Define constants
        KUNGFU_MAX_ENEMIES = 5
        
        # Define shapes for observation keys
        enemy_vector_shape = (KUNGFU_MAX_ENEMIES * 3,)  # e.g., 15
        combat_status_shape = (1,)
        projectile_vectors_shape = (10,)
        closest_enemy_direction_shape = (2,)  # Assuming a 2D direction vector
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'viewport': spaces.Box(low=0, high=255, shape=first_frame.shape, dtype=np.uint8),
            'enemy_vector': spaces.Box(low=0, high=255, shape=enemy_vector_shape, dtype=np.float32),
            'combat_status': spaces.Box(low=0, high=1, shape=combat_status_shape, dtype=np.float32),
            'projectile_vectors': spaces.Box(low=0, high=255, shape=projectile_vectors_shape, dtype=np.float32),
            'enemy_proximity': spaces.Box(low=0, high=1, shape=(KUNGFU_MAX_ENEMIES,), dtype=np.float32),
            'boss_info': spaces.Box(low=0, high=255, shape=(3,), dtype=np.float32),
            'closest_enemy_direction': spaces.Box(low=-1, high=1, shape=closest_enemy_direction_shape, dtype=np.float32),
        })
        
        self.action_space = spaces.MultiBinary(len(self.actions[0]))

    def reset(self):
        """Reset to beginning of current NPZ file or load a new one"""
        self.current_file_idx = np.random.randint(0, len(self.npz_files))
        self.load_current_npz()
        
        # Define shapes for consistency
        KUNGFU_MAX_ENEMIES = 5
        enemy_vector_shape = (KUNGFU_MAX_ENEMIES * 3,)
        combat_status_shape = (1,)
        projectile_vectors_shape = (10,)
        closest_enemy_direction_shape = (2,)  # Assuming a 2D direction vector
        
        # Return observation with placeholders
        return {
            'viewport': self.frames[0],
            'enemy_vector': np.zeros(enemy_vector_shape, dtype=np.float32),
            'combat_status': np.zeros(combat_status_shape, dtype=np.float32),
            'projectile_vectors': np.zeros(projectile_vectors_shape, dtype=np.float32),
            'enemy_proximity': np.zeros(KUNGFU_MAX_ENEMIES, dtype=np.float32),
            'boss_info': np.zeros(3, dtype=np.float32),  # placeholder for boss info
            'closest_enemy_direction': np.zeros(closest_enemy_direction_shape, dtype=np.float32),
        }
    
    def step(self, action):
        """Step through the pre-recorded data"""
        frame = self.frames[self.current_frame_idx]
        reward = self.rewards[self.current_frame_idx]
        
        self.current_frame_idx += 1
        done = self.current_frame_idx >= len(self.frames) - 1
        if done:
            self.current_frame_idx = len(self.frames) - 1
        
        # Define shapes for consistency
        KUNGFU_MAX_ENEMIES = 5
        enemy_vector_shape = (KUNGFU_MAX_ENEMIES * 3,)
        combat_status_shape = (1,)
        projectile_vectors_shape = (10,)
        closest_enemy_direction_shape = (2,)
        
        # Return observation with placeholders
        obs = {
            'viewport': frame,
            'enemy_vector': np.zeros(enemy_vector_shape, dtype=np.float32),
            'combat_status': np.zeros(combat_status_shape, dtype=np.float32),
            'projectile_vectors': np.zeros(projectile_vectors_shape, dtype=np.float32),
            'enemy_proximity': np.zeros(KUNGFU_MAX_ENEMIES, dtype=np.float32),
            'boss_info': np.zeros(3, dtype=np.float32),  # placeholder for boss info,
            'closest_enemy_direction': np.zeros(closest_enemy_direction_shape, dtype=np.float32),
        }
        
        return obs, reward, done, {"mimic_training": True}
    
    def close(self):
        """Clean up resources"""
        pass

def create_npz_replay_env():
    """Factory function for NPZ replay environment"""
    env = NPZReplayEnvironment(args.npz_dir)
    return env

def train(args):
    global current_model, global_logger, global_model_path, experience_data
    experience_data = []
    
    global_logger = setup_logging(args.log_dir)
    global_logger.info(f"Starting training with {args.num_envs} envs and {args.timesteps} timesteps")
    global_logger.info(f"Maximum number of enemies: {KUNGFU_MAX_ENEMIES}")
    global_model_path = args.model_path
    current_model = None

    if args.num_envs < 1:
        raise ValueError("Number of environments must be at least 1")
    
    # Determine training mode based on npz_dir argument
    training_mode = "mimic" if args.npz_dir else "live"
    global_logger.info(f"Training mode: {training_mode}")
    
    # Create environments
    if training_mode == "mimic":
        global_logger.info(f"Loading NPZ data from: {args.npz_dir}")
        env_fns = [create_npz_replay_env for _ in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
    else:
        env_fns = [make_env for _ in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)
    
    # Frame stacking without automatic transposition
    env = VecFrameStack(env, n_stack=4, channels_order='last')

    # Disable automatic image transposition
    from stable_baselines3.common.vec_env import VecTransposeImage
    env = VecTransposeImage(env, skip=True)  # Skip the transposition
    
    # Policy kwargs using imported SimpleCNN
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 512, "n_stack": 4},
        "net_arch": dict(pi=[256, 256, 128], vf=[512, 512, 256])
    }
    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)

    # Training parameters
    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": 0.1,
        "n_steps": min(2048, args.timesteps // args.num_envs if args.num_envs > 0 else args.timesteps),
        "batch_size": 64,
        "n_epochs": 10
    }

    # Load or initialize model
    if args.resume and os.path.exists(args.model_path + ".zip"):
        global_logger.info(f"Resuming training from {args.model_path}")
        old_model = PPO.load(args.model_path, device="cuda" if args.cuda else "cpu")
        model = PPO(
            "MultiInputPolicy",
            env,
            **params,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )
        model.policy.load_state_dict(old_model.policy.state_dict())
        current_model = model
    else:
        global_logger.info("Starting new training session")
        model = PPO(
            "MultiInputPolicy",
            env,
            **params,
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )
        current_model = model

    # Set up callbacks
    save_callback = SaveBestModelCallback(save_path=args.model_path)
    exp_callback = ExperienceCollectionCallback()
    callback = CallbackList([save_callback, exp_callback])
    
    # Train the model
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)
    experience_data = exp_callback.experience_data
    
    experience_count = len(experience_data)
    global_logger.info(f"Training completed. Total experience collected: {experience_count} steps")
    
    with open(f"{args.model_path}_experience_count.txt", "w") as f:
        f.write(f"Total experience collected: {experience_count} steps\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training mode: {training_mode}\n")
        f.write(f"Training parameters: num_envs={args.num_envs}, timesteps={args.timesteps}\n")
        if training_mode == "mimic":
            f.write(f"NPZ directory: {args.npz_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO model for Kung Fu")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo_best", help="Path to save the trained model")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total timesteps for training")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Default clip range for PPO")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--npz_dir", help="Directory containing NPZ recordings for mimic training")
    
    args = parser.parse_args()
    train(args)