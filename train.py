import argparse
import os
import numpy as np
import torch
import zipfile
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecTransposeImage
import logging
import sys
import signal
import time
import glob
from kungfu_env import KungFuWrapper, SimpleCNN, KUNGFU_MAX_ENEMIES, MAX_PROJECTILES
import threading
import queue
import gymnasium as gym
from gymnasium import spaces
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Trajectory
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
import pickle

current_model = None
global_logger = None
global_model_path = None
experience_data = []

def emergency_save_handler(signum, frame):
    global current_model, global_logger, global_model_path, experience_data
    if current_model is not None and global_model_path is not None:
        try:
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
        except Exception as e:
            if global_logger:
                global_logger.error(f"Emergency save failed: {e}")
            else:
                print(f"Emergency save failed: {e}")
        
        if hasattr(current_model, 'env'):
            try:
                current_model.env.close()
            except Exception as e:
                if global_logger:
                    global_logger.warning(f"Failed to close environment: {e}")
                else:
                    print(f"Failed to close environment: {e}")
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
        
        total_hits = sum([info.get('enemy_hit', 0) for info in infos])
        total_hp = sum([info.get('hp', 0) for info in infos])
        total_dodge_reward = sum([info.get('dodge_reward', 0) for info in infos])
        total_survival_reward = sum([info.get('survival_reward_total', 0) for info in infos])
        avg_normalized_reward = sum([info.get('normalized_reward', 0) for info in infos]) / max(1, len(infos))
        avg_min_enemy_dist = sum([info.get('min_enemy_dist', 255) for info in infos]) / max(1, len(infos))
        
        action_diversity = 0
        if infos and 'action_percentages' in infos[0]:
            action_percentages = infos[0].get('action_percentages', [])
            if len(action_percentages) > 1:
                action_diversity = -sum(p * np.log(p + 1e-6) for p in action_percentages if p > 0)
                action_diversity = action_diversity / np.log(len(action_percentages))
        
        close_combat_bonus = 0
        if avg_min_enemy_dist <= 30:
            close_combat_bonus = 10.0
        
        score = (
            total_hits * 10 +
            total_hp / 255.0 * 20 +
            total_dodge_reward * 15 +
            total_survival_reward * 12 +
            avg_normalized_reward * 200 +
            action_diversity * 25 +
            close_combat_bonus * (1 + total_hits)
        )
        
        if score > self.best_score:
            self.best_score = score
            try:
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(f"Saved best model with score {self.best_score:.2f} at step {self.num_timesteps}")
                    print(f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Dodge: {total_dodge_reward:.2f}, "
                        f"Survival: {total_survival_reward:.2f}, Norm. Reward: {avg_normalized_reward:.2f}, "
                        f"Action Diversity: {action_diversity:.2f}, Min Enemy Dist: {avg_min_enemy_dist:.1f}")
                    
                    if infos and 'action_percentages' in infos[0] and 'action_names' in infos[0]:
                        action_percentages = infos[0].get('action_percentages', [])
                        action_names = infos[0].get('action_names', [])
                        if len(action_percentages) == len(action_names):
                            print("  Action Percentages:")
                            for name, perc in zip(action_names, action_percentages):
                                print(f"    {name}: {perc * 100:.1f}%")
            except Exception as e:
                print(f"Failed to save model: {e}")
        
        if self.num_timesteps % 5000 == 0 and self.verbose > 0:
            print(f"Step {self.num_timesteps} Progress:")
            print(f"  Current Score: {score:.2f}, Best Score: {self.best_score:.2f}")
            print(f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Norm. Reward: {avg_normalized_reward:.2f}, "
                f"Min Enemy Dist: {avg_min_enemy_dist:.1f}, Survival: {total_survival_reward:.2f}")
        
        return True

class RenderCallback(BaseCallback):
    def __init__(self, render_freq=256, fps=30, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.fps = fps
        self.step_count = 0
        self.render_queue = queue.Queue(maxsize=1)
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.last_render_time = time.time()
        self.render_env = None
        self.render_thread.start()
    
    def _render_loop(self):
        while True:
            try:
                if not self.render_queue.empty():
                    self.render_env = self.render_queue.get()
                    self.render_queue.task_done()
                
                if self.render_env:
                    start_time = time.time()
                    target_frame_time = 1.0 / self.fps
                    if start_time - self.last_render_time >= target_frame_time:
                        self.render_env.render(mode='human')
                        self.last_render_time = start_time
                        render_time = time.time() - start_time
                        global_logger.debug(f"Render time: {render_time:.3f}s, Queue size: {self.render_queue.qsize()}")
                        if render_time > target_frame_time:
                            global_logger.debug(f"Render time exceeds frame target: {render_time:.3f}s vs {target_frame_time:.3f}s")
                    time.sleep(max(0, target_frame_time - (time.time() - start_time)))
                else:
                    time.sleep(0.01)
            except Exception as e:
                global_logger.warning(f"Render thread error: {e}")
                time.sleep(0.01)
    
    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            try:
                if self.render_queue.qsize() == 0:
                    self.render_queue.put(self.training_env)
                    global_logger.debug(f"Queued render at step {self.step_count}")
            except Exception as e:
                global_logger.warning(f"Queue error: {e}")
        return True
    
    def _on_training_end(self) -> None:
        if self.render_env:
            try:
                self.render_env.close()
                global_logger.info("Closed render environment")
            except Exception as e:
                global_logger.warning(f"Error closing render env: {e}")

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'training.log'))
        ]
    )
    return logging.getLogger()

class NPZReplayEnvironment(gym.Env):
    def __init__(self, npz_directory):
        super().__init__()
        self.npz_files = glob.glob(os.path.join(npz_directory, '*.npz'))
        if not self.npz_files:
            raise ValueError(f"No NPZ files found in directory: {npz_directory}")
        
        self.current_file_idx = 0
        self.current_frame_idx = 0
        self.load_current_npz()
        
        self.num_envs = 1
        self._setup_spaces()
    
    def load_current_npz(self):
        npz_data = np.load(self.npz_files[self.current_file_idx])
        self.frames = npz_data['frames']
        self.actions = npz_data['actions']
        self.rewards = npz_data['rewards']
        self.current_frame_idx = 0
    
    def _setup_spaces(self):
        first_frame = self.frames[0]
        
        self.observation_space = spaces.Dict({
            'viewport': spaces.Box(low=0, high=255, shape=first_frame.shape, dtype=np.uint8),
            'enemy_vector': spaces.Box(low=-255, high=255, shape=(KUNGFU_MAX_ENEMIES * 2,), dtype=np.float32),
            'combat_status': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'projectile_vectors': spaces.Box(low=-255, high=255, shape=(MAX_PROJECTILES * 4,), dtype=np.float32),
            'enemy_proximity': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            'boss_info': spaces.Box(low=-255, high=255, shape=(3,), dtype=np.float32),
            'closest_enemy_direction': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })
        
        self.action_space = spaces.Discrete(11)
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_file_idx = np.random.randint(0, len(self.npz_files))
        self.load_current_npz()
        
        obs = {
            'viewport': self.frames[0],
            'enemy_vector': np.zeros(KUNGFU_MAX_ENEMIES * 2, dtype=np.float32),
            'combat_status': np.zeros(2, dtype=np.float32),
            'projectile_vectors': np.zeros(MAX_PROJECTILES * 4, dtype=np.float32),
            'enemy_proximity': np.zeros(1, dtype=np.float32),
            'boss_info': np.zeros(3, dtype=np.float32),
            'closest_enemy_direction': np.zeros(1, dtype=np.float32),
        }
        return obs, {"mimic_training": True}
    
    def step(self, action):
        frame = self.frames[self.current_frame_idx]
        reward = self.rewards[self.current_frame_idx]
        
        self.current_frame_idx += 1
        terminated = self.current_frame_idx >= len(self.frames) - 1
        truncated = False
        
        if terminated:
            self.current_frame_idx = len(self.frames) - 1
        
        obs = {
            'viewport': frame,
            'enemy_vector': np.zeros(KUNGFU_MAX_ENEMIES * 2, dtype=np.float32),
            'combat_status': np.zeros(2, dtype=np.float32),
            'projectile_vectors': np.zeros(MAX_PROJECTILES * 4, dtype=np.float32),
            'enemy_proximity': np.zeros(1, dtype=np.float32),
            'boss_info': np.zeros(3, dtype=np.float32),
            'closest_enemy_direction': np.zeros(1, dtype=np.float32),
        }
        
        return obs, reward, terminated, truncated, {"mimic_training": True}
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.frames[self.current_frame_idx]
        return None
    
    def close(self):
        pass

def make_kungfu_env():
    base_env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
    env = KungFuWrapper(base_env)
    return env

def create_npz_replay_env(npz_dir):
    return lambda: NPZReplayEnvironment(npz_dir)

def create_expert_dataset(npz_dir):
    """Create an expert dataset for imitation's GAIL from NPZ files."""
    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
    if not npz_files:
        raise ValueError(f"No NPZ files found in directory: {npz_dir}")
    
    trajectories = []
    for npz_file in npz_files:
        npz_data = np.load(npz_file)
        frames = npz_data['frames']
        actions = npz_data['actions']
        
        # Ensure len(obs) == len(acts) + 1
        if len(frames) == len(actions):
            # Truncate actions to be one less than frames
            actions = actions[:-1]
        elif len(frames) != len(actions) + 1:
            # Skip or log invalid files
            print(f"Warning: Skipping {npz_file} due to invalid length: {len(frames)} frames, {len(actions)} actions")
            continue
        
        # Convert frames to observation dicts
        observations = []
        for frame in frames:
            obs = {
                'viewport': frame,
                'enemy_vector': np.zeros(KUNGFU_MAX_ENEMIES * 2, dtype=np.float32),
                'combat_status': np.zeros(2, dtype=np.float32),
                'projectile_vectors': np.zeros(MAX_PROJECTILES * 4, dtype=np.float32),
                'enemy_proximity': np.zeros(1, dtype=np.float32),
                'boss_info': np.zeros(3, dtype=np.float32),
                'closest_enemy_direction': np.zeros(1, dtype=np.float32),
            }
            observations.append(obs)
        
        # Create a Trajectory object
        trajectory = Trajectory(
            obs=np.array(observations, dtype=object),
            acts=actions,
            infos=None,
            terminal=len(frames) >= len(actions) + 1
        )
        trajectories.append(trajectory)
    
    # Save trajectories to a pickle file for compatibility
    dataset_path = os.path.join(npz_dir, "expert_trajectories.pkl")
    with open(dataset_path, 'wb') as f:
        pickle.dump(trajectories, f)
    
    return trajectories

def train(args):
    global current_model, global_logger, global_model_path, experience_data
    experience_data = []
    
    global_logger = setup_logging(args.log_dir)
    global_logger.info(f"Starting training with {args.num_envs} envs and {args.timesteps} total timesteps")
    global_logger.info(f"Maximum number of enemies: {KUNGFU_MAX_ENEMIES}")
    global_model_path = args.model_path
    current_model = None

    if args.num_envs < 1:
        raise ValueError("Number of environments must be at least 1")
    
    training_mode = "mimic" if args.npz_dir and os.path.exists(args.npz_dir) else "live"
    if training_mode == "mimic":
        global_logger.info(f"Training in mimic mode using NPZ directory: {args.npz_dir}")
    else:
        global_logger.info("Training in live mode")
    
    if args.render:
        if training_mode == "mimic":
            global_logger.warning("Rendering is not supported in mimic mode. Ignoring --render flag.")
            args.render = False
        else:
            global_logger.info("Rendering enabled. Using a single environment for visualization.")
            args.num_envs = 1

    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 256, "n_stack": 4},
        "net_arch": dict(pi=[128, 128], vf=[256, 256])
    }
    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)

    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": 0.1,
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 5,
    }

    def initialize_model(env, expert_trajectories=None):
        if args.resume and os.path.exists(args.model_path + ".zip") and zipfile.is_zipfile(args.model_path + ".zip"):
            global_logger.info(f"Resuming training from {args.model_path}")
            try:
                custom_objects = {"policy_kwargs": policy_kwargs}
                if training_mode == "mimic" and expert_trajectories is not None:
                    # Load the underlying PPO policy and re-initialize GAIL
                    ppo_model = PPO.load(args.model_path, env=env, custom_objects=custom_objects,
                                       device="cuda" if args.cuda else "cpu")
                    # Create reward_net for the discriminator
                    reward_net = BasicRewardNet(
                        observation_space=env.observation_space,
                        action_space=env.action_space,
                        normalize_input_layer=RunningNorm,
                        hid_sizes=(64, 64),
                    )
                    model = GAIL(
                        venv=env,
                        gen_algo=ppo_model,
                        demonstrations=expert_trajectories,
                        demo_batch_size=params["batch_size"],
                        reward_net=reward_net,
                        gamma=0.99,
                        gae_lambda=0.95,
                        verbose=1,
                        device="cuda" if args.cuda else "cpu"
                    )
                else:
                    model = PPO.load(args.model_path, env=env, custom_objects=custom_objects,
                                   device="cuda" if args.cuda else "cpu")
                global_logger.info("Successfully loaded existing model")
                return model
            except Exception as e:
                global_logger.warning(f"Failed to load model: {e}. Starting new training session.")
        
        global_logger.info("Starting new training session")
        if training_mode == "mimic" and expert_trajectories is not None:
            ppo_policy = PPO(
                "MultiInputPolicy",
                env,
                learning_rate=params["learning_rate"],
                clip_range=params["clip_range"],
                ent_coef=params["ent_coef"],
                n_steps=params["n_steps"],
                batch_size=params["batch_size"],
                n_epochs=params["n_epochs"],
                gamma=0.99,
                gae_lambda=0.95,
                verbose=1,
                policy_kwargs=policy_kwargs,
                device="cuda" if args.cuda else "cpu"
            )
            # Create reward_net for the discriminator
            reward_net = BasicRewardNet(
                observation_space=env.observation_space,
                action_space=env.action_space,
                normalize_input_layer=RunningNorm,
                hid_sizes=(64, 64),
            )
            return GAIL(
                venv=env,
                gen_algo=ppo_policy,
                demonstrations=expert_trajectories,
                demo_batch_size=params["batch_size"],
                reward_net=reward_net,
                gamma=0.99,
                gae_lambda=0.95,
                verbose=1,
                device="cuda" if args.cuda else "cpu"
            )
        return PPO(
            "MultiInputPolicy",
            env,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu"
        )

    # Create environment and expert dataset
    expert_trajectories = None
    if training_mode == "mimic":
        env_fns = [create_npz_replay_env(args.npz_dir) for _ in range(args.num_envs)]
        try:
            expert_trajectories = create_expert_dataset(args.npz_dir)
            global_logger.info(f"Loaded expert dataset with {len(expert_trajectories)} trajectories")
        except Exception as e:
            global_logger.error(f"Failed to create expert dataset: {e}")
            raise
    else:
        env_fns = [make_kungfu_env for _ in range(args.num_envs)]
    
    if args.render:
        env = DummyVecEnv([make_kungfu_env])
    else:
        env = SubprocVecEnv(env_fns)
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env, skip=True)
    
    current_model = initialize_model(env, expert_trajectories)
    
    save_callback = SaveBestModelCallback(save_path=args.model_path)
    exp_callback = ExperienceCollectionCallback()
    callbacks = [save_callback, exp_callback]
    if args.render:
        callbacks.append(RenderCallback(render_freq=args.render_freq, fps=30))
    callback = CallbackList(callbacks)
    
    try:
        current_model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)
        experience_data.extend(exp_callback.experience_data)
    except Exception as e:
        global_logger.error(f"Training failed: {e}")
        try:
            env.close()
        except Exception as close_e:
            global_logger.warning(f"Failed to close environment: {close_e}")
        raise
    
    try:
        current_model.save(args.model_path)
        global_logger.info(f"Final model saved at {args.model_path}")
    except Exception as e:
        global_logger.error(f"Failed to save final model: {e}")
    
    try:
        env.close()
    except Exception as e:
        global_logger.warning(f"Failed to close environment during cleanup: {e}")
    
    experience_count = len(experience_data)
    global_logger.info(f"Training completed. Total experience collected: {experience_count} steps")
    
    with open(f"{global_model_path}_experience_count.txt", "w") as f:
        f.write(f"Total experience collected: {experience_count} steps\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training parameters: num_envs={args.num_envs}, total_timesteps={args.timesteps}, mode={training_mode}\n")
        if args.npz_dir and training_mode == "mimic":
            f.write(f"NPZ directory: {args.npz_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO or GAIL model for Kung Fu with live or mimic training")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo", help="Path to save the trained model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Default clip range for PPO/GAIL")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--npz_dir", default=None, help="Directory containing NPZ recordings for mimic training with GAIL")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--render_freq", type=int, default=256, help="Render every N steps")
    parser.add_argument("--render_fps", type=int, default=30, help="Target rendering FPS")
    
    args = parser.parse_args()
    train(args)