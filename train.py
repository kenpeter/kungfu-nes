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
from kungfu_env import KungFuWrapper, SimpleCNN, KUNGFU_MAX_ENEMIES, MAX_PROJECTILES, KUNGFU_OBSERVATION_SPACE, KUNGFU_ACTIONS, KUNGFU_ACTION_NAMES
import threading
import queue
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, save_path, verbose=1, behaviour_mode=False):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_score = 0
        self.behaviour_mode = behaviour_mode
        self.actions = KUNGFU_ACTIONS
        self.action_names = KUNGFU_ACTION_NAMES

    def _on_step(self) -> bool:
        if self.behaviour_mode:
            if self.num_timesteps % 5000 == 0:
                try:
                    self.model.save(self.save_path)
                    if self.verbose > 0:
                        print(f"Behaviour model saved at step {self.num_timesteps}")
                except Exception as e:
                    print(f"Failed to save model: {e}")
            return True
            
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

class BehaviourCloningLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, model_output, target_actions):
        return self.loss_fn(model_output, target_actions.float())

class ExpertReplayEnvironment(gym.Env):
    def __init__(self, npz_directory):
        super().__init__()
        self.npz_files = glob.glob(os.path.join(npz_directory, '*.npz'))
        if not self.npz_files:
            raise ValueError(f"No NPZ files found in directory: {npz_directory}")
        
        self.current_file_idx = 0
        self.current_frame_idx = 0
        
        self.base_env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
        self.env = KungFuWrapper(self.base_env)
        self.load_current_npz()
        
        self.num_envs = 1
        self._setup_spaces()
    
    def load_current_npz(self):
        npz_data = np.load(self.npz_files[self.current_file_idx])
        self.frames = npz_data['frames']
        self.actions = npz_data['actions']
        if np.max(self.actions) >= len(KUNGFU_ACTIONS):
            global_logger.warning(f"Clipping invalid action indices in {self.npz_files[self.current_file_idx]} (max={np.max(self.actions)})")
            self.actions = np.clip(self.actions, 0, len(KUNGFU_ACTIONS) - 1)
        self.rewards = np.ones_like(npz_data['rewards'])
        self.current_frame_idx = 0
        global_logger.debug(f"Loaded NPZ file {self.npz_files[self.current_file_idx]} with {len(self.actions)} actions")
    
    def _setup_spaces(self):
        self.observation_space = KUNGFU_OBSERVATION_SPACE
        self.action_space = spaces.Discrete(len(KUNGFU_ACTIONS))
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_file_idx = np.random.randint(0, len(self.npz_files))
        self.load_current_npz()
        
        obs, info = self.env.reset()
        obs['viewport'] = self.frames[0]
        expert_action = int(self.actions[0]) if len(self.actions) > 0 else 0
        return obs, {"behaviour_training": True, "expert_action": expert_action}
    
    def step(self, action):
        frame = self.frames[self.current_frame_idx]
        expert_action = int(self.actions[self.current_frame_idx]) if self.current_frame_idx < len(self.actions) else 0
        
        action_vector = KUNGFU_ACTIONS[action]
        expert_action_vector = KUNGFU_ACTIONS[expert_action]
        
        obs, _, terminated, truncated, info = self.env.step(action)
        obs['viewport'] = frame
        
        reward = 1.0 if action == expert_action else 0.0
        
        self.current_frame_idx += 1
        terminated = self.current_frame_idx >= len(self.frames) - 1
        truncated = False
        
        if terminated:
            self.current_file_idx = (self.current_file_idx + 1) % len(self.npz_files)
            self.load_current_npz()
        
        next_frame_idx = min(self.current_frame_idx, len(self.frames) - 1)
        next_expert_action = int(self.actions[next_frame_idx]) if next_frame_idx < len(self.actions) else 0
        obs['viewport'] = self.frames[next_frame_idx]
        
        info.update({
            "behaviour_training": True,
            "expert_action": next_expert_action,
            "action_match": action == expert_action
        })
        
        return obs, reward, terminated, truncated, info
    
    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.frames[self.current_frame_idx]
        return None
    
    def close(self):
        self.env.close()

def create_expert_replay_env(npz_dir):
    return lambda: ExpertReplayEnvironment(npz_dir)

def extract_expert_data(npz_dir):
    npz_files = glob.glob(os.path.join(npz_dir, '*.npz'))
    if not npz_files:
        return None, 0
    
    total_frames = 0
    total_actions = 0
    action_counts = {}
    
    for npz_file in npz_files:
        try:
            npz_data = np.load(npz_file)
            frames = npz_data['frames']
            actions = npz_data['actions']
            
            total_frames += len(frames)
            total_actions += len(actions)
            
            for action in actions:
                action_idx = int(action)
                if action_idx < 0 or action_idx >= len(KUNGFU_ACTIONS):
                    global_logger.warning(f"Invalid action index in {npz_file}: {action_idx}")
                    continue
                action_key = action_idx
                if action_key not in action_counts:
                    action_counts[action_key] = 0
                action_counts[action_key] += 1
        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
    
    return action_counts, total_frames

class BehaviourCloningCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.matches = 0
        self.total = 0
        self.last_print_time = time.time()
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        for info in infos:
            if info.get('behaviour_training', False):
                if 'action_match' in info:
                    self.matches += 1 if info['action_match'] else 0
                    self.total += 1
        
        current_time = time.time()
        if current_time - self.last_print_time >= 10:
            if self.total > 0:
                match_rate = self.matches / self.total * 100
                print(f"Behaviour cloning: {self.matches}/{self.total} actions matched ({match_rate:.2f}%)")
            self.last_print_time = current_time
        
        return True

def make_kungfu_env():
    base_env = retro.make('KungFu-Nes', use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array")
    env = KungFuWrapper(base_env)
    return env

def map_multibinary_to_discrete(action):
    for i, predefined_action in enumerate(KUNGFU_ACTIONS):
        if np.array_equal(action, predefined_action):
            return i
    return 0

def map_discrete_to_multibinary(action_idx):
    return np.array(KUNGFU_ACTIONS[action_idx], dtype=np.uint8)

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
    
    training_mode = "behaviour" if args.npz_dir and os.path.exists(args.npz_dir) else "live"
    if training_mode == "behaviour":
        global_logger.info(f"Training in behaviour cloning mode using NPZ directory: {args.npz_dir}")
        action_counts, total_frames = extract_expert_data(args.npz_dir)
        if action_counts:
            global_logger.info(f"Expert data contains {total_frames} frames with action distribution:")
            for action, count in sorted(action_counts.items()):
                global_logger.info(f"  Action {action} ({KUNGFU_ACTION_NAMES[action]}): {count} occurrences ({count/total_frames*100:.2f}%)")
    else:
        global_logger.info("Training in live reinforcement learning mode")
    
    if args.render:
        if training_mode == "behaviour":
            global_logger.warning("Rendering is not supported in behaviour cloning mode. Ignoring --render flag.")
            args.render = False
        else:
            global_logger.info("Rendering enabled. Using a single environment for visualization.")
            args.num_envs = 1

    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 256, "n_stack": 4},
        "net_arch": dict(pi=[128, 128], vf=[256, 256]),
        "activation_fn": nn.ReLU,
    }
    
    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)
    ent_coef = 0.1
    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": ent_coef,
        "n_steps": 512,
        "batch_size": 32,
        "n_epochs": 5,
    }

    def initialize_model(env):
        if args.resume and os.path.exists(args.model_path + ".zip") and zipfile.is_zipfile(args.model_path + ".zip"):
            global_logger.info(f"Resuming training from {args.model_path}")
            try:
                custom_objects = {"policy_kwargs": policy_kwargs}
                model = PPO.load(args.model_path, env=env, custom_objects=custom_objects,
                                 device="cuda" if args.cuda else "cpu")
                expected_actions = len(KUNGFU_ACTIONS)
                model_actions = model.policy.action_space.n
                if model_actions != expected_actions:
                    global_logger.warning(f"Model action space ({model_actions}) does not match environment ({expected_actions}). Retraining recommended.")
                global_logger.info("Successfully loaded existing model")
                return model
            except Exception as e:
                global_logger.warning(f"Failed to load model: {e}. Starting new training session.")
        
        global_logger.info("Starting new training session")
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

    env_fns = [create_expert_replay_env(args.npz_dir) if training_mode == "behaviour" else make_kungfu_env for _ in range(args.num_envs)]
    
    if args.render:
        env = DummyVecEnv([make_kungfu_env])
    else:
        env = SubprocVecEnv(env_fns)
    
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    env = VecTransposeImage(env, skip=True)
    
    current_model = initialize_model(env)
    
    save_callback = SaveBestModelCallback(save_path=args.model_path, behaviour_mode=(training_mode == "behaviour"))
    exp_callback = ExperienceCollectionCallback()
    callbacks = [save_callback, exp_callback]
    
    if training_mode == "behaviour":
        bc_callback = BehaviourCloningCallback()
        callbacks.append(bc_callback)
    
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
    experience_data = []
    global_logger.info(f"Training completed. Total experience collected: {experience_count} steps")
    
    # Print final clone percentage if in behaviour cloning mode
    if training_mode == "behaviour":
        for cb in callbacks:
            if isinstance(cb, BehaviourCloningCallback):
                if cb.total > 0:
                    final_match_rate = (cb.matches / cb.total) * 100
                    global_logger.info(f"Final behaviour cloning match rate: {cb.matches}/{cb.total} actions matched ({final_match_rate:.2f}%)")
                else:
                    global_logger.info("No behaviour cloning data collected during training.")
                break
    
    with open(f"{global_model_path}_experience_count.txt", "w") as f:
        f.write(f"Total experience collected: {experience_count} steps\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training parameters: num_envs={args.num_envs}, total_timesteps={args.timesteps}, mode={training_mode}\n")
        if args.npz_dir and training_mode == "behaviour":
            f.write(f"NPZ directory: {args.npz_dir}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO model for Kung Fu with live RL or behaviour cloning")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo", help="Path to save the trained model")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total timesteps for training")
    parser.add_argument("--clip_range", type=float, default=0.2, help="Default clip range for PPO")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--resume", action="store_true", help="Resume training from the saved model")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--npz_dir", default=None, help="Directory containing NPZ recordings for behaviour cloning")
    parser.add_argument("--render", action="store_true", help="Render the environment during training")
    parser.add_argument("--render_freq", type=int, default=256, help="Render every N steps")
    parser.add_argument("--render_fps", type=int, default=30, help="Target rendering FPS")
    
    args = parser.parse_args()
    try:
        train(args)
    finally:
        if current_model and hasattr(current_model, 'env'):
            try:
                current_model.env.close()
            except Exception as e:
                global_logger.error(f"Failed to close environment during final cleanup: {e}")
        if args.cuda:
            torch.cuda.empty_cache()