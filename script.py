import argparse
import os
import gym
import retro
import numpy as np
import torch
import torch.nn as nn
import signal
import sys
import logging
import cv2
import time
from collections import deque
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.logger import configure
from tqdm import tqdm
from gym import spaces, Wrapper

global_model = None
global_model_path = None
terminate_flag = False
subproc_envs = None

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    sb3_logger = configure(log_dir, ["tensorboard", "log", "stdout"])
    logger = logging.getLogger('kungfu_ppo')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    log_path = os.path.join(log_dir, "training.log")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(file_formatter)
    logger.addHandler(console_handler)
    return logger, sb3_logger

def cleanup():
    global subproc_envs
    logger = logging.getLogger('kungfu_ppo')
    if subproc_envs is not None:
        try:
            subproc_envs.close()
        except Exception as e:
            logger.warning(f"Error closing subproc_envs: {e}")
        subproc_envs = None
    torch.cuda.empty_cache()

def signal_handler(sig, frame):
    global terminate_flag, global_model, global_model_path
    logger = logging.getLogger('kungfu_ppo')
    logger.info(f"Signal {sig} received! Preparing to terminate...")
    if global_model is not None and global_model_path is not None:
        try:
            global_model.save(f"{global_model_path}/kungfu_ppo")
            logger.info(f"Emergency save completed: {global_model_path}/kungfu_ppo.zip")
        except Exception as e:
            logger.error(f"Error saving model during signal handler: {e}")
    terminate_flag = True
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class VisionTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=256, num_heads=8, num_layers=4, patch_size=12):
        super().__init__(observation_space, features_dim=embed_dim)
        self.img_size = observation_space.shape[1]  # e.g., 84
        self.in_channels = observation_space.shape[0]  # e.g., 16 (n_stack)
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2  # e.g., (84 // 12) ^ 2 = 49
        self.embed_dim = embed_dim

        # Patch embedding convolution
        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Initialize positional embeddings correctly
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=1024, 
            dropout=0.1,  # Reduced dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, in_channels, height, width), e.g., (batch_size, 16, 84, 84)
        x = self.patch_embed(x)  # (batch_size, embed_dim, H', W'), e.g., (batch_size, 256, 7, 7)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim), e.g., (batch_size, 49, 256)
        
        # Ensure positional embeddings match the input size
        pos_embed = self.pos_embed[:, :x.shape[1], :]  # Truncate if necessary
        x = x + pos_embed  # Add positional embeddings
        
        x = self.transformer(x)  # (batch_size, num_patches, embed_dim)
        x = self.norm(x)  # Normalize
        x = x.mean(dim=1)  # Global average pooling: (batch_size, embed_dim)
        return x
    
class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=VisionTransformer,
            features_extractor_kwargs=dict(embed_dim=256, num_heads=8, num_layers=4, patch_size=12),
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            **kwargs
        )

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(11)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # Kick
            [0,1,0,0,0,0,0,0,0,0,0,0],  # Punch
            [1,0,0,0,0,0,1,0,0,0,0,0],  # Kick+Left
            [1,0,0,0,0,0,0,0,1,0,0,0],  # Kick+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # Punch+Left
            [0,1,0,0,0,0,0,0,1,0,0,0],  # Punch+Right
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Down (Duck)
            [0,0,1,0,0,0,0,0,0,0,0,0]   # Up (Jump)
        ]
        self.action_names = ["No action", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right", "Duck", "Jump"]

    def action(self, action):
        return self._actions[action]

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[..., np.newaxis].astype(np.uint8)

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.ram_positions = {
            'score': 0x0531,
            'scroll': 0x00E5,
            'hero_hp': 0x04A6,
            'hero_pos_x': 0x0094,
            'stage': 0x00E4
        }
        self.last_score = 0
        self.last_scroll = 0
        self.last_hp = None
        self.last_pos_x = 0
        self.start_time = None
        self.episode_start_time = None

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        ram = self.env.get_ram()
        self.last_hp = ram[self.ram_positions['hero_hp']]
        self.last_score = 0
        self.last_scroll = 0
        self.last_pos_x = ram[self.ram_positions['hero_pos_x']]
        self.episode_start_time = time.time()
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)
        ram = self.env.get_ram()
        
        current_score = ram[self.ram_positions['score']] * 100
        current_scroll = ram[self.ram_positions['scroll']]
        current_hp = ram[self.ram_positions['hero_hp']]
        current_pos_x = ram[self.ram_positions['hero_pos_x']]
        current_stage = ram[self.ram_positions['stage']]

        # Overflow-safe delta calculations
        score_delta = current_score - self.last_score
        scroll_delta = current_scroll - self.last_scroll if current_scroll >= self.last_scroll else (current_scroll + 256 - self.last_scroll)
        pos_delta = current_pos_x - self.last_pos_x if current_pos_x >= self.last_pos_x else (current_pos_x + 256 - self.last_pos_x)
        hp_loss = max(0, int(self.last_hp) - int(current_hp)) if self.last_hp is not None else 0

        # Reward components
        reward = 0
        reward += score_delta * 5           # Score points
        reward += scroll_delta * 10         # Level progress
        reward += pos_delta * 2.0           # Movement (increased from 0.2)
        reward -= hp_loss * 10              # Health penalty (reduced from 50)
        
        # Small penalty for standing still
        if pos_delta == 0:
            reward -= 0.1                   # Reduced from 0.3

        # Update last values
        self.last_score = current_score
        self.last_scroll = current_scroll
        self.last_hp = current_hp
        self.last_pos_x = current_pos_x

        # Info dictionary
        survival_time = time.time() - self.episode_start_time if self.episode_start_time else 0
        info.update({
            'score': current_score,
            'hp': current_hp,
            'scroll': current_scroll,
            'pos_x': current_pos_x,
            'stage': current_stage,
            'survival_time': survival_time,
            'score_delta': score_delta,
            'scroll_delta': scroll_delta,
            'hp_loss': hp_loss,
            'pos_delta': pos_delta
        })

        return obs, reward, done, info

def make_kungfu_env(seed=None, render=False):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    env = PreprocessFrame(env)
    env = KungFuRewardWrapper(env)
    if seed is not None:
        env.seed(seed)
    if render:
        env.render_mode = 'human'
    return env

def make_env(rank, seed=0, render=False):
    def _init():
        env = make_kungfu_env(seed=seed + rank, render=(rank == 0 and render))
        return env
    return _init

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, log_interval=10):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_scores = []
        self.episode_scrolls = []
        self.episode_survival_times = []
        self.episode_hps = []
        self.last_log_time = 0
        self.episode_count = 0
        self.reward_buffer = deque(maxlen=100)
        self.score_buffer = deque(maxlen=100)
        self.scroll_buffer = deque(maxlen=100)
        self.time_buffer = deque(maxlen=100)
        self.hp_buffer = deque(maxlen=100)

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress", leave=True)
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if terminate_flag:
            return False
        
        current_steps = self.model.num_timesteps
        if current_steps > self.pbar.n:
            self.pbar.n = min(current_steps, self.total_timesteps)
            self.pbar.refresh()
            
        if len(self.model.ep_info_buffer) > 0 and "r" in self.model.ep_info_buffer[0]:
            ep_info = self.model.ep_info_buffer[0]
            self.episode_count += 1
            self.episode_rewards.append(ep_info["r"])
            self.reward_buffer.append(ep_info["r"])
            
            if "score" in ep_info:
                self.episode_scores.append(ep_info["score"])
                self.score_buffer.append(ep_info["score"])
                
            if "scroll" in ep_info:
                self.episode_scrolls.append(ep_info["scroll"])
                self.scroll_buffer.append(ep_info["scroll"])
                
            if "t" in ep_info:
                self.episode_survival_times.append(ep_info["t"])
                self.time_buffer.append(ep_info["t"])
                
            if "hp" in ep_info:
                self.episode_hps.append(ep_info["hp"])
                self.hp_buffer.append(ep_info["hp"])
                
            if "l" in ep_info:
                self.episode_lengths.append(ep_info["l"])
                
            current_time = time.time()
            if current_time - self.last_log_time >= self.log_interval:
                self._log_metrics()
                self.last_log_time = current_time
                
        return True

    def _log_metrics(self):
        if len(self.reward_buffer) > 0:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.reward_buffer))
            self.logger.record("rollout/ep_rew_max", np.max(self.reward_buffer))
            self.logger.record("rollout/ep_rew_min", np.min(self.reward_buffer))
            
            if len(self.score_buffer) > 0:
                self.logger.record("metrics/score_mean", np.mean(self.score_buffer))
                self.logger.record("metrics/score_max", np.max(self.score_buffer))
                
            if len(self.scroll_buffer) > 0:
                self.logger.record("metrics/scroll_mean", np.mean(self.scroll_buffer))
                self.logger.record("metrics/scroll_max", np.max(self.scroll_buffer))
                
            if len(self.time_buffer) > 0:
                self.logger.record("metrics/survival_time_mean", np.mean(self.time_buffer))
                self.logger.record("metrics/survival_time_max", np.max(self.time_buffer))
                
            if len(self.hp_buffer) > 0:
                self.logger.record("metrics/hp_mean", np.mean(self.hp_buffer))
                self.logger.record("metrics/hp_min", np.min(self.hp_buffer))
                
            if len(self.episode_lengths) > 0:
                self.logger.record("rollout/ep_len_mean", np.mean(self.episode_lengths))
                
            elapsed_time = time.time() - self.start_time
            self.logger.record("time/episodes", self.episode_count)
            self.logger.record("time/time_elapsed", elapsed_time)
            self.logger.record("time/fps", int(self.model.num_timesteps / elapsed_time))
            
            # Debug metrics
            if "pos_delta" in self.model.ep_info_buffer[0]:
                self.logger.record("debug/pos_delta", self.model.ep_info_buffer[0]["pos_delta"])
            if "scroll_delta" in self.model.ep_info_buffer[0]:
                self.logger.record("debug/scroll_delta", self.model.ep_info_buffer[0]["scroll_delta"])
            if "hp_loss" in self.model.ep_info_buffer[0]:
                self.logger.record("debug/hp_loss", self.model.ep_info_buffer[0]["hp_loss"])
                
            self.logger.dump(self.model.num_timesteps)

    def _on_training_end(self):
        self.pbar.close()
        if global_model_path:
            np.save(os.path.join(global_model_path, 'episode_rewards.npy'), np.array(self.episode_rewards))
            np.save(os.path.join(global_model_path, 'episode_scores.npy'), np.array(self.episode_scores))
            np.save(os.path.join(global_model_path, 'episode_scrolls.npy'), np.array(self.episode_scrolls))
            np.save(os.path.join(global_model_path, 'episode_survival_times.npy'), np.array(self.episode_survival_times))
            np.save(os.path.join(global_model_path, 'episode_hps.npy'), np.array(self.episode_hps))
            
            with open(os.path.join(global_model_path, 'training_summary.txt'), 'w') as f:
                f.write(f"Training Summary\n")
                f.write(f"Total timesteps: {self.model.num_timesteps}\n")
                f.write(f"Average reward: {np.mean(self.episode_rewards) if self.episode_rewards else 0:.2f}\n")
                f.write(f"Max reward: {np.max(self.episode_rewards) if self.episode_rewards else 0:.2f}\n")
                f.write(f"Average score: {np.mean(self.episode_scores) if self.episode_scores else 0:.2f}\n")
                f.write(f"Max score: {np.max(self.episode_scores) if self.episode_scores else 0:.2f}\n")
                f.write(f"Average scroll: {np.mean(self.episode_scrolls) if self.episode_scrolls else 0:.2f}\n")
                f.write(f"Max scroll: {np.max(self.episode_scrolls) if self.episode_scrolls else 0:.2f}\n")
                f.write(f"Average survival: {np.mean(self.episode_survival_times) if self.episode_survival_times else 0:.2f}s\n")
                f.write(f"Training time: {time.time() - self.start_time:.2f}s\n")

def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value
    return func

def create_base_env(args, seed_offset=0, render=False):
    """Create the base environment without frame stacking or normalization."""
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed + seed_offset) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed + seed_offset, render)])
    return env

def train(args):
    global global_model, global_model_path, subproc_envs
    global_model_path = args.model_path
    os.makedirs(args.model_path, exist_ok=True)
    logger, sb3_logger = setup_logging(args.log_dir)
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    model_file = f"{args.model_path}/kungfu_ppo.zip"
    vecnorm_file = f"{args.model_path}/kungfu_ppo_vecnormalize.pkl"
    desired_n_stack = 16  # The desired frame stacking for this training session

    # Create the environment with the desired configuration
    env = create_base_env(args)
    env = VecFrameStack(env, n_stack=desired_n_stack)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # Check if we're resuming training
    if args.resume and os.path.exists(model_file):
        logger.info(f"Resuming training from {model_file}")
        model = PPO.load(model_file, env=env, device=device)
        if os.path.exists(vecnorm_file):
            env = VecNormalize.load(vecnorm_file, env)
            logger.info("Loaded saved VecNormalize stats")
    else:
        logger.info("Starting new training.")
        # Initialize a new PPO model
        ppo_kwargs = dict(
            policy=TransformerPolicy,
            env=env,
            learning_rate=linear_schedule(args.learning_rate),
            n_steps=2048,
            batch_size=64,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            ent_coef=0.01,
            verbose=1,
            device=device,
            tensorboard_log=args.log_dir,
            vf_coef=0.5,
            max_grad_norm=0.5
        )
        model = PPO(**ppo_kwargs)

    model.set_logger(sb3_logger)
    global_model = model
    callbacks = [ProgressCallback(args.timesteps)] if args.progress_bar else []

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=1,
            tb_log_name="PPO_KungFu",
            reset_num_timesteps=not args.resume  # Don't reset timestep counter if resuming
        )
        model.save(model_file)
        env.save(vecnorm_file)
        logger.info(f"Training completed! Saved to {model_file}")
    except Exception as e:
        logger.error(f"Training error: {e}")
        model.save(model_file)
        env.save(vecnorm_file)
    finally:
        cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KungFu Master with PPO")
    parser.add_argument("--train", action="store_true", help="Train mode")
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Model directory")
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render gameplay")
    parser.add_argument("--timesteps", type=int, default=500000, help="Training steps")
    parser.add_argument("--num_envs", type=int, default=8, help="Parallel environments")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--clip_range", type=float, default=0.1, help="PPO clip range")
    parser.add_argument("--log_dir", default="logs", help="Log directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    
    args = parser.parse_args()
    if args.train:
        train(args)