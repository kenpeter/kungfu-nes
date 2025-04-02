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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
from gym import spaces, Wrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global_model = None
global_model_path = None
terminate_flag = False
subproc_envs = None

def cleanup():
    global subproc_envs
    if subproc_envs is not None:
        try:
            subproc_envs.close()
        except Exception as e:
            logging.warning(f"Error closing subproc_envs: {e}")
        subproc_envs = None
    torch.cuda.empty_cache()

def signal_handler(sig, frame):
    global terminate_flag, global_model, global_model_path
    logging.info(f"Signal {sig} received! Preparing to terminate...")
    if global_model is not None and global_model_path is not None:
        try:
            global_model.save(f"{global_model_path}/kungfu_ppo")
            logging.info(f"Emergency save completed: {global_model_path}/kungfu_ppo.zip")
        except Exception as e:
            logging.error(f"Error saving model during signal handler: {e}")
    terminate_flag = True
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Vision Transformer Feature Extractor
class VisionTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=256, num_heads=8, num_layers=6, patch_size=14):
        super().__init__(observation_space, features_dim=embed_dim)
        self.img_size = observation_space.shape[1]
        self.in_channels = observation_space.shape[0]
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

# Custom Policy using Vision Transformer
class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=VisionTransformer,
            features_extractor_kwargs=dict(embed_dim=256, num_heads=8, num_layers=6, patch_size=14),
            **kwargs
        )

# Simplified Action Wrapper
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

# Fixed Observation Wrapper
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        # Convert to grayscale
        obs = np.dot(obs[..., :3], [0.299, 0.587, 0.114])  # Shape: (H, W)
        # Resize to 84x84 directly
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[..., np.newaxis].astype(np.uint8)  # Shape: (84, 84, 1)

# Simplified Reward Wrapper
class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_scroll = 0
        self.last_hp = None
        self.ram_positions = {
            'score': 0x0531,  # Simplified to one byte for demo
            'scroll': 0x00E5,
            'hero_hp': 0x04A6,
            'hero_pos_x': 0x0094
        }

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        ram = self.env.get_ram()
        self.last_hp = ram[self.ram_positions['hero_hp']]
        self.last_score = 0
        self.last_scroll = 0
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)
        ram = self.env.get_ram()
        current_score = ram[self.ram_positions['score']] * 100
        current_scroll = ram[self.ram_positions['scroll']]
        current_hp = ram[self.ram_positions['hero_hp']]

        # Simple reward calculation
        reward = 0
        score_delta = current_score - self.last_score
        scroll_delta = current_scroll - self.last_scroll if current_scroll >= self.last_scroll else (current_scroll + 256 - self.last_scroll)
        # Cast to int to avoid uint8 overflow
        hp_loss = max(0, int(self.last_hp) - int(current_hp)) if self.last_hp is not None else 0

        reward += score_delta * 5
        reward += scroll_delta * 10
        reward -= hp_loss * 50

        self.last_score = current_score
        self.last_scroll = current_scroll
        self.last_hp = current_hp

        info['score'] = current_score
        info['hp'] = current_hp
        return obs, reward, done, info

# Environment Factory
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

# Training Callback
class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        if terminate_flag:
            return False
        self.pbar.update(1)
        return True

    def _on_training_end(self):
        self.pbar.close()

# Training Function
def train(args):
    global global_model, global_model_path, subproc_envs
    global_model_path = args.model_path

    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    if args.num_envs > 1:
        subproc_envs = SubprocVecEnv([make_env(i, args.seed) for i in range(args.num_envs)])
        env = subproc_envs
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.render)])

    env = VecFrameStack(env, n_stack=4)

    model_file = f"{args.model_path}/kungfu_ppo.zip"
    ppo_kwargs = dict(
        policy=TransformerPolicy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=0.01,
        verbose=1,
        device=device
    )

    if os.path.exists(model_file) and args.resume:
        logging.info(f"Resuming training from {model_file}")
        try:
            model = PPO.load(model_file, env=env, device=device)
        except Exception as e:
            logging.error(f"Failed to load model: {e}. Starting new training.")
            model = PPO(**ppo_kwargs)
    else:
        logging.info("Starting new training.")
        model = PPO(**ppo_kwargs)

    global_model = model

    callbacks = [ProgressCallback(args.timesteps)] if args.progress_bar else []

    try:
        model.learn(total_timesteps=args.timesteps, callback=callbacks)
        model.save(model_file)
        logging.info(f"Training completed. Model saved to {model_file}")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        model.save(model_file)
        logging.info(f"Model saved due to error: {model_file}")
    finally:
        cleanup()

# Play Function
def play(args):
    env = make_kungfu_env(render=args.render)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if not os.path.exists(model_file):
        logging.error(f"No model found at {model_file}.")
        return

    model = PPO.load(model_file, env=env)
    obs = env.reset()
    done = False
    total_reward = 0

    while not done and not terminate_flag:
        action, _ = model.predict(obs, deterministic=args.deterministic)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        if args.render:
            env.render()

    logging.info(f"Total reward: {total_reward}")
    env.close()
    cleanup()

# Evaluate Function
def evaluate(args):
    env = make_kungfu_env()
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if not os.path.exists(model_file):
        logging.error(f"No model found at {model_file}.")
        return

    model = PPO.load(model_file, env=env)
    rewards = []

    for _ in range(args.eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward[0]
        rewards.append(episode_reward)

    avg_reward = np.mean(rewards)
    logging.info(f"Average reward over {args.eval_episodes} episodes: {avg_reward:.2f}")
    env.close()
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play KungFu Master with PPO")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Train the model")
    mode_group.add_argument("--play", action="store_true", help="Play with the trained model")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate the model")

    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps")
    parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel envs")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Number of eval episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip_range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")

    args = parser.parse_args()
    if not any([args.train, args.play, args.evaluate]):
        args.train = True

    if args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        evaluate(args)