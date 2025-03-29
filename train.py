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

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('training.log')]
)

global_model = None
global_model_path = None

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(5)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Left
            [1,0,0,0,0,0,1,0,0,0,0,0],  # B+Left
            [0,1,0,0,0,0,1,0,0,0,0,0],  # A+Left
            [1,1,0,0,0,0,1,0,0,0,0,0]   # B+A+Left
        ]

    def action(self, action):
        return self._actions[action]

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset()

    def reset(self):
        self.last_score = 0
        self.last_x = 0
        self.last_health = 46  # Correct initial health
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_score = info.get('score', 0)
        current_x = info.get('x_pos', 0)
        health = info.get('health', 46)
        
        score_delta = current_score - self.last_score
        x_delta = current_x - self.last_x
        
        reward = (
            score_delta * 50.0 +  # Boost combat reward
            (-x_delta) * 500 +
            min(0, x_delta) * -200 +
            (health - self.last_health) * 10.0 -
            0.01
        )
        
        logging.debug(f"Score: {current_score} (Delta: {score_delta}), X: {current_x} (Delta: {x_delta}), Health: {health}, Reward: {reward:.2f}")
        
        self.last_score = current_score
        self.last_x = current_x
        self.last_health = health
        
        done = done or health <= 0
        return obs, reward, done, info

class ProgressBarCallback(BaseCallback):
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

class SaveCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix="model"):
        super().__init__()
        self.save_freq = save_freq
        self.save_path = os.path.abspath(save_path)
        self.name_prefix = name_prefix
        self.timesteps_elapsed = 0
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _on_step(self):
        self.timesteps_elapsed += self.training_env.num_envs
        if self.timesteps_elapsed >= self.save_freq:
            self.timesteps_elapsed = 0
            checkpoint_path = f"{self.save_path}_checkpoint_{self.num_timesteps}"
            self.model.save(checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
        return True

def signal_handler(sig, frame):
    if global_model is not None and global_model_path is not None:
        logging.info("Interrupt received! Saving model...")
        emergency_path = f"{global_model_path}_emergency_{int(time.time())}"
        global_model.save(emergency_path)
        logging.info(f"Emergency save completed: {emergency_path}")
    sys.exit(0)

def make_env(render=False):
    try:
        env = retro.make(
            game="KungFu-Nes",
            use_restricted_actions=retro.Actions.ALL,
            obs_type=retro.Observations.IMAGE
        )
        env = KungFuDiscreteWrapper(env)
        env = KungFuRewardWrapper(env)
        env = TimeLimit(env, max_episode_steps=5000)
        return env
    except Exception as e:
        logging.error(f"Failed to create environment: {e}")
        raise

def train(args):
    global global_model, global_model_path
    
    logging.info("Initializing training...")
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        env = make_vec_env(
            lambda: make_env(args.render),
            n_envs=8 if args.render else 8,
            vec_env_cls=SubprocVecEnv if args.num_envs > 1 else DummyVecEnv,
            monitor_dir="./monitor_logs",
            seed=args.seed
        )
        env = VecFrameStack(env, n_stack=4)
        env = VecMonitor(env)
        logging.info("Environment created successfully")
    except Exception as e:
        logging.error(f"Environment creation failed: {e}")
        raise

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")
    
    model_path = os.path.abspath(args.model_path)
    global_model_path = model_path
    
    if args.resume and os.path.exists(f"{model_path}.zip"):
        model = PPO.load(f"{model_path}.zip", env=env, device=device, custom_objects={"learning_rate": args.learning_rate})
        logging.info(f"Resumed training from {model_path}.zip")
    else:
        model = PPO(
            "CnnPolicy",
            env,
            device=device,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            tensorboard_log=args.log_dir
        )
        logging.info("Created new model")
    
    global_model = model

    callbacks = []
    if args.progress_bar:
        callbacks.append(ProgressBarCallback(args.timesteps))
    if args.save_freq > 0:
        callbacks.append(SaveCheckpointCallback(save_freq=args.save_freq, save_path=model_path))

    try:
        logging.info(f"Starting training for {args.timesteps} timesteps")
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            tb_log_name=os.path.basename(model_path),
            reset_num_timesteps=not args.resume
        )
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        logging.info("Saving final model...")
        model.save(model_path)
        env.close()
        logging.info(f"Final model saved to {model_path}.zip")

def play(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model_path = f"{args.model_path}.zip" if not args.model_path.endswith(".zip") else args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(model_path, device=device)
    logging.info(f"Loaded model from {model_path}")

    env = make_vec_env(lambda: make_env(render=args.render), n_envs=1, vec_env_cls=DummyVecEnv)
    env = VecFrameStack(env, n_stack=4)
    raw_env = env.envs[0].env.env if args.render else None

    obs = env.reset()
    total_reward = 0
    episode_count = 0
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=False)  # Explore actions
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            if args.render and raw_env:
                raw_env.render()
            if done:
                episode_count += 1
                logging.info(f"Episode {episode_count} finished. Score: {info[0].get('score', 0)}, Total Reward: {total_reward:.2f}")
                total_reward = 0
                obs = env.reset()
    except KeyboardInterrupt:
        logging.info("Play session ended by user")
    finally:
        env.close()
        logging.info("Play environment closed")

def list_checkpoints(args):
    base_path = args.model_path
    checkpoint_dir = os.path.dirname(base_path) or "."
    base_name = os.path.basename(base_path)
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith(base_name) and (f.endswith(".zip") or "checkpoint" in f or "emergency" in f)]
    
    if not checkpoints:
        print(f"No checkpoints found for {base_name}")
        return
    
    print(f"Available checkpoints for {base_name}:")
    for i, ckpt in enumerate(sorted(checkpoints)):
        size = os.path.getsize(os.path.join(checkpoint_dir, ckpt)) / (1024*1024)
        print(f"  {i+1}. {ckpt} ({size:.1f} MB)")

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Train or play KungFu Master using PPO")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Run in training mode")
    mode_group.add_argument("--play", action="store_true", help="Run in play mode")
    mode_group.add_argument("--list", action="store_true", help="List available checkpoints")
    
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save/load model")
    parser.add_argument("--cuda", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the environment during training or playing")
    
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--timesteps", type=int, default=40_000, help="Number of training timesteps")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--save_freq", type=int, default=10000, help="Frequency for saving checkpoints")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar during training")
    
    parser.add_argument("--learning_rate", type=float, default=2.5e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    
    parser.add_argument("--log_dir", default="logs", help="Directory for TensorBoard logs")
    
    args = parser.parse_args()
    
    if not any([args.train, args.play, args.list]):
        args.train = True
    
    try:
        if args.list:
            list_checkpoints(args)
        elif args.play:
            play(args)
        else:
            train(args)
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        sys.exit(1)