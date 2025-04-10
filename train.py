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
from kungfu_env import make_env, SimpleCNN  # Import SimpleCNN too

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
        total_hits = sum([info.get('enemy_hit', 0) for info in infos])
        total_hp = sum([info.get('hp', 0) for info in infos])
        score = total_hits * 10 + total_hp / 255.0
        
        if score > self.best_score:
            self.best_score = score
            self.model.save(self.save_path)
            if self.verbose > 0:
                print(f"Saved best model with score {self.best_score} at step {self.num_timesteps}")
        return True

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger()

def train(args):
    global current_model, global_logger, global_model_path, experience_data
    experience_data = []
    
    global_logger = setup_logging(args.log_dir)
    global_logger.info(f"Starting training with {args.num_envs} envs and {args.timesteps} timesteps")
    global_model_path = args.model_path
    current_model = None

    if args.num_envs < 1:
        raise ValueError("Number of environments must be at least 1")
    
    env_fns = [make_env for _ in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=12)
    
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,  # Use shared SimpleCNN
        "net_arch": dict(pi=[256, 256, 128], vf=[512, 512, 256])
    }
    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)

    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": 0.1,
        "n_steps": min(2048, args.timesteps // args.num_envs if args.num_envs > 0 else args.timesteps),
        "batch_size": 64,
        "n_epochs": 10
    }

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

    save_callback = SaveBestModelCallback(save_path=args.model_path)
    exp_callback = ExperienceCollectionCallback()
    callback = CallbackList([save_callback, exp_callback])
    
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=args.progress_bar)
    experience_data = exp_callback.experience_data
    
    experience_count = len(experience_data)
    global_logger.info(f"Training completed. Total experience collected: {experience_count} steps")
    
    with open(f"{args.model_path}_experience_count.txt", "w") as f:
        f.write(f"Total experience collected: {experience_count} steps\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training parameters: num_envs={args.num_envs}, timesteps={args.timesteps}\n")

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
    
    args = parser.parse_args()
    train(args)