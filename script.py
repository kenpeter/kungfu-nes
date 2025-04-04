import argparse
import os
import retro
import numpy as np
import torch
import torch.nn as nn
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces, Wrapper
import optuna
import logging

class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super(SimpleCNN, self).__init__(observation_space, features_dim)
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict"
        
        # CNN expects 4 input channels (from VecFrameStack n_stack=4)
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute flattened size
        with torch.no_grad():
            # Use the actual observation space dimensions for calculation
            sample_input = torch.zeros(1, 4, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        # Debug print to see the actual flattened size
        #print(f"CNN flattened output size: {n_flatten}")
        
        # Check enemy vector size by examining observation space
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        # When using VecFrameStack with n_stack=4, each frame's enemy vector (4 values) gets stacked
        # resulting in 16 values total
        
        #print(f"Enemy vector size from observation space: {enemy_vec_size}")
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + enemy_vec_size, features_dim),  # Dynamically set input size
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        
        # Debug the input shapes
        #print(f"Viewport shape before processing: {viewport.shape}")
        #print(f"Enemy vector shape before processing: {enemy_vector.shape}")
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
            
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
            
        # Handle single and batched observations
        if len(viewport.shape) == 3:  # Single observation
            # Add batch dimension
            viewport = viewport.unsqueeze(0)
            
        if len(viewport.shape) == 4:  # Batched observations
            # Check if we need to rearrange dimensions
            if viewport.shape[3] == 4 or viewport.shape[3] == 1:  # Shape is [batch, H, W, C]
                viewport = viewport.permute(0, 3, 1, 2)  # Convert to [batch, C, H, W]
        
        # Final check on dimensions
        #print(f"Viewport shape after processing: {viewport.shape}")
        if viewport.shape[1] != 4:
            raise ValueError(f"Expected 4 channels in viewport, got shape {viewport.shape}")
            
        # Process with CNN
        cnn_output = self.cnn(viewport)
        
        # Handle enemy vector dimensions
        if len(enemy_vector.shape) == 1:  # Single observation
            enemy_vector = enemy_vector.unsqueeze(0)
            
        # Combine features
        #print(f"CNN output shape: {cnn_output.shape}, Enemy vector shape: {enemy_vector.shape}")
        combined = torch.cat([cnn_output, enemy_vector], dim=1)
        return self.linear(combined)
        
class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (4,), np.float32)
        })
        self.actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,1,0,0,0,0,0], [0,0,0,0,0,0,0,0,1,0,0,0],
            [1,0,0,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0,0,0], [1,0,0,0,0,0,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,0,1,0,0,0], [0,1,0,0,0,0,1,0,0,0,0,0], [0,1,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0,0,0,0,0]
        ]
        self.projectiles = []
        self.last_hp = 0
        self.action_counts = np.zeros(11)
        self.last_frame = None
        self.success_dodges = 0
        self.failed_dodges = 0

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.action_counts = np.zeros(11)
        self.projectiles = []
        self.last_frame = None
        self.success_dodges = 0
        self.failed_dodges = 0
        return self._get_obs(obs)

    def step(self, action):
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        score = sum(ram[0x0531:0x0536]) * 100
        scroll = ram[0x00E5]
        hp = ram[0x04A6]
        pos_x = ram[0x0094]
        
        prev_projectiles = self.projectiles.copy()
        self._update_projectiles(obs)
        
        hp_loss = max(0, self.last_hp - hp)
        dodge_reward = self._calculate_dodge_reward(prev_projectiles, pos_x, hp_loss)
        reward = (score * 0.1 + scroll * 0.5 - hp_loss * 10 + dodge_reward)
        self.last_hp = hp
        
        info.update({
            "score": score, "hp": hp, "pos_x": pos_x,
            "action_percentages": self.action_counts / (sum(self.action_counts) + 1e-6),
            "projectiles": len(self.projectiles),
            "success_dodges": self.success_dodges,
            "failed_dodges": self.failed_dodges
        })
        return self._get_obs(obs), reward, done, info

    def _get_obs(self, obs):
        # Convert RGB image to grayscale and resize
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        viewport = cv2.resize(gray, self.viewport_size)[..., np.newaxis]
        
        # Get enemy positions from RAM
        ram = self.env.get_ram()
        hero_x = int(ram[0x0094])
        enemy_xs = [
            int(ram[0x008E]),  # Enemy 1 Pos X
            int(ram[0x008F]),  # Enemy 2 Pos X
            int(ram[0x0090]),  # Enemy 3 Pos X
            int(ram[0x0091])   # Enemy 4 Pos X
        ]
        
        # Calculate relative positions (-ve: enemy to left, +ve: enemy to right)
        enemy_vector = np.array([e - hero_x if e != 0 else 0 for e in enemy_xs], dtype=np.float32)
        
        return {
            "viewport": viewport.astype(np.uint8),  # Shape: (84, 84, 1)
            "enemy_vector": enemy_vector            # Shape: (4,)
        }

    def _update_projectiles(self, obs):
        if self.last_frame is None:
            self.last_frame = obs
            self.projectiles = []
            return

        curr_gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        last_gray = cv2.cvtColor(self.last_frame, cv2.COLOR_RGB2GRAY)
        
        frame_diff = cv2.absdiff(curr_gray, last_gray)
        thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)[1]
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        new_projectiles = []
        for contour in contours:
            if 10 < cv2.contourArea(contour) < 100:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                
                speed_x = 0
                for old_x, old_y, old_speed in self.projectiles:
                    if abs(old_y - cy) < 10:
                        speed_x = cx - old_x
                        break
                
                new_projectiles.append((cx, cy, speed_x))
        
        self.projectiles = new_projectiles
        self.last_frame = obs.copy()

    def _calculate_dodge_reward(self, prev_projectiles, hero_x, hp_loss):
        dodge_reward = 0
        for px, py, speed in prev_projectiles:
            if abs(px - hero_x) < 20 and speed != 0:
                new_dist = abs(px + speed - hero_x)
                if hp_loss == 0 and new_dist > 20:
                    dodge_reward += 5
                    self.success_dodges += 1
                elif hp_loss > 0:
                    dodge_reward -= 5
                    self.failed_dodges += 1
        return dodge_reward

def debug_network_dimensions():
    """Debug function to check dimensions of the observation space and CNN output."""
    #print("Debugging network dimensions...")
    
    # Create test environment
    env = make_env()
    obs = env.reset()
    
    # Print observation shapes
    #print(f"Viewport shape: {obs['viewport'].shape}")
    #print(f"Enemy vector shape: {obs['enemy_vector'].shape}")
    
    # Test CNN
    cnn = SimpleCNN(env.observation_space, features_dim=256)
    
    # Convert to torch tensors
    viewport_tensor = torch.from_numpy(obs['viewport']).unsqueeze(0).float()  # Add batch dim
    viewport_tensor = viewport_tensor.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
    enemy_tensor = torch.from_numpy(obs['enemy_vector']).unsqueeze(0).float()  # Add batch dim
    
    # Stack 4 frames as VecFrameStack would do
    viewport_stacked = torch.cat([viewport_tensor] * 4, dim=1)  # Simulate 4 stacked frames
    
    #print(f"Stacked viewport shape: {viewport_stacked.shape}")
    
    # Forward through CNN only
    with torch.no_grad():
        cnn_output = cnn.cnn(viewport_stacked)
        combined = torch.cat([cnn_output, enemy_tensor], dim=1)
        final_output = cnn.linear(combined)
    
    #print(f"CNN output shape: {cnn_output.shape}")
    #print(f"Combined shape: {combined.shape}")
    #print(f"Final output shape: {final_output.shape}")
    
    env.close()
    return

def make_env(render=False):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    if render:
        env.render_mode = 'human'
    return KungFuWrapper(env)

def make_vec_env(num_envs, render=False):
    if num_envs > 1:
        return SubprocVecEnv([lambda: make_env(render) for _ in range(num_envs)])
    return DummyVecEnv([lambda: make_env(render)])

class TrainingCallback(BaseCallback):
    def __init__(self, progress_bar=False, logger=None):
        super().__init__()
        self.projectile_logs = []
        self.total_reward = 0
        self.progress_bar = progress_bar
        self.logger = logger
        if progress_bar:
            from tqdm import tqdm
            self.pbar = tqdm(total=10000)  # Default, updated in train

    def _on_step(self):
        env = self.training_env.envs[0]
        ram = self.training_env.envs[0].env.get_ram()
        info = self.locals["infos"][0]
        
        self.projectile_logs.append({
            "agent_x": ram[0x0094],
            "agent_y": ram[0x00B6],
            "hp_loss": max(0, env.last_hp - ram[0x04A6]),
            "action_percentages": info["action_percentages"],
            "projectiles": [(x, y, speed) for x, y, speed in env.projectiles]
        })
        
        self.total_reward += self.locals["rewards"][0]
        
        if self.logger:
            self.logger.info(f"Step: {self.num_timesteps}, Reward: {self.locals['rewards'][0]}, Score: {info['score']}")
        
        if self.logger:
            self.logger.record("rollout/ep_rew_mean", np.mean(self.locals["rewards"]))
            self.logger.record("metrics/score", info["score"])
            self.logger.record("metrics/hp", info["hp"])
            self.logger.record("metrics/projectiles", info["projectiles"])
            self.logger.record("metrics/success_dodges", info["success_dodges"])
            self.logger.record("metrics/failed_dodges", info["failed_dodges"])
        
        if self.progress_bar:
            self.pbar.update(1)
        return True

    def _on_training_end(self):
        if self.progress_bar:
            self.pbar.close()

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    n_steps = trial.suggest_int("n_steps", 2048, 16384, step=2048)
    batch_size = trial.suggest_int("batch_size", 64, 512, step=64)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)

    env = make_vec_env(1)  # Use single env for hyperparameter tuning for simplicity
    env = VecFrameStack(env, n_stack=4)
    
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs={"features_extractor_class": SimpleCNN},
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        clip_range=clip_range,
        tensorboard_log=args.log_dir,
        device="cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    )

    callback = TrainingCallback(progress_bar=args.progress_bar)
    try:
        model.learn(
            total_timesteps=args.eval_timesteps,
            callback=callback,
            tb_log_name=f"PPO_KungFu_trial_{trial.number}"
        )
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('-inf')  # Return a bad score if the trial fails
    
    return callback.total_reward / args.eval_timesteps

def train(args):
    # Run the debug function to check dimensions
    debug_network_dimensions()
    
    if args.enable_file_logging:
        logging.basicConfig(
            filename=os.path.join(args.log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
    else:
        logger = None

    # For resuming, we'll use the best hyperparameters we found previously or defaults
    best_params = {
        "learning_rate": 0.0001,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "clip_range": 0.2
    }
    
    env = make_vec_env(args.num_envs, render=args.render)
    env = VecFrameStack(env, n_stack=4)
    
    model_path = args.model_path if os.path.isdir(args.model_path) else os.path.dirname(args.model_path)
    model_file = os.path.join(model_path, "kungfu_ppo_optuna")
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.resume and os.path.exists(model_file + ".zip"):
        model = PPO.load(model_file, env=env, device=device)
        print(f"Resumed from {model_file}.zip")
    else:
        if not args.skip_optuna:
            print("Running Optuna optimization...")
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
            
            print("Best hyperparameters:", study.best_params)
            print("Best value:", study.best_value)
            best_params = study.best_params
        
        model = PPO(
            policy="MultiInputPolicy",
            env=env,
            policy_kwargs={"features_extractor_class": SimpleCNN},
            learning_rate=best_params["learning_rate"],
            n_steps=best_params["n_steps"],
            batch_size=best_params["batch_size"],
            n_epochs=best_params["n_epochs"],
            gamma=best_params["gamma"],
            clip_range=best_params["clip_range"],
            tensorboard_log=args.log_dir,
            device=device
        )

    callback = TrainingCallback(progress_bar=args.progress_bar, logger=logger)
    if args.progress_bar:
        callback.pbar.total = args.timesteps
    
    model.learn(
        total_timesteps=args.timesteps,
        callback=callback,
        tb_log_name="PPO_KungFu_Best",
        reset_num_timesteps=not args.resume
    )
    
    # Make sure the model directory exists
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    model.save(model_file)
    print(f"Model saved to {model_file}.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_path", default="models/kungfu_ppo_optuna")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--eval_timesteps", type=int, default=50000)
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--enable_file_logging", action="store_true", help="Enable file logging")
    parser.add_argument("--progress_bar", action="store_true", help="Show progress bar")
    parser.add_argument("--resume", action="store_true", help="Resume training from saved model")
    parser.add_argument("--skip_optuna", action="store_true", help="Skip Optuna hyperparameter optimization")
    
    args = parser.parse_args()
    if args.train:
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        train(args)