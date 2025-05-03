from collections import deque
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
import logging
import signal
import atexit
import traceback
import time
import gc
import sys

# Import threat detection types
from threat_detection import AgentActionType, ThreatType, ThreatDirection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="kungfu_training.log",
)
logger = logging.getLogger("kungfu_train")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# Import custom environment
try:
    from kung_fu_env import (
        make_kungfu_env,
        MODEL_PATH,
        RetroEnvManager,
        ENV_CONFIG,
    )
except ImportError as e:
    logger.error(f"Failed to import kung_fu_env: {e}")
    raise

# Global environment registry for cleanup on exit
active_environments = set()


def cleanup_all_resources():
    """Clean up all resources when program exits"""
    global active_environments

    logger.info("Beginning cleanup of all resources...")

    # Close all active environments
    for env in list(active_environments):
        try:
            if env is not None and hasattr(env, "close"):
                logger.info(f"Closing environment: {type(env)}")
                env.close()
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    active_environments.clear()

    # Use RetroEnvManager to clean up any remaining environments
    try:
        RetroEnvManager.get_instance().cleanup_all_envs()
    except Exception as e:
        logger.error(f"Error in RetroEnvManager cleanup: {e}")

    # Force garbage collection
    gc.collect()

    logger.info("Cleanup complete")


# Register cleanup on normal exit
atexit.register(cleanup_all_resources)


# Register cleanup on SIGINT (Ctrl+C) and SIGTERM
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, cleaning up and exiting...")
    cleanup_all_resources()
    exit(1)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Import gym to avoid issues
try:
    import gymnasium as gym
except ImportError:
    import gym


# New: Improved frame stacking with asymmetric temporal sampling
# Fixed AsymmetricFrameStack class for train.py
class AsymmetricFrameStack(gym.Wrapper):
    def __init__(self, env, stack_indices=[0, 1, 4, 9]):
        """
        Stack frames with asymmetric temporal sampling for better pattern recognition

        Args:
            env: The environment to wrap
            stack_indices: Indices of frames to stack, e.g. [0, 1, 4, 9] means
                          current frame, previous frame, 4 frames ago, and 9 frames ago
        """
        super().__init__(env)

        self.stack_indices = stack_indices
        self.max_history = max(stack_indices) + 1
        self.frames = deque(maxlen=self.max_history)

        # Determine frame shape by inspecting env observation space
        if len(env.observation_space.shape) == 3:  # (C, H, W)
            self.frame_shape = env.observation_space.shape
            self.is_channel_first = True
        else:
            # Default assumption for unknown shapes
            self.frame_shape = (1, 84, 84)
            self.is_channel_first = True

        logger.info(f"Frame shape in AsymmetricFrameStack: {self.frame_shape}")

        # Update observation space for stacked frames
        stack_shape = (len(stack_indices) * self.frame_shape[0],) + self.frame_shape[1:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=stack_shape, dtype=np.uint8
        )

        logger.info(f"Created AsymmetricFrameStack with indices {stack_indices}")
        logger.info(f"New observation space shape: {self.observation_space.shape}")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Ensure observation has correct shape
        observation = self._preprocess_observation(observation)

        # Fill frame buffer with copies of initial observation
        for _ in range(self.max_history):
            self.frames.append(observation.copy())

        # Create stacked observation
        stacked_obs = self._get_stacked_obs()

        return stacked_obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Ensure observation has correct shape
        observation = self._preprocess_observation(observation)

        # Add new observation to frame buffer
        self.frames.append(observation)

        # Create stacked observation
        stacked_obs = self._get_stacked_obs()

        return stacked_obs, reward, terminated, truncated, info

    def _preprocess_observation(self, observation):
        """Ensure observation has the expected shape"""
        # Handle different input shapes
        if len(observation.shape) == 4 and observation.shape[0] == 1:
            # (1, C, H, W) -> (C, H, W)
            return observation.squeeze(0)
        elif len(observation.shape) == 3 and self.is_channel_first:
            # Already (C, H, W)
            return observation
        elif len(observation.shape) == 3 and not self.is_channel_first:
            # (H, W, C) -> (C, H, W)
            return np.transpose(observation, (2, 0, 1))
        elif len(observation.shape) == 2:
            # (H, W) -> (1, H, W)
            return observation[np.newaxis, ...]

        # If we reach here, we have an unexpected shape
        logger.warning(f"Unexpected observation shape: {observation.shape}")
        return observation

    def _get_stacked_obs(self):
        """Stack selected frames based on stack_indices"""
        if len(self.frames) == 0:
            # Handle empty buffer case
            logger.warning("Empty frame buffer in _get_stacked_obs")
            empty_frame = np.zeros(self.frame_shape, dtype=np.uint8)
            return np.tile(empty_frame, (len(self.stack_indices), 1, 1))

        # Stack selected frames based on stack_indices
        frames_list = []
        for idx in self.stack_indices:
            if idx < len(self.frames):
                frames_list.append(self.frames[-(idx + 1)])
            else:
                # If we don't have enough frames yet, use the oldest one we have
                frames_list.append(self.frames[0])

        # Concatenate along the channel dimension
        if self.is_channel_first:
            return np.concatenate(frames_list, axis=0)
        else:
            return np.concatenate(frames_list, axis=2)


# Modified KungFuCNNExtractor for train.py
class KungFuCNNExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        super().__init__()

        # Store features_dim as an instance attribute
        self.features_dim = features_dim

        # Get the shape of the input observation
        n_input_channels = observation_space.shape[0]

        # Implement spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

        # CNN for image processing with residual connections
        self.conv1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0), nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0), nn.ReLU()
        )

        # Additional layer for better feature extraction
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0), nn.ReLU()
        )

        self.flatten = nn.Flatten()

        # Dynamically determine output size with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + observation_space.shape)
            attention_map = self.spatial_attention(dummy_input)
            attended_input = dummy_input * attention_map

            x = self.conv1(attended_input)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.flatten(x)
            cnn_output_shape = x.size(1)

        # Improved feature extraction with layered network
        self.fc1 = nn.Linear(cnn_output_shape, 256)
        self.fc2 = nn.Linear(256, self.features_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

    def forward(self, observations):
        # Normalize the observations
        x = observations.float() / 255.0

        # Apply spatial attention
        attention_map = self.spatial_attention(x)
        x = x * attention_map

        # Feature extraction with improved network
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)

        # Multi-layer feature projection with dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        return x


# Custom policy with dynamic entropy scheduling and enhanced threat awareness
# Modified KungFuPolicy class for train.py
class KungFuPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        total_timesteps=1000000,
        *args,
        **kwargs,
    ):
        # Store total timesteps for entropy scheduling
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.initial_entropy_coef = 0.01

        # Extract features_extractor_class from kwargs if present
        features_extractor_class = kwargs.pop("features_extractor_class", None)

        # Use custom feature extractor
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=features_extractor_class or KungFuCNNExtractor,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        # Override to use our custom CNN extractor
        return super()._build_mlp_extractor()

    def forward(self, obs, deterministic=False):
        """
        Override forward to implement dynamic entropy scaling
        """
        # Call parent implementation for normal forward pass
        actions, values, log_probs = super().forward(obs, deterministic)

        # Update step counter
        self.current_step += 1

        return actions, values, log_probs

    def get_entropy_coef(self):
        """
        Calculate dynamic entropy coefficient based on training progress
        """
        # Calculate progress ratio (0 to 1)
        progress = min(1.0, self.current_step / self.total_timesteps)

        # High entropy early on, gradually decreasing over time
        # Schedule: initial value * (0.95 ^ (progress * 10))
        entropy_coef = self.initial_entropy_coef * (0.95 ** (progress * 10))

        return entropy_coef


# Enhanced training callback with improved metrics tracking
class TrainingCallback(BaseCallback):
    def __init__(
        self,
        log_freq=1000,
        log_dir="logs",
        model_path=MODEL_PATH,
        save_freq=50000,
        eval_freq=100000,
        total_timesteps=1000000,
    ):
        super().__init__()
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.eval_freq = eval_freq
        self.best_mean_reward = -float("inf")
        self.log_dir = log_dir
        self.model_path = model_path
        self.total_timesteps = total_timesteps
        self.stage_progress = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.stage_completion_time = {}

        # Track reward components for analysis
        self.reward_components = {
            "score": [],
            "progress": [],
            "combat": [],
            "defensive": [],
            "damage": [],
        }

        # Track stage-specific metrics
        self.stage_metrics = {
            stage: {
                "visits": 0,
                "max_progress": 0,
                "completed": False,
                "avg_reward": 0,
                "success_rate": 0,
            }
            for stage in range(1, 6)
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "stage_analysis"), exist_ok=True)

        # Create metrics header with enhanced tracking
        metrics_header = (
            "timestamp,steps,mean_reward,mean_score,mean_damage,mean_progress,max_stage,"
            "combat_engagement,threats_detected,recommended_actions_taken,"
            "successful_dodges,strategic_retreats,training_progress,entropy_coef"
        )

        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(metrics_header + "\n")

        # Create stage progress tracking file
        self.stage_progress_file = os.path.join(log_dir, "stage_progress.csv")
        stage_progress_header = (
            "timestamp,steps,stage_1,stage_2,stage_3,stage_4,stage_5"
        )

        if not os.path.exists(self.stage_progress_file):
            with open(self.stage_progress_file, "w") as f:
                f.write(stage_progress_header + "\n")

    def _on_step(self):
        # Update training progress
        progress = self.n_calls / self.total_timesteps

        # Log metrics at regular intervals
        if self.n_calls % self.log_freq == 0:
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            mean_score = 0
            mean_damage = 0
            mean_progress = 0
            max_stage = 0
            combat_engagement = 0
            threats_detected = 0
            recommended_actions_taken = 0
            successful_dodges = 0
            strategic_retreats = 0

            if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
                valid_episodes = [
                    info for info in self.model.ep_info_buffer if isinstance(info, dict)
                ]
                if valid_episodes:
                    scores = [
                        ep_info.get("score_increase", 0) for ep_info in valid_episodes
                    ]
                    damages = [
                        ep_info.get("damage_taken", 0) for ep_info in valid_episodes
                    ]
                    progress = [
                        ep_info.get("progress_made", 0) for ep_info in valid_episodes
                    ]
                    stages = [
                        ep_info.get("current_stage", 0) for ep_info in valid_episodes
                    ]

                    # Track stage visits and progress
                    for stage in stages:
                        if 1 <= stage <= 5:
                            self.stage_metrics[stage]["visits"] += 1

                    # Update stage progress tracking
                    max_stage_seen = max(stages) if stages else 0
                    if max_stage_seen > 0:
                        for stage in range(1, max_stage_seen + 1):
                            self.stage_progress[stage] = 1.0

                        # Estimate progress in current highest stage
                        if max_stage_seen <= 5:
                            progress_in_stage = max(progress) / 1000 if progress else 0
                            self.stage_progress[max_stage_seen] = min(
                                progress_in_stage, 1.0
                            )

                    # Track combat metrics
                    combat_rewards = [
                        ep_info.get("combat_engagement_reward", 0)
                        for ep_info in valid_episodes
                    ]
                    combat_engagement = np.mean(combat_rewards) if combat_rewards else 0

                    # Track threat metrics
                    threats = [
                        1 if ep_info.get("threat_detected", False) else 0
                        for ep_info in valid_episodes
                    ]
                    actions_taken = [
                        1 if ep_info.get("recommended_action_taken", False) else 0
                        for ep_info in valid_episodes
                    ]

                    # Track defensive metrics
                    dodges = [
                        ep_info.get("successful_dodges", 0)
                        for ep_info in valid_episodes
                    ]
                    retreats = [
                        ep_info.get("strategic_retreat", 0)
                        for ep_info in valid_episodes
                    ]

                    threats_detected = np.mean(threats) if threats else 0
                    recommended_actions_taken = (
                        np.mean(actions_taken) if actions_taken else 0
                    )
                    successful_dodges = np.mean(dodges) if dodges else 0
                    strategic_retreats = np.mean(retreats) if retreats else 0

                    mean_score = np.mean(scores) if scores else 0
                    mean_damage = np.mean(damages) if damages else 0
                    mean_progress = np.mean(progress) if progress else 0
                    max_stage = np.max(stages) if stages else 0

                    # Store metrics in reward components for analysis
                    self.reward_components["score"].append(mean_score)
                    self.reward_components["combat"].append(combat_engagement)
                    self.reward_components["damage"].append(
                        -mean_damage
                    )  # Negative because it's a penalty
                    self.reward_components["progress"].append(mean_progress)
                    self.reward_components["defensive"].append(successful_dodges)

                    # Truncate reward component lists if they get too long
                    max_history = 100
                    for key in self.reward_components:
                        if len(self.reward_components[key]) > max_history:
                            self.reward_components[key] = self.reward_components[key][
                                -max_history:
                            ]

            # Get current entropy coefficient if available
            entropy_coef = 0.01  # Default
            if hasattr(self.model.policy, "get_entropy_coef"):
                entropy_coef = self.model.policy.get_entropy_coef()

            # Build log message
            log_msg = (
                f"Step: {self.n_calls}/{self.total_timesteps} ({progress:.1%}), "
                f"Mean reward: {mean_reward:.2f}, "
                f"Mean score: {mean_score:.1f}, Mean damage: {mean_damage:.1f}, "
                f"Mean progress: {mean_progress:.1f}, Max stage: {max_stage}, "
                f"Combat engagement: {combat_engagement:.2f}, "
                f"Entropy: {entropy_coef:.4f}, "
                f"Stage progress: {[f'{s}:{p:.1%}' for s, p in self.stage_progress.items()]}"
            )

            logger.info(log_msg)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Build metrics data string
                metrics = (
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{mean_score:.1f},"
                    f"{mean_damage:.1f},{mean_progress:.1f},{max_stage},{combat_engagement:.2f},"
                    f"{threats_detected:.2f},{recommended_actions_taken:.2f},"
                    f"{successful_dodges:.2f},{strategic_retreats:.2f},{progress:.3f},{entropy_coef:.4f}"
                )

                with open(self.metrics_file, "a") as f:
                    f.write(metrics + "\n")

                # Log stage progress
                stage_progress_metrics = (
                    f"{timestamp},{self.n_calls},"
                    f"{self.stage_progress[1]:.2f},{self.stage_progress[2]:.2f},"
                    f"{self.stage_progress[3]:.2f},{self.stage_progress[4]:.2f},"
                    f"{self.stage_progress[5]:.2f}"
                )

                with open(self.stage_progress_file, "a") as f:
                    f.write(stage_progress_metrics + "\n")

                # Every 10 log intervals, analyze reward components and save to CSV
                if self.n_calls % (self.log_freq * 10) == 0:
                    self._analyze_reward_components()

            except Exception as e:
                logger.error(f"Error writing to metrics file: {e}")

            # Save checkpoint at regular intervals
            if self.n_calls % self.save_freq == 0:
                try:
                    checkpoint_path = f"{os.path.dirname(self.model_path)}/kungfu_step_{self.n_calls}.zip"
                    self.model.save(checkpoint_path)
                    logger.info(f"Model checkpoint saved at step {self.n_calls}")

                    # Save a copy with progress percentage for easy reference
                    progress_pct = int(progress * 100)
                    progress_path = f"{os.path.dirname(self.model_path)}/kungfu_{progress_pct}pct.zip"
                    import shutil

                    shutil.copy(checkpoint_path, progress_path)

                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")

            # Save if best model so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                try:
                    self.model.save(self.model_path)
                    logger.info(f"New best model saved with reward {mean_reward:.2f}")
                except Exception as e:
                    logger.error(f"Error saving best model: {e}")

        # Evaluate agent performance on specific stages periodically
        if self.n_calls % self.eval_freq == 0 and self.n_calls > 0:
            self._evaluate_stage_performance()

        return True

    def _analyze_reward_components(self):
        """Analyze and save reward component contributions"""
        try:
            # Calculate average contribution of each component
            component_avgs = {}
            for key, values in self.reward_components.items():
                if values:
                    component_avgs[key] = np.mean(values)
                else:
                    component_avgs[key] = 0

            # Calculate relative contributions (percentage)
            total = sum(
                max(0.01, v) for v in component_avgs.values()
            )  # avoid division by zero
            component_pcts = {k: (v / total) * 100 for k, v in component_avgs.items()}

            # Save to CSV
            analysis_file = os.path.join(self.log_dir, "reward_analysis.csv")

            # Create header if file doesn't exist
            if not os.path.exists(analysis_file):
                with open(analysis_file, "w") as f:
                    header = (
                        "timestamp,steps,"
                        + ",".join(self.reward_components.keys())
                        + ","
                        + ",".join([f"{k}_pct" for k in self.reward_components.keys()])
                    )
                    f.write(header + "\n")

            # Append new analysis
            with open(analysis_file, "a") as f:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                row = f"{timestamp},{self.n_calls},"
                row += ",".join(
                    [
                        f"{component_avgs.get(k, 0):.2f}"
                        for k in self.reward_components.keys()
                    ]
                )
                row += ","
                row += ",".join(
                    [
                        f"{component_pcts.get(k, 0):.1f}"
                        for k in self.reward_components.keys()
                    ]
                )
                f.write(row + "\n")

            logger.info(f"Reward component analysis saved at step {self.n_calls}")

        except Exception as e:
            logger.error(f"Error in reward component analysis: {e}")

    def _evaluate_stage_performance(self):
        """Evaluate agent performance on specific stages"""
        logger.info(f"Stage performance evaluation at step {self.n_calls}")

        # Log current stage metrics
        stage_metrics_file = os.path.join(
            self.log_dir, "stage_analysis", f"stage_metrics_{self.n_calls}.csv"
        )

        try:
            with open(stage_metrics_file, "w") as f:
                f.write("stage,visits,max_progress,completed,avg_reward,success_rate\n")
                for stage, metrics in self.stage_metrics.items():
                    f.write(
                        f"{stage},{metrics['visits']},{metrics['max_progress']:.2f},"
                        f"{metrics['completed']},{metrics['avg_reward']:.2f},"
                        f"{metrics['success_rate']:.2f}\n"
                    )
        except Exception as e:
            logger.error(f"Error saving stage metrics: {e}")


def create_model(env, resume=False, model_path=MODEL_PATH, total_timesteps=1000000):
    if resume and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        try:
            # Create a new model with the current environment
            model = PPO(
                "CnnPolicy",  # Use built-in CnnPolicy for now
                env=env,
                learning_rate=schedule_lr,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                tensorboard_log="./logs/tensorboard/",
                verbose=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    normalize_images=False,
                    optimizer_class=torch.optim.Adam,
                    optimizer_kwargs=dict(eps=1e-5),
                ),
            )

            # Load old model and try to copy weights
            try:
                old_model = PPO.load(model_path)
                # Copy policy parameters where shapes match
                for name, param in model.policy.named_parameters():
                    if name in dict(old_model.policy.named_parameters()):
                        old_param = dict(old_model.policy.named_parameters())[name]
                        if param.shape == old_param.shape:
                            param.data.copy_(old_param.data)
                            logger.info(f"Copied weights for {name}")
                        else:
                            logger.warning(
                                f"Shape mismatch for {name}: {param.shape} vs {old_param.shape}"
                            )
            except Exception as e:
                logger.error(f"Error loading/transferring weights: {e}")

            logger.info("Model initialized with new observation space")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating new model instead")

    logger.info("Creating new model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create model directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Use the built-in CnnPolicy for simplicity
    model = PPO(
        "CnnPolicy",
        env=env,
        learning_rate=schedule_lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="./logs/tensorboard/",
        verbose=1,
        device=device,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            normalize_images=False,
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
        ),
    )

    return model


def schedule_lr(progress):
    """
    Dynamic learning rate scheduling

    Starts with higher learning rate and gradually decreases
    with a small bump in the middle to escape local minima
    """
    initial_lr = 3e-4

    # If in first 10% of training - use higher learning rate
    if progress < 0.1:
        return initial_lr
    # If in 40-60% of training - slight bump to escape local minima
    elif 0.4 <= progress <= 0.6:
        return initial_lr * 0.7
    # Gradual decrease after 60%
    elif progress > 0.6:
        return initial_lr * 0.5 * (1 - min(0.8, (progress - 0.6) / 0.4))
    # Normal rate in between
    else:
        return initial_lr * 0.5


def train_model(model, timesteps):
    callback = TrainingCallback(
        log_freq=1000, save_freq=50000, eval_freq=100000, total_timesteps=timesteps
    )

    try:
        logger.info(f"Starting training for {timesteps} timesteps")
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            tb_log_name="kungfu_training",
            progress_bar=True,
        )
        logger.info("Training completed successfully")
        model.save(MODEL_PATH)
        logger.info(f"Final model saved to {MODEL_PATH}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        model.save(f"{os.path.dirname(MODEL_PATH)}/kungfu_interrupted.zip")
        logger.info("Interrupted model saved")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())


def main():
    parser = argparse.ArgumentParser(
        description="Train a PPO agent for Kung Fu Master with improved reward and exploration"
    )
    parser.add_argument(
        "--timesteps", type=int, default=5000000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to save or load model (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--combat-weight",
        type=float,
        default=1.8,
        help="Initial weight for combat engagement rewards (default: 1.8)",
    )
    parser.add_argument(
        "--progression-weight",
        type=float,
        default=1.5,
        help="Initial weight for progression rewards (default: 1.5)",
    )
    parser.add_argument(
        "--defensive-bonus",
        type=float,
        default=0.5,
        help="Weight for defensive positioning bonus (default: 0.5)",
    )
    parser.add_argument(
        "--strategic-retreat-bonus",
        type=float,
        default=0.3,
        help="Weight for strategic retreat bonus (default: 0.3)",
    )
    parser.add_argument(
        "--stack-indices",
        nargs="+",
        type=int,
        default=[0, 1, 4, 9],
        help="Frame indices to stack for temporal patterns (default: 0 1 4 9)",
    )

    args = parser.parse_args()

    # Override model path if specified
    model_path = args.model_path
    if model_path != MODEL_PATH:
        logger.info(f"Using custom model path: {model_path}")

    # Set environment config based on arguments
    ENV_CONFIG["combat_engagement_weight"] = args.combat_weight
    ENV_CONFIG["progression_weight"] = args.progression_weight
    ENV_CONFIG["defensive_bonus"] = args.defensive_bonus
    ENV_CONFIG["strategic_retreat_bonus"] = args.strategic_retreat_bonus

    # Log configuration
    logger.info("Environment configuration:")
    logger.info(f"  combat_weight: {ENV_CONFIG['combat_engagement_weight']}")
    logger.info(f"  progression_weight: {ENV_CONFIG['progression_weight']}")
    logger.info(f"  defensive_bonus: {ENV_CONFIG['defensive_bonus']}")
    logger.info(f"  strategic_retreat_bonus: {ENV_CONFIG['strategic_retreat_bonus']}")
    logger.info(f"  frame_stack_indices: {args.stack_indices}")

    try:
        logger.info("Creating Kung Fu Master environment with enhanced features...")
        gc.collect()
        time.sleep(1)

        # Create base environment
        env = make_kungfu_env(
            is_play_mode=args.render,
            frame_stack=1,  # Set to 1 since we'll use our custom frame stacking
            use_dfp=False,
        )

        # Apply custom asymmetric frame stacking
        env = AsymmetricFrameStack(env, stack_indices=args.stack_indices)
        active_environments.add(env)

        logger.info("Creating model...")
        model = create_model(
            env,
            resume=args.resume,
            model_path=model_path,
            total_timesteps=args.timesteps,
        )

        train_model(model, args.timesteps)

        logger.info("Closing training environment")
        env.close()
        active_environments.discard(env)

        logger.info("Training completed")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup_all_resources()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main program: {e}")
        logger.error(traceback.format_exc())
    finally:
        cleanup_all_resources()
