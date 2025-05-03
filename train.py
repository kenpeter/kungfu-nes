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
from collections import deque

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


# Standard frame stacking - consecutive frames instead of asymmetric
class StandardFrameStack(gym.Wrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        # Determine frame shape by inspecting env observation space
        if len(env.observation_space.shape) == 3:  # (C, H, W)
            self.frame_shape = env.observation_space.shape
            self.is_channel_first = True
        else:
            # Default assumption for unknown shapes
            self.frame_shape = (1, 84, 84)
            self.is_channel_first = True

        logger.info(f"Frame shape in StandardFrameStack: {self.frame_shape}")

        # Update observation space for stacked frames
        stack_shape = (num_stack * self.frame_shape[0],) + self.frame_shape[1:]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=stack_shape, dtype=np.uint8
        )

        logger.info(f"Created StandardFrameStack with {num_stack} frames")
        logger.info(f"New observation space shape: {self.observation_space.shape}")

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        # Ensure observation has correct shape
        observation = self._preprocess_observation(observation)

        # Fill frame buffer with copies of initial observation
        for _ in range(self.num_stack):
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
        """Stack consecutive frames"""
        if len(self.frames) == 0:
            # Handle empty buffer case
            logger.warning("Empty frame buffer in _get_stacked_obs")
            empty_frame = np.zeros(self.frame_shape, dtype=np.uint8)
            return np.tile(empty_frame, (self.num_stack, 1, 1))

        # Concatenate along the channel dimension
        if self.is_channel_first:
            return np.concatenate(list(self.frames), axis=0)
        else:
            return np.concatenate(list(self.frames), axis=2)


# Simplified CNN feature extractor with minimal architecture
class SimplifiedCNNExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        super().__init__()

        # Store features_dim as an instance attribute
        self.features_dim = features_dim

        # Get the shape of the input observation
        n_input_channels = observation_space.shape[0]

        # Standard CNN for image processing - no spatial attention
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically determine output size with dummy input
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + observation_space.shape)
            x = self.cnn(dummy_input)
            cnn_output_shape = x.size(1)

        # Simple feature extraction - no dropout
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_shape, self.features_dim), nn.ReLU()
        )

    def forward(self, observations):
        # Normalize the observations
        x = observations.float() / 255.0
        x = self.cnn(x)
        x = self.fc(x)
        return x


# Simplified policy with fixed entropy coefficient
class SimplifiedKungFuPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        *args,
        **kwargs,
    ):
        # Use our simplified feature extractor
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=SimplifiedCNNExtractor,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self):
        # Use parent implementation
        return super()._build_mlp_extractor()


# Basic training callback with simplified metrics tracking
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

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # Create metrics header with core metrics
        metrics_header = (
            "timestamp,steps,mean_reward,mean_score,mean_damage,mean_progress,max_stage,"
            "combat_engagement,threats_detected,recommended_actions_taken,training_progress"
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
        # Ensure total_timesteps is an integer
        if isinstance(self.total_timesteps, list):
            total_steps = self.total_timesteps[0] if self.total_timesteps else 1000000
        else:
            total_steps = self.total_timesteps

        progress = self.n_calls / total_steps

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
                    progress_values = [
                        ep_info.get("progress_made", 0) for ep_info in valid_episodes
                    ]
                    stages = [
                        ep_info.get("current_stage", 0) for ep_info in valid_episodes
                    ]

                    # Update stage progress tracking
                    max_stage_seen = max(stages) if stages else 0
                    if max_stage_seen > 0:
                        for stage in range(1, max_stage_seen + 1):
                            self.stage_progress[stage] = 1.0

                        # Estimate progress in current highest stage
                        if max_stage_seen <= 5:
                            progress_in_stage = (
                                max(progress_values) / 1000 if progress_values else 0
                            )
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

                    threats_detected = np.mean(threats) if threats else 0
                    recommended_actions_taken = (
                        np.mean(actions_taken) if actions_taken else 0
                    )

                    mean_score = np.mean(scores) if scores else 0
                    mean_damage = np.mean(damages) if damages else 0
                    mean_progress = np.mean(progress_values) if progress_values else 0
                    max_stage = np.max(stages) if stages else 0

            # Create a string representation of stage progress for logging
            stage_progress_str = ", ".join(
                [f"{s}:{self.stage_progress[s]:.1%}" for s in self.stage_progress]
            )

            # Build log message
            log_msg = (
                f"Step: {self.n_calls}/{total_steps} ({progress:.1%}), "
                f"Mean reward: {mean_reward:.2f}, "
                f"Mean score: {mean_score:.1f}, Mean damage: {mean_damage:.1f}, "
                f"Mean progress: {mean_progress:.1f}, Max stage: {max_stage}, "
                f"Combat engagement: {combat_engagement:.2f}, "
                f"Stage progress: {stage_progress_str}"
            )

            logger.info(log_msg)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Build metrics data string
                metrics = (
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{mean_score:.1f},"
                    f"{mean_damage:.1f},{mean_progress:.1f},{max_stage},{combat_engagement:.2f},"
                    f"{threats_detected:.2f},{recommended_actions_taken:.2f},{progress:.3f}"
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

        return True


def create_model(env, resume=False, model_path=MODEL_PATH, total_timesteps=1000000):
    if resume and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        try:
            # Create a new model with the current environment
            model = PPO(
                "CnnPolicy",  # Use built-in CnnPolicy
                env=env,
                learning_rate=3e-4,  # Fixed learning rate
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # Fixed entropy coefficient
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

            # Load old model weights where possible
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
        learning_rate=3e-4,  # Fixed learning rate
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Fixed entropy coefficient
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
    """Simple learning rate scheduling - constant rate"""
    return 3e-4  # Fixed learning rate


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
        default=1.5,
        help="Initial weight for combat engagement rewards (default: 1.5)",
    )
    parser.add_argument(
        "--progression-weight",
        type=float,
        default=2.0,
        help="Initial weight for progression rewards (default: 2.0)",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of frames to stack (default: 4)",
    )

    args = parser.parse_args()

    # Override model path if specified
    model_path = args.model_path
    if model_path != MODEL_PATH:
        logger.info(f"Using custom model path: {model_path}")

    # Set environment config based on arguments
    ENV_CONFIG["combat_engagement_weight"] = args.combat_weight
    ENV_CONFIG["progression_weight"] = args.progression_weight

    # Log configuration
    logger.info("Environment configuration:")
    logger.info(f"  combat_weight: {ENV_CONFIG['combat_engagement_weight']}")
    logger.info(f"  progression_weight: {ENV_CONFIG['progression_weight']}")
    logger.info(f"  frame_stack: {args.frame_stack}")

    try:
        logger.info("Creating Kung Fu Master environment with enhanced features...")
        gc.collect()
        time.sleep(1)

        # Create base environment with standard frame stacking
        env = make_kungfu_env(
            is_play_mode=args.render,
            frame_stack=args.frame_stack,
            use_dfp=False,
        )
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
