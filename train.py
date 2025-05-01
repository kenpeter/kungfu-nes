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
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import logging
import signal
import atexit
import traceback
import time
import gc
import sys

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


# Simplified DFP feature extractor focused on aggressive combat and progression
class DFPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        # Get image space from observation space
        self.image_space = observation_space.spaces["image"]

        # Initialize counter for logging
        self.n_calls = 0

        # Set frame stack size (based on the shape of the observation space)
        # For channel-first observations, frame_stack is the first dimension
        # For channel-last observations with LazyFrames, we need to determine the frame stack in forward()
        if len(self.image_space.shape) >= 3:
            if self.image_space.shape[0] == 4:  # Channel-first with 4 frames
                self.frame_stack = 4
            else:
                self.frame_stack = 4  # Default to 4 frames
        else:
            self.frame_stack = 4  # Default

        # Number of channels (includes frame stacking)
        # Handle different image formats (channels first or channels last)
        if len(self.image_space.shape) == 3 and self.image_space.shape[-1] == 1:
            # Channel last format: (H, W, C)
            n_input_channels = self.image_space.shape[-1] * 4  # 4-frame stack
            height = self.image_space.shape[0]
            width = self.image_space.shape[1]
        else:
            # Channel first format: (C, H, W)
            n_input_channels = self.image_space.shape[0]
            height = self.image_space.shape[1]
            width = self.image_space.shape[2]

        logger.info(
            f"Image shape: {self.image_space.shape}, Using channels: {n_input_channels}, height: {height}, width: {width}"
        )

        # Get measurement and goal dimensions
        self.measurement_dim = observation_space.spaces["measurements"].shape[0]
        self.goal_dim = observation_space.spaces["goals"].shape[0]

        logger.info(
            f"DFP Feature Extractor - Image shape: {self.image_space.shape}, Measurements: {self.measurement_dim}"
        )

        # ENHANCED: Improved CNN architecture for better perception
        # CNN for image processing with deeper layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(
                64, 32, kernel_size=3, stride=1, padding=0
            ),  # Additional layer for better feature extraction
            nn.ReLU(),
            nn.Flatten(),
        )

        # Determine if we're dealing with stacked frames as channels or as separate dim
        self.channel_last = (
            len(self.image_space.shape) == 3 and self.image_space.shape[-1] == 1
        )

        # Create CNN based on image shape
        if self.channel_last:
            # For LazyGym style: frames are stacked as separate dim, not channels
            # Shape is (frames, height, width, channels)
            self.cnn = nn.Sequential(
                nn.Conv2d(self.frame_stack, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    64, 32, kernel_size=3, stride=1, padding=0
                ),  # Additional layer
                nn.ReLU(),
                nn.Flatten(),
            )
            # Test with dummy input in correct format
            with torch.no_grad():
                # For frame stack with channel last format (B, F, H, W, C)
                dummy_img = torch.zeros((1, self.frame_stack, height, width, 1))
                # Reshape to (B, F, H, W)
                dummy_img = dummy_img.squeeze(-1)
                cnn_out = self.cnn(dummy_img)
                self.cnn_out_size = cnn_out.shape[1]
        else:
            # Traditional format: frames stacked as channels
            # Shape is (channels, height, width)
            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    64, 32, kernel_size=3, stride=1, padding=0
                ),  # Additional layer
                nn.ReLU(),
                nn.Flatten(),
            )
            # Test with dummy input
            with torch.no_grad():
                dummy_img = torch.zeros((1, n_input_channels, height, width))
                cnn_out = self.cnn(dummy_img)
                self.cnn_out_size = cnn_out.shape[1]

        logger.info(
            f"CNN output features: {self.cnn_out_size}, Channel last: {self.channel_last}"
        )

        # ENHANCED: Improved measurement network
        # Simple measurement network with more capacity
        measurement_input_size = self.measurement_dim + self.goal_dim
        self.measurement_net = nn.Sequential(
            nn.Linear(measurement_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # Additional layer for better processing
            nn.ReLU(),
        )
        self.measurement_out_size = 128

        # Combined network
        combined_size = self.cnn_out_size + self.measurement_out_size
        self.combined_net = nn.Sequential(
            nn.Linear(combined_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Additional layer for better integration
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Increment call counter
        self.n_calls += 1

        # Process image input
        image_tensor = observations["image"]

        # Log shape occasionally for debugging
        if self.n_calls % 1000 == 0:
            logger.info(f"Original image tensor shape: {image_tensor.shape}")

        # Handle 5D tensor with shape [batch, frames, height, width, channels]
        if len(image_tensor.shape) == 5:
            batch_size = image_tensor.shape[0]
            frames = image_tensor.shape[1]
            height = image_tensor.shape[2]
            width = image_tensor.shape[3]

            # Remove channel dimension if it's 1
            if image_tensor.shape[4] == 1:
                image_tensor = image_tensor.squeeze(
                    4
                )  # Now [batch, frames, height, width]

                # For channel-first format, permute to [batch, frames, height, width]
                if not self.channel_last:
                    # Handle single batch case separately
                    if batch_size == 1:
                        # [1, frames, height, width] -> use frames as channels
                        image_tensor = image_tensor.squeeze(
                            0
                        )  # [frames, height, width]
                        image_tensor = image_tensor.unsqueeze(
                            0
                        )  # [1, frames, height, width]

        # Handle 4D tensor with shape [batch, height, width, frames] or [batch, frames, height, width]
        elif len(image_tensor.shape) == 4:
            # Check if it's channel-last format [batch, height, width, frames/channels]
            if self.channel_last:
                # If the last dimension is 1, squeeze it
                if image_tensor.shape[3] == 1:
                    image_tensor = image_tensor.squeeze(3)
                    # Add stack dimension if needed
                    image_tensor = image_tensor.unsqueeze(1)
            # If it's already in correct format [batch, frames, height, width], do nothing

        # Normalize image
        image_features = self.cnn(image_tensor.float() / 255.0)

        # Process measurements and goals
        measurements = observations["measurements"]
        goals = observations["goals"]
        # Handle single batch case
        if len(measurements.shape) == 1:
            measurements = measurements.unsqueeze(0)
        if len(goals.shape) == 1:
            goals = goals.unsqueeze(0)

        # Combine measurements and goals
        measurement_tensor = torch.cat([measurements, goals], dim=1)
        measurement_features = self.measurement_net(measurement_tensor)

        # Combine all features
        combined_features = torch.cat([image_features, measurement_features], dim=1)

        return self.combined_net(combined_features)


# Simplified DFP prediction head
class DFPPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim,
        measurement_dims=3,
        prediction_horizons=[1, 3, 5, 10, 20],
        horizon_weights=[1.0, 0.9, 0.8, 0.7, 0.6],
    ):
        super().__init__()
        self.input_dim = input_dim
        self.measurement_dims = measurement_dims
        self.prediction_horizons = prediction_horizons
        # ENHANCED: Updated horizon weights to focus more on medium-term outcomes
        self.horizon_weights = [1.0, 1.1, 1.2, 0.9, 0.7]  # Emphasize 3-5 step horizons
        self.n_horizons = len(prediction_horizons)
        self.output_dim = measurement_dims * self.n_horizons

        # ENHANCED: Improved prediction network
        self.prediction_net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),  # Additional layer for better future prediction
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, features):
        # Generate all future predictions
        predictions = self.prediction_net(features)
        return predictions

    def get_measurements_for_horizon(self, predictions, horizon_idx):
        # Extract measurements for a specific horizon
        start_idx = horizon_idx * self.measurement_dims
        end_idx = start_idx + self.measurement_dims
        return predictions[:, start_idx:end_idx]

    def get_weighted_predictions(self, predictions, goals):
        """Get weighted prediction values using horizon weights and goals"""
        weighted_sum = torch.zeros(predictions.shape[0], device=predictions.device)

        # For each prediction horizon
        for i in range(self.n_horizons):
            # Get predictions for this horizon
            horizon_preds = self.get_measurements_for_horizon(predictions, i)

            # Apply goal weights to the measurements
            if goals is not None and goals.shape[1] == self.measurement_dims:
                # Weighted sum of measurements based on goals
                weighted_preds = torch.sum(horizon_preds * goals, dim=1)
            else:
                # Simple sum if goals not provided
                weighted_preds = torch.sum(horizon_preds, dim=1)

            # Apply horizon weight and add to total
            weighted_sum += weighted_preds * self.horizon_weights[i]

        return weighted_sum


# Simplified DFP Policy focusing on aggressive combat and progression
class DFPPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Initialize with feature extractor
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=DFPFeaturesExtractor,
            features_extractor_kwargs={"features_dim": 512},
            *args,
            **kwargs,
        )

        # Get measurement dimensions from observation space
        # Each measurement vector contains history for 5 time steps with 3 values each
        # Total measurement vector size is 15 (3 measurements * 5 history steps)
        # We need to extract just the number of base measurements (3)
        measurement_dims = 3  # Hardcode to 3 for [score, damage, progress]

        # Create prediction head
        self.dfp_predictor = DFPPredictionHead(
            input_dim=512,
            measurement_dims=measurement_dims,
            prediction_horizons=[1, 3, 5, 10, 20],
            horizon_weights=[1.0, 1.1, 1.2, 0.9, 0.7],  # Updated weights
        )

    def forward(self, obs, deterministic=False):
        # Extract features from observation
        features = self.extract_features(obs)
        # Get latent policy and value function representations
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Get action distribution
        distribution = self._get_action_dist_from_latent(latent_pi)
        # Get actions
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # Use the base value network for value function
        values = self.value_net(latent_vf)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_weighted_future(self, obs):
        """Get weighted prediction of future outcomes"""
        features = self.extract_features(obs)
        predictions = self.dfp_predictor(features)
        # Use goals from observation
        goals = obs.get("goals", None)
        # Get weighted predictions
        return self.dfp_predictor.get_weighted_predictions(predictions, goals)


def create_dfp_model(env, resume=False, model_path=MODEL_PATH):
    if resume and os.path.exists(model_path):
        logger.info(f"Loading existing DFP model from {model_path}")
        try:
            model = PPO.load(model_path, env=env, policy=DFPPolicy)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating new model instead")

    logger.info("Creating new DFP model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create model directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = PPO(
        policy=DFPPolicy,
        env=env,
        learning_rate=3e-4,
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
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
        ),
    )

    return model


# Training callback with focus on training metrics
class TrainingCallback(BaseCallback):
    def __init__(
        self, log_freq=1000, log_dir="logs", model_path=MODEL_PATH, save_freq=50000
    ):
        super().__init__()
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.best_mean_reward = -float("inf")
        self.log_dir = log_dir
        self.model_path = model_path

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # Create metrics header
        metrics_header = "timestamp,steps,mean_reward,mean_score,mean_damage,mean_progress,max_stage,combat_engagement"

        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(metrics_header + "\n")

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            mean_score = 0
            mean_damage = 0
            mean_progress = 0
            max_stage = 0
            combat_engagement = 0

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

                    # Track combat metrics
                    combat_rewards = [
                        ep_info.get("combat_engagement_reward", 0)
                        for ep_info in valid_episodes
                    ]
                    combat_engagement = np.mean(combat_rewards) if combat_rewards else 0

                    mean_score = np.mean(scores) if scores else 0
                    mean_damage = np.mean(damages) if damages else 0
                    mean_progress = np.mean(progress) if progress else 0
                    max_stage = np.max(stages) if stages else 0

            # Build log message
            log_msg = (
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Mean score: {mean_score:.1f}, Mean damage: {mean_damage:.1f}, "
                f"Mean progress: {mean_progress:.1f}, Max stage: {max_stage}, "
                f"Combat engagement: {combat_engagement:.2f}"
            )

            logger.info(log_msg)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Build metrics data string
                metrics = f"{timestamp},{self.n_calls},{mean_reward:.2f},{mean_score:.1f},{mean_damage:.1f},{mean_progress:.1f},{max_stage},{combat_engagement:.2f}"

                with open(self.metrics_file, "a") as f:
                    f.write(metrics + "\n")
            except Exception as e:
                logger.error(f"Error writing to metrics file: {e}")

            if self.n_calls % self.save_freq == 0:
                try:
                    self.model.save(
                        f"{os.path.dirname(self.model_path)}/kungfu_step_{self.n_calls}.zip"
                    )
                    logger.info(f"Model checkpoint saved at step {self.n_calls}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")

            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                try:
                    self.model.save(self.model_path)
                    logger.info(f"New best model saved with reward {mean_reward:.2f}")
                except Exception as e:
                    logger.error(f"Error saving best model: {e}")

        return True


def train_model(model, timesteps):
    callback = TrainingCallback(log_freq=1000, save_freq=50000)
    try:
        logger.info(f"Starting training for {timesteps} timesteps")
        model.learn(
            total_timesteps=timesteps,
            callback=callback,
            tb_log_name="kungfu_dfp_training",
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
        description="Train a Direct Future Prediction agent for Kung Fu Master with aggressive combat"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000000, help="Number of timesteps to train"
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
        default=1.2,
        help="Weight for combat engagement rewards (default: 1.2)",
    )
    parser.add_argument(
        "--progression-weight",
        type=float,
        default=1.5,
        help="Weight for progression rewards (default: 1.5)",
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

    try:
        logger.info(
            "Creating Kung Fu Master environment with aggressive combat settings..."
        )
        gc.collect()
        time.sleep(1)
        env = make_kungfu_env(is_play_mode=args.render, frame_stack=4, use_dfp=True)
        active_environments.add(env)

        logger.info("Creating DFP model...")
        model = create_dfp_model(env, resume=args.resume, model_path=model_path)

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
