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


# Custom CNN features extractor for Kung Fu Master
class KungFuCNNExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=512):
        super().__init__()

        # Get the shape of the input observation
        n_input_channels = observation_space.shape[0]

        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Test with dummy input to determine CNN output shape
        with torch.no_grad():
            dummy_input = torch.zeros((1,) + observation_space.shape)
            cnn_output = self.cnn(dummy_input)
            cnn_output_shape = cnn_output.size(1)

        # Final output layer
        self.fc = nn.Sequential(nn.Linear(cnn_output_shape, features_dim), nn.ReLU())

    def forward(self, observations):
        # Normalize the observations
        x = observations.float() / 255.0
        x = self.cnn(x)
        return self.fc(x)


# Custom policy with enhanced threat awareness
class KungFuPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Use custom feature extractor
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            *args,
            **kwargs,
        )

        # Create CNN feature extractor
        self.features_extractor = KungFuCNNExtractor(observation_space)

    def _build_mlp_extractor(self):
        # Override to use our custom CNN extractor
        return super()._build_mlp_extractor()


# Training callback with focus on training metrics and threat detection
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

        # Create metrics header with threat tracking
        metrics_header = (
            "timestamp,steps,mean_reward,mean_score,mean_damage,mean_progress,max_stage,"
            "combat_engagement,threats_detected,recommended_actions_taken"
        )

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
                    mean_progress = np.mean(progress) if progress else 0
                    max_stage = np.max(stages) if stages else 0

            # Build log message
            log_msg = (
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Mean score: {mean_score:.1f}, Mean damage: {mean_damage:.1f}, "
                f"Mean progress: {mean_progress:.1f}, Max stage: {max_stage}, "
                f"Combat engagement: {combat_engagement:.2f}, "
                f"Threats detected: {threats_detected:.2f}, "
                f"Actions taken: {recommended_actions_taken:.2f}"
            )

            logger.info(log_msg)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # Build metrics data string
                metrics = (
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{mean_score:.1f},"
                    f"{mean_damage:.1f},{mean_progress:.1f},{max_stage},{combat_engagement:.2f},"
                    f"{threats_detected:.2f},{recommended_actions_taken:.2f}"
                )

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


def create_model(env, resume=False, model_path=MODEL_PATH):
    if resume and os.path.exists(model_path):
        logger.info(f"Loading existing model from {model_path}")
        try:
            model = PPO.load(model_path, env=env)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating new model instead")

    logger.info("Creating new model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create model directory if needed
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    model = PPO(
        policy="CnnPolicy",
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
            normalize_images=False,  # Add this line
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
        ),
    )

    return model


def train_model(model, timesteps):
    callback = TrainingCallback(log_freq=1000, save_freq=50000)
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
        description="Train a PPO agent for Kung Fu Master with threat detection"
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
        default=1.8,
        help="Weight for combat engagement rewards (default: 1.8)",
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
        logger.info("Creating Kung Fu Master environment with threat detection...")
        gc.collect()
        time.sleep(1)
        env = make_kungfu_env(is_play_mode=args.render, frame_stack=4, use_dfp=False)
        active_environments.add(env)

        logger.info("Creating model...")
        model = create_model(env, resume=args.resume, model_path=model_path)

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
