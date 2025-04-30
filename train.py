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
import matplotlib.pyplot as plt
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


# Direct Future Prediction feature extractor
class DFPFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.image_space = observation_space.spaces["image"]
        # self.image_space.shape = (C, H, W) after VecTransposeImage
        n_input_channels = self.image_space.shape[0]  # Number of channels
        height = self.image_space.shape[1]
        width = self.image_space.shape[2]

        self.measurement_dim = observation_space.spaces["measurements"].shape[0]
        self.goal_dim = observation_space.spaces["goals"].shape[0]
        logger.info(f"DFP Feature Extractor - Image shape: {self.image_space.shape}")

        # Define CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size with a dummy input
        with torch.no_grad():
            dummy_img = torch.zeros(
                (1, n_input_channels, height, width)
            )  # (batch, C, H, W)
            cnn_out = self.cnn(dummy_img)
            self.cnn_out_size = cnn_out.shape[1]
        logger.info(f"CNN output features: {self.cnn_out_size}")

        # Measurement network
        measurement_input_size = self.measurement_dim + self.goal_dim
        self.measurement_net = nn.Sequential(
            nn.Linear(measurement_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.measurement_out_size = 128

        # Combined network
        combined_size = self.cnn_out_size + self.measurement_out_size
        self.combined_net = nn.Sequential(
            nn.Linear(combined_size, features_dim), nn.ReLU()
        )

    def forward(self, observations):
        image_tensor = observations["image"]
        # image_tensor is already (num_envs, C, H, W) due to VecTransposeImage
        image_features = self.cnn(image_tensor / 255.0)

        # Handle measurements and goals
        measurements = observations["measurements"]
        goals = observations["goals"]
        if len(measurements.shape) == 1:
            measurements = measurements.unsqueeze(0)
        if len(goals.shape) == 1:
            goals = goals.unsqueeze(0)
        measurement_tensor = torch.cat([measurements, goals], dim=1)
        measurement_features = self.measurement_net(measurement_tensor)

        # Combine features
        combined_features = torch.cat([image_features, measurement_features], dim=1)
        return self.combined_net(combined_features)


# DFP prediction head for value function
class DFPPredictionHead(nn.Module):
    def __init__(
        self, input_dim, measurement_dims=3, prediction_horizons=[1, 3, 5, 10, 20]
    ):
        super().__init__()
        self.input_dim = input_dim
        self.measurement_dims = measurement_dims
        self.prediction_horizons = prediction_horizons
        self.n_horizons = len(prediction_horizons)
        self.output_dim = measurement_dims * self.n_horizons
        self.prediction_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.output_dim),
        )

    def forward(self, features):
        predictions = self.prediction_net(features)
        return predictions

    def get_measurements_for_horizon(self, predictions, horizon_idx):
        start_idx = horizon_idx * self.measurement_dims
        end_idx = start_idx + self.measurement_dims
        return predictions[:, start_idx:end_idx]


# Custom DFP policy for PPO
class DFPPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            features_extractor_class=DFPFeaturesExtractor,
            features_extractor_kwargs={"features_dim": 512},
            *args,
            **kwargs,
        )
        self.dfp_predictor = DFPPredictionHead(
            input_dim=512, measurement_dims=3, prediction_horizons=[1, 3, 5, 10, 20]
        )

    def forward(self, obs, deterministic=False):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return actions, values, log_prob

    def predict_future_measurements(self, obs):
        features = self.extract_features(obs)
        predictions = self.dfp_predictor(features)
        return predictions

    def evaluate_actions(self, obs, actions):
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy


# Custom DFP loss function for PPO
class DFPLoss:
    def __init__(self, policy, prediction_weight=0.5):
        self.policy = policy
        self.prediction_weight = prediction_weight
        self.prediction_loss = nn.MSELoss()

    def __call__(self, rollout_data):
        obs = rollout_data.observations
        actions = rollout_data.actions
        values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)
        advantages = rollout_data.advantages
        returns = rollout_data.returns
        policy_loss = -(advantages * log_prob).mean()
        value_loss = nn.functional.mse_loss(values, returns)
        entropy_loss = -entropy.mean()
        prediction_loss = torch.tensor(0.0)
        if hasattr(rollout_data, "future_measurements"):
            future_measurements = rollout_data.future_measurements
            predictions = self.policy.predict_future_measurements(obs)
            for i, horizon in enumerate(self.policy.dfp_predictor.prediction_horizons):
                horizon_key = f"future_{horizon}"
                if horizon_key in future_measurements:
                    horizon_preds = (
                        self.policy.dfp_predictor.get_measurements_for_horizon(
                            predictions, i
                        )
                    )
                    targets = future_measurements[horizon_key]
                    horizon_loss = self.prediction_loss(horizon_preds, targets)
                    prediction_loss += horizon_loss
        total_loss = (
            policy_loss
            + 0.5 * value_loss
            - 0.01 * entropy_loss
            + self.prediction_weight * prediction_loss
        )
        return total_loss


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
            net_arch=dict(pi=[256, 256], vf=[256, 256]),  # Updated to avoid warning
            optimizer_class=torch.optim.Adam,
            optimizer_kwargs=dict(eps=1e-5),
        ),
    )

    return model


# Simplified training callback with logging
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

        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(
                    "timestamp,steps,mean_reward,mean_score,mean_damage,mean_progress,max_stage\n"
                )

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            mean_score = 0
            mean_damage = 0
            mean_progress = 0
            max_stage = 0

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

                    mean_score = np.mean([sum(s) for s in scores]) if scores else 0
                    mean_damage = np.mean([sum(d) for d in damages]) if damages else 0
                    mean_progress = (
                        np.mean([sum(p) for p in progress]) if progress else 0
                    )
                    max_stage = np.max(stages) if stages else 0

            logger.info(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Mean score: {mean_score:.1f}, Mean damage: {mean_damage:.1f}, "
                f"Mean progress: {mean_progress:.1f}, Max stage: {max_stage}"
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(
                        f"{timestamp},{self.n_calls},{mean_reward:.2f},{mean_score:.1f},"
                        f"{mean_damage:.1f},{mean_progress:.1f},{max_stage}\n"
                    )
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


def plot_training_metrics(metrics_file="logs/training_metrics.csv"):
    plots_dir = "logs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    if os.path.exists(metrics_file):
        try:
            metrics = pd.read_csv(metrics_file)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            axes[0, 0].plot(metrics["steps"], metrics["mean_reward"])
            axes[0, 0].set_title("Mean Reward vs. Training Steps")
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Mean Reward")
            axes[0, 0].grid(True)

            axes[0, 1].plot(metrics["steps"], metrics["mean_score"])
            axes[0, 1].set_title("Mean Score vs. Training Steps")
            axes[0, 1].set_xlabel("Steps")
            axes[0, 1].set_ylabel("Score")
            axes[0, 1].grid(True)

            axes[1, 0].plot(metrics["steps"], metrics["mean_damage"])
            axes[1, 0].set_title("Mean Damage vs. Training Steps")
            axes[1, 0].set_xlabel("Steps")
            axes[1, 0].set_ylabel("Damage")
            axes[1, 0].grid(True)

            axes[1, 1].plot(metrics["steps"], metrics["mean_progress"])
            axes[1, 1].set_title("Mean Progress vs. Training Steps")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Progress")
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
            plt.close()

            logger.info(
                f"Training metrics plotted and saved to {plots_dir}/training_metrics.png"
            )
        except Exception as e:
            logger.error(f"Error plotting training metrics: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Metrics file {metrics_file} not found.")


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


def evaluate_model(model_path=MODEL_PATH, num_episodes=10, render=True):
    logger.info(f"Evaluating model from {model_path}")
    env = make_kungfu_env(is_play_mode=render, frame_stack=4, use_dfp=True)
    active_environments.add(env)
    try:
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, policy=DFPPolicy)
        episode_rewards = []
        episode_scores = []
        episode_damages = []
        episode_progress = []
        max_stage_reached = 0

        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            episode_score = 0
            episode_damage = 0
            episode_prog = 0
            current_stage = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                if isinstance(info, dict):
                    episode_score += info.get("score_increase", 0)
                    episode_damage += info.get("damage_taken", 0)
                    episode_prog += info.get("progress_made", 0)
                    current_stage = max(current_stage, info.get("current_stage", 0))

            episode_rewards.append(episode_reward)
            episode_scores.append(episode_score)
            episode_damages.append(episode_damage)
            episode_progress.append(episode_prog)
            max_stage_reached = max(max_stage_reached, current_stage)

            logger.info(
                f"Episode {episode+1}/{num_episodes}: Reward={episode_reward:.2f}, "
                f"Score={episode_score:.1f}, Damage={episode_damage:.1f}, "
                f"Progress={episode_prog:.1f}, Stage={current_stage}"
            )

        avg_reward = np.mean(episode_rewards)
        avg_score = np.mean(episode_scores)
        avg_damage = np.mean(episode_damages)
        avg_progress = np.mean(episode_progress)

        logger.info(f"Evaluation results over {num_episodes} episodes:")
        logger.info(f"Average reward: {avg_reward:.2f}")
        logger.info(f"Average score: {avg_score:.1f}")
        logger.info(f"Average damage: {avg_damage:.1f}")
        logger.info(f"Average progress: {avg_progress:.1f}")
        logger.info(f"Max stage reached: {max_stage_reached}")

        return {
            "avg_reward": avg_reward,
            "avg_score": avg_score,
            "avg_damage": avg_damage,
            "avg_progress": avg_progress,
            "max_stage": max_stage_reached,
        }
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        logger.info("Closing evaluation environment")
        env.close()
        active_environments.discard(env)


def main():
    parser = argparse.ArgumentParser(
        description="Train a Direct Future Prediction agent for Kung Fu Master"
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
        "--eval-only", action="store_true", help="Only evaluate the model, no training"
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Plot training metrics after training",
    )
    args = parser.parse_args()

    if args.eval_only:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Error: Model file {MODEL_PATH} not found.")
            return
        logger.info(f"Running evaluation-only mode on model: {MODEL_PATH}")
        evaluate_model(MODEL_PATH, num_episodes=10, render=True)
        return

    try:
        logger.info("Creating Kung Fu Master environment with DFP...")
        gc.collect()
        time.sleep(1)
        env = make_kungfu_env(is_play_mode=args.render, frame_stack=4, use_dfp=True)
        active_environments.add(env)
        logger.info("Creating DFP model...")
        model = create_dfp_model(env, resume=args.resume, model_path=MODEL_PATH)
        train_model(model, args.timesteps)
        logger.info("Closing training environment")
        env.close()
        active_environments.discard(env)
        if args.plot_metrics:
            plot_training_metrics()
        logger.info("Evaluating trained model...")
        evaluate_model(MODEL_PATH, num_episodes=5, render=True)
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
