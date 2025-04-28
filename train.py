import os
import argparse
import numpy as np
import torch
import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt

# Import our custom environment and policy
from kung_fu_env import (
    make_enhanced_kungfu_env,
    create_enhanced_kungfu_model,
    MODEL_PATH,
)


# Enhanced SaveCallback with projectile metrics tracking
class EnhancedSaveCallback(BaseCallback):
    """Callback for saving the model and tracking projectile avoidance metrics"""

    def __init__(
        self, check_freq=10000, log_freq=1000, log_dir="logs", model_path=None
    ):
        super().__init__()
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.best_mean_reward = -float("inf")
        self.best_projectile_avoidance = -float("inf")
        self.log_dir = log_dir
        self.model_path = model_path or MODEL_PATH

        # Create necessary directories
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Enhanced metrics file
        self.metrics_file = os.path.join(log_dir, "enhanced_training_metrics.csv")

        # Check if file exists, if not create header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(
                    "timestamp,steps,mean_reward,defensive_success_rate,projectiles_detected,"
                    "projectile_defensive_actions,projectile_avoidance_rate\n"
                )

    def _on_step(self):
        # Save model periodically
        if self.n_calls % self.check_freq == 0:
            path = f"{self.model_path.split('.')[0]}_{self.n_calls}.zip"
            self.model.save(path)
            print(f"Model saved to {path}")

        # Log training metrics periodically
        if self.n_calls % self.log_freq == 0:
            # Get basic metrics
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            # Initialize enhanced metrics
            defensive_success_rate = 0
            projectiles_detected = 0
            projectile_defensive_actions = 0
            projectile_avoidance_rate = 0

            if len(self.model.ep_info_buffer) > 0:
                # Extract metrics from episode info buffer
                valid_episodes = [
                    info
                    for info in self.model.ep_info_buffer
                    if "defensive_success_rate" in info
                ]

                if valid_episodes:
                    # Basic defensive metrics
                    defensive_rates = [
                        ep_info.get("defensive_success_rate", 0)
                        for ep_info in valid_episodes
                    ]
                    defensive_success_rate = np.mean(defensive_rates)

                    # Enhanced projectile metrics
                    projectiles = [
                        ep_info.get("detected_projectiles", 0)
                        for ep_info in valid_episodes
                    ]
                    proj_actions = [
                        ep_info.get("projectile_defensive_actions", 0)
                        for ep_info in valid_episodes
                    ]
                    proj_avoidance = [
                        ep_info.get("projectile_avoidance_rate", 0)
                        for ep_info in valid_episodes
                    ]

                    projectiles_detected = np.mean(projectiles)
                    projectile_defensive_actions = np.mean(proj_actions)
                    projectile_avoidance_rate = np.mean(proj_avoidance)

            # Log to console with enhanced metrics
            print(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Defensive success rate: {defensive_success_rate:.1f}%, "
                f"Projectiles detected: {projectiles_detected:.1f}, "
                f"Projectile avoidance rate: {projectile_avoidance_rate:.1f}%"
            )

            # Log metrics to CSV file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.metrics_file, "a") as f:
                f.write(
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{defensive_success_rate:.1f},"
                    f"{projectiles_detected:.1f},{projectile_defensive_actions:.1f},{projectile_avoidance_rate:.1f}\n"
                )

            # Save model if mean reward improves
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = f"{self.model_path.split('.')[0]}_best_reward.zip"
                self.model.save(best_path)
                print(
                    f"New best model with reward {mean_reward:.2f} saved to {best_path}"
                )

            # Also save model if projectile avoidance rate improves
            if (
                projectile_avoidance_rate > self.best_projectile_avoidance
                and projectile_avoidance_rate > 0
            ):
                self.best_projectile_avoidance = projectile_avoidance_rate
                best_proj_path = f"{self.model_path.split('.')[0]}_best_projectile.zip"
                self.model.save(best_proj_path)
                print(
                    f"New best projectile avoidance model ({projectile_avoidance_rate:.1f}%) saved to {best_proj_path}"
                )

        return True


def create_enhanced_model(env, resume=False, model_path=None):
    """Create a new PPO model with enhanced architecture for projectile detection"""
    # Set model path
    model_path = model_path or MODEL_PATH

    # Improved network architecture - deeper for better feature extraction
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 128, 64], vf=[256, 128, 64]),
        normalize_images=True,
    )

    if resume and os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        model = PPO.load(model_path, env=env, tensorboard_log="./logs/tensorboard/")
        print("Model loaded successfully")
    else:
        print("Creating new model with enhanced architecture")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0001,  # Lower learning rate for stability
            n_steps=2048,  # Collect more experience before updating
            batch_size=64,  # Smaller batch size for more updates
            n_epochs=10,  # More gradient updates per batch
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE parameter
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,  # Higher entropy for more exploration
            device=device,
            tensorboard_log="./logs/tensorboard/",
        )

    return model


def train_enhanced_model(model, timesteps, model_path=None):
    """Train the model with focus on projectile avoidance"""
    # Set model path
    model_path = model_path or MODEL_PATH

    # Create enhanced callback
    save_callback = EnhancedSaveCallback(
        check_freq=10000, log_freq=1000, model_path=model_path
    )

    print(f"Starting enhanced training for {timesteps} timesteps")
    model.learn(
        total_timesteps=timesteps,
        callback=save_callback,
        tb_log_name="kungfu_enhanced_training",
        progress_bar=True,
    )

    # Save final model
    model.save(model_path)
    print(f"Training completed. Final model saved to {model_path}")


def evaluate_enhanced_model(model, n_eval_episodes=5):
    """Evaluate model with focus on projectile avoidance"""
    print(f"Evaluating enhanced model over {n_eval_episodes} episodes...")
    eval_env = make_enhanced_kungfu_env(is_play_mode=True)

    # Metrics
    mean_reward = 0
    mean_stages_cleared = 0
    total_projectiles_detected = 0
    total_defensive_actions = 0
    total_successful_defensive_actions = 0
    total_projectile_defensive_actions = 0
    total_successful_projectile_avoidance = 0

    for i in range(n_eval_episodes):
        obs = eval_env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        max_stage = 0

        # Episode metrics
        episode_projectiles = 0
        episode_defensive_actions = 0
        episode_successful_defensive = 0
        episode_projectile_actions = 0
        episode_successful_projectile = 0

        # Run episode
        while not done:
            action, _ = model.predict(obs, deterministic=True)

            # Get pre-step state if available
            if hasattr(eval_env, "get_attr"):
                try:
                    prev_hp = eval_env.get_attr("prev_hp")[0]
                except:
                    prev_hp = 0
            else:
                prev_hp = 0

            # Take step
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Track maximum stage
            if "current_stage" in info and info["current_stage"] > max_stage:
                max_stage = info["current_stage"]

        # Get episode metrics from environment
        if hasattr(eval_env, "get_attr"):
            try:
                # Get projectile metrics
                episode_projectiles = eval_env.get_attr("detected_projectiles")[0]
                episode_defensive_actions = eval_env.get_attr("defensive_actions")[0]
                episode_successful_defensive = eval_env.get_attr(
                    "successful_defensive_actions"
                )[0]
                episode_projectile_actions = eval_env.get_attr(
                    "projectile_defensive_actions"
                )[0]
                episode_successful_projectile = eval_env.get_attr(
                    "successful_projectile_avoidance"
                )[0]

                # Update totals
                total_projectiles_detected += episode_projectiles
                total_defensive_actions += episode_defensive_actions
                total_successful_defensive_actions += episode_successful_defensive
                total_projectile_defensive_actions += episode_projectile_actions
                total_successful_projectile_avoidance += episode_successful_projectile
            except:
                pass

        # Calculate episode metrics
        defensive_success_rate = 0
        if episode_defensive_actions > 0:
            defensive_success_rate = (
                episode_successful_defensive / episode_defensive_actions
            ) * 100

        projectile_avoidance_rate = 0
        if episode_projectile_actions > 0:
            projectile_avoidance_rate = (
                episode_successful_projectile / episode_projectile_actions
            ) * 100

        # Update reward and stage metrics
        mean_reward += total_reward
        mean_stages_cleared += max_stage

        # Log episode results
        print(
            f"Episode {i+1}: Reward: {total_reward:.1f}, Steps: {steps}, Stage: {max_stage}, "
            f"Projectiles detected: {episode_projectiles}, "
            f"Projectile avoidance rate: {projectile_avoidance_rate:.1f}%"
        )

    # Calculate averages
    mean_reward /= n_eval_episodes
    mean_stages_cleared /= n_eval_episodes
    mean_projectiles = total_projectiles_detected / n_eval_episodes

    # Calculate overall defensive success rates
    overall_defensive_success_rate = 0
    if total_defensive_actions > 0:
        overall_defensive_success_rate = (
            total_successful_defensive_actions / total_defensive_actions
        ) * 100

    # Calculate overall projectile avoidance rate
    overall_projectile_avoidance_rate = 0
    if total_projectile_defensive_actions > 0:
        overall_projectile_avoidance_rate = (
            total_successful_projectile_avoidance / total_projectile_defensive_actions
        ) * 100

    # Print evaluation summary
    print(f"\nEnhanced evaluation results over {n_eval_episodes} episodes:")
    print(f"- Mean reward: {mean_reward:.2f}")
    print(f"- Mean max stage: {mean_stages_cleared:.1f}")
    print(f"- Average projectiles detected: {mean_projectiles:.1f}")
    print(f"- Overall defensive success rate: {overall_defensive_success_rate:.1f}%")
    print(
        f"- Overall projectile avoidance rate: {overall_projectile_avoidance_rate:.1f}%"
    )
    print(
        f"- Total projectile defenses: {total_successful_projectile_avoidance}/{total_projectile_defensive_actions}"
    )

    return mean_reward, mean_stages_cleared, overall_projectile_avoidance_rate


def plot_training_metrics(metrics_file="logs/enhanced_training_metrics.csv"):
    """Plot training metrics from CSV file"""
    if not os.path.exists(metrics_file):
        print(f"Metrics file {metrics_file} not found.")
        return

    try:
        # Load metrics
        metrics = pd.read_csv(metrics_file)

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot mean reward
        axes[0, 0].plot(metrics["steps"], metrics["mean_reward"])
        axes[0, 0].set_title("Mean Reward vs. Training Steps")
        axes[0, 0].set_xlabel("Steps")
        axes[0, 0].set_ylabel("Mean Reward")
        axes[0, 0].grid(True)

        # Plot defensive success rate
        axes[0, 1].plot(metrics["steps"], metrics["defensive_success_rate"])
        axes[0, 1].set_title("Defensive Success Rate vs. Training Steps")
        axes[0, 1].set_xlabel("Steps")
        axes[0, 1].set_ylabel("Success Rate (%)")
        axes[0, 1].grid(True)

        # Plot projectiles detected
        axes[1, 0].plot(metrics["steps"], metrics["projectiles_detected"])
        axes[1, 0].set_title("Projectiles Detected vs. Training Steps")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Avg Projectiles Detected")
        axes[1, 0].grid(True)

        # Plot projectile avoidance rate
        axes[1, 1].plot(metrics["steps"], metrics["projectile_avoidance_rate"])
        axes[1, 1].set_title("Projectile Avoidance Rate vs. Training Steps")
        axes[1, 1].set_xlabel("Steps")
        axes[1, 1].set_ylabel("Avoidance Rate (%)")
        axes[1, 1].grid(True)

        plt.tight_layout()

        # Save figure
        plots_dir = "logs/plots"
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
        plt.close()

        print(f"Training metrics plotted and saved to {plots_dir}/training_metrics.png")
    except Exception as e:
        print(f"Error plotting metrics: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train an enhanced AI to play Kung Fu Master with projectile detection"
    )
    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during training"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate the model after training"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only evaluate the model, no training"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to a specific model to evaluate",
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Plot training metrics after training",
    )
    # Add new argument for using projectile features
    parser.add_argument(
        "--use-projectile-features",
        action="store_true",
        help="Use explicit projectile features in observations",
        default=True,
    )
    args = parser.parse_args()

    # Fixed frame stack size of 8 for better projectile detection
    frame_stack = 4

    # Create enhanced environment with projectile features if requested
    print(
        f"Creating enhanced Kung Fu environment with frame stacking (n_stack={frame_stack}) for projectile detection..."
    )
    env = make_enhanced_kungfu_env(
        is_play_mode=args.render,
        frame_stack=frame_stack,
        use_projectile_features=args.use_projectile_features,
    )

    # Model paths with feature indication
    model_suffix = "_with_projectile_features" if args.use_projectile_features else ""
    model_path = (
        args.model_path
        if args.model_path
        else f"{MODEL_PATH.split('.')[0]}{model_suffix}.zip"
    )

    # Evaluation only mode
    if args.eval_only:
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return

        print(f"Loading model from {model_path} for evaluation...")
        if args.use_projectile_features:
            # Use appropriate model loading based on features
            model = create_enhanced_kungfu_model(
                env, resume=True, model_path=model_path
            )
        else:
            # Use standard model
            model = PPO.load(model_path, env=env)

        evaluate_enhanced_model(model, n_eval_episodes=5)
        env.close()
        return

    # Create or load model
    if args.use_projectile_features:
        # Use custom model with projectile awareness
        print("Creating model with projectile feature support...")
        model = create_enhanced_kungfu_model(
            env, resume=args.resume, model_path=model_path
        )
    else:
        # Use standard model
        model = create_enhanced_model(env, resume=args.resume, model_path=model_path)

    # Train model
    train_enhanced_model(model, args.timesteps, model_path=model_path)

    # Plot metrics if requested
    if args.plot_metrics:
        plot_training_metrics()

    # Evaluate model if requested
    if args.eval:
        evaluate_enhanced_model(model)

    # Clean up
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
