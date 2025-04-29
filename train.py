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


# Simplified BestModelSaveCallback - Only saves one model when performance improves
class BestModelSaveCallback(BaseCallback):
    """
    Callback that saves the model only when mean reward improves significantly.
    Only keeps one best model file at model/kungfu.zip
    """

    def __init__(
        self, log_freq=1000, log_dir="logs", model_path=MODEL_PATH, save_threshold=0.05
    ):
        super().__init__()
        self.log_freq = log_freq
        self.best_mean_reward = -float("inf")
        self.log_dir = log_dir
        self.model_path = model_path
        self.save_threshold = (
            save_threshold  # Minimum improvement threshold to save model
        )

        # Create necessary directories
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Enhanced metrics file
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        # Check if file exists, if not create header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(
                    "timestamp,steps,mean_reward,defensive_success_rate,projectiles_detected,"
                    "projectile_defensive_actions,projectile_avoidance_rate,max_stage\n"
                )

    def _on_step(self):
        # Log training metrics periodically
        if self.n_calls % self.log_freq == 0:
            # Get basic metrics
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            # Initialize enhanced metrics
            defensive_success_rate = 0
            projectiles_detected = 0
            projectile_defensive_actions = 0
            projectile_avoidance_rate = 0
            max_stage = 0

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
                    stages = [
                        ep_info.get("current_stage", 0) for ep_info in valid_episodes
                    ]

                    projectiles_detected = np.mean(projectiles)
                    projectile_defensive_actions = np.mean(proj_actions)
                    projectile_avoidance_rate = np.mean(proj_avoidance)
                    max_stage = np.max(stages) if stages else 0

            # Log to console with enhanced metrics
            print(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Defensive success rate: {defensive_success_rate:.1f}%, "
                f"Projectiles detected: {projectiles_detected:.1f}, "
                f"Projectile avoidance rate: {projectile_avoidance_rate:.1f}%, "
                f"Max stage: {max_stage}"
            )

            # Log metrics to CSV file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.metrics_file, "a") as f:
                f.write(
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{defensive_success_rate:.1f},"
                    f"{projectiles_detected:.1f},{projectile_defensive_actions:.1f},"
                    f"{projectile_avoidance_rate:.1f},{max_stage}\n"
                )

            # Only save model when mean reward significantly improves
            if mean_reward > (self.best_mean_reward * (1 + self.save_threshold)) or (
                mean_reward > 0 and self.best_mean_reward < 0
            ):
                relative_improvement = (
                    (mean_reward - self.best_mean_reward)
                    / max(1, abs(self.best_mean_reward))
                ) * 100
                self.best_mean_reward = mean_reward

                # Save to the single model path
                self.model.save(self.model_path)
                print(
                    f"New best model with reward {mean_reward:.2f} saved to {self.model_path} "
                    f"(+{relative_improvement:.1f}% improvement)"
                )

        return True


def train_model(model, timesteps, model_path=MODEL_PATH):
    """Train the model with focus on projectile avoidance"""
    # Create callback
    save_callback = BestModelSaveCallback(
        log_freq=1000, model_path=model_path, save_threshold=0.05
    )

    print(f"Starting training for {timesteps} timesteps")
    model.learn(
        total_timesteps=timesteps,
        callback=save_callback,
        tb_log_name="kungfu_training",
        progress_bar=True,
    )

    print(f"Training completed")


def evaluate_model(model, n_eval_episodes=5, env=None):
    """Evaluate model with focus on projectile avoidance"""
    print(f"Evaluating model over {n_eval_episodes} episodes...")
    # Use the provided environment or create a new one
    eval_env = env if env is not None else make_enhanced_kungfu_env(is_play_mode=True)

    # Metrics
    mean_reward = 0
    mean_stages_cleared = 0
    total_projectiles_detected = 0
    total_defensive_actions = 0
    total_successful_defensive_actions = 0
    total_projectile_defensive_actions = 0
    total_successful_projectile_avoidance = 0

    for i in range(n_eval_episodes):

        # Handle different return types from reset()
        reset_result = eval_env.reset()

        # Check what type of result we got
        if isinstance(reset_result, tuple):
            # If it's a tuple, unpack it (usually obs, info)
            obs = reset_result[0]
        else:
            # If it's not a tuple, it's just the observation
            obs = reset_result

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
    print(f"\nEvaluation results over {n_eval_episodes} episodes:")
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


def plot_training_metrics(metrics_file="logs/training_metrics.csv"):
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
        description="Train an AI to play Kung Fu Master with projectile detection"
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
        "--plot-metrics",
        action="store_true",
        help="Plot training metrics after training",
    )
    args = parser.parse_args()

    # Fixed frame stack size for better projectile detection
    frame_stack = 4

    # Create enhanced environment with projectile features always enabled
    print(
        f"Creating enhanced Kung Fu environment with frame stacking (n_stack={frame_stack}) for projectile detection..."
    )
    env = make_enhanced_kungfu_env(
        is_play_mode=args.render,
        frame_stack=frame_stack,
        use_projectile_features=True,  # Always use projectile features
    )

    # Evaluation only mode
    if args.eval_only:
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file {MODEL_PATH} not found.")
            return

        print(f"Loading model from {MODEL_PATH} for evaluation...")
        model = create_enhanced_kungfu_model(env, resume=True, model_path=MODEL_PATH)

        # Close the environment before creating a new one for evaluation
        env.close()

        # Create a new environment for evaluation
        eval_env = make_enhanced_kungfu_env(
            is_play_mode=True,
            frame_stack=frame_stack,
            use_projectile_features=True,  # Always use projectile features
        )

        # Pass the evaluation environment to the evaluation function
        evaluate_model(model, n_eval_episodes=5, env=eval_env)
        eval_env.close()
        return

    # Create or load model - always use projectile features
    print("Creating model with projectile feature support...")
    model = create_enhanced_kungfu_model(env, resume=args.resume, model_path=MODEL_PATH)

    # Train model
    train_model(model, args.timesteps, model_path=MODEL_PATH)

    # Plot metrics if requested
    if args.plot_metrics:
        plot_training_metrics()

    # Evaluate model if requested
    if args.eval:
        evaluate_model(model)

    # Clean up
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
