import os
import argparse
import numpy as np
import torch
import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import logging
import signal
import atexit
import traceback
import time
import gc
import multiprocessing as mp
import tempfile
from copy import deepcopy
import sys
import subprocess

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

# Import our custom environment and policy
try:
    from kung_fu_env import (
        make_enhanced_kungfu_env,
        create_enhanced_kungfu_model,
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


# Enhanced BestModelSaveCallback with breakthrough tracking
class BestModelSaveCallback(BaseCallback):
    """
    Enhanced callback that saves the model when performance improves and tracks breakthrough metrics.
    """

    def __init__(
        self,
        log_freq=1000,
        log_dir="logs",
        model_path=MODEL_PATH,
        save_threshold=0.05,
        checkpoint_freq=100000,  # Add periodic checkpoints
    ):
        super().__init__()
        self.log_freq = log_freq
        self.best_mean_reward = -float("inf")
        self.best_breakthrough_count = 0
        self.best_max_stage = 0
        self.log_dir = log_dir
        self.model_path = model_path
        self.save_threshold = save_threshold
        self.checkpoint_freq = checkpoint_freq
        self.last_checkpoint_step = 0

        # Create necessary directories
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)

        # Enhanced metrics file
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        # Check if file exists, if not create header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write(
                    "timestamp,steps,mean_reward,defensive_success_rate,projectiles_detected,"
                    "projectile_defensive_actions,projectile_avoidance_rate,max_stage,"
                    "breakthrough_rewards,offensive_success_rate,consecutive_defenses,stagnation\n"
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
            breakthrough_rewards = 0
            offensive_success_rate = 0
            consecutive_defenses = 0
            stagnation_counter = 0

            if hasattr(self.model, "ep_info_buffer") and self.model.ep_info_buffer:
                # Extract metrics from episode info buffer
                valid_episodes = [
                    info
                    for info in self.model.ep_info_buffer
                    if isinstance(info, dict) and "defensive_success_rate" in info
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

                    # New metrics
                    breakthroughs = [
                        ep_info.get("breakthrough_rewards_given", 0)
                        for ep_info in valid_episodes
                    ]
                    offensive_rates = [
                        ep_info.get("offensive_success_rate", 0)
                        for ep_info in valid_episodes
                    ]
                    defense_counts = [
                        ep_info.get("consecutive_defenses", 0)
                        for ep_info in valid_episodes
                    ]
                    stagnation_values = [
                        ep_info.get("stagnation_counter", 0)
                        for ep_info in valid_episodes
                    ]

                    projectiles_detected = np.mean(projectiles)
                    projectile_defensive_actions = np.mean(proj_actions)
                    projectile_avoidance_rate = np.mean(proj_avoidance)
                    max_stage = np.max(stages) if stages else 0
                    breakthrough_rewards = np.mean(breakthroughs)
                    offensive_success_rate = np.mean(offensive_rates)
                    consecutive_defenses = np.mean(defense_counts)
                    stagnation_counter = np.mean(stagnation_values)

            # Log to console with enhanced metrics
            logger.info(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Defensive success rate: {defensive_success_rate:.1f}%, "
                f"Projectiles detected: {projectiles_detected:.1f}, "
                f"Proj. avoidance: {projectile_avoidance_rate:.1f}%, "
                f"Breakthroughs: {breakthrough_rewards:.1f}, "
                f"Max stage: {max_stage}, "
                f"Stagnation: {stagnation_counter:.1f}"
            )

            # Log metrics to CSV file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                with open(self.metrics_file, "a") as f:
                    f.write(
                        f"{timestamp},{self.n_calls},{mean_reward:.2f},{defensive_success_rate:.1f},"
                        f"{projectiles_detected:.1f},{projectile_defensive_actions:.1f},"
                        f"{projectile_avoidance_rate:.1f},{max_stage},"
                        f"{breakthrough_rewards:.1f},{offensive_success_rate:.1f},"
                        f"{consecutive_defenses:.1f},{stagnation_counter:.1f}\n"
                    )
            except Exception as e:
                logger.error(f"Error writing to metrics file: {e}")

            # Save model based on combined metrics: reward, breakthroughs, and max stage
            save_model = False
            save_reason = ""

            # Check for reward improvement
            if mean_reward > (self.best_mean_reward * (1 + self.save_threshold)) or (
                mean_reward > 0 and self.best_mean_reward < 0
            ):
                relative_improvement = (
                    (mean_reward - self.best_mean_reward)
                    / max(1, abs(self.best_mean_reward))
                ) * 100
                self.best_mean_reward = mean_reward
                save_model = True
                save_reason = f"reward improved to {mean_reward:.2f} (+{relative_improvement:.1f}%)"

            # Check for breakthrough count improvement
            if breakthrough_rewards > self.best_breakthrough_count:
                self.best_breakthrough_count = breakthrough_rewards
                save_model = True
                if save_reason:
                    save_reason += " and "
                save_reason += f"breakthroughs increased to {breakthrough_rewards:.1f}"

            # Check for stage improvement
            if max_stage > self.best_max_stage:
                self.best_max_stage = max_stage
                save_model = True
                if save_reason:
                    save_reason += " and "
                save_reason += f"max stage reached {max_stage}"

            # Save if any improvement was detected
            if save_model:
                try:
                    # Save to the single model path
                    self.model.save(self.model_path)
                    logger.info(
                        f"New best model saved to {self.model_path} - {save_reason}"
                    )
                except Exception as e:
                    logger.error(f"Error saving best model: {e}")

            # Periodically save checkpoints regardless of performance
            if self.n_calls - self.last_checkpoint_step >= self.checkpoint_freq:
                try:
                    checkpoint_path = os.path.join(
                        self.log_dir, "checkpoints", f"kungfu_step_{self.n_calls}.zip"
                    )
                    self.model.save(checkpoint_path)
                    logger.info(f"Periodic checkpoint saved at step {self.n_calls}")
                    self.last_checkpoint_step = self.n_calls
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")

        return True


def run_subprocess_eval(model_path, num_episodes=10, render=True):
    """
    Run evaluation in a separate subprocess to guarantee clean environment creation
    """
    logger.info("Starting evaluation in a separate process...")

    # Construct the command to run the evaluation script
    cmd = [
        sys.executable,  # Current Python interpreter
        "eval_subprocess.py",  # The evaluation script (we'll create this)
        "--model",
        model_path,
        "--episodes",
        str(num_episodes),
    ]

    if render:
        cmd.append("--render")

    # Run the subprocess
    try:
        # Run process and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)

        # Log the output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"Eval subprocess: {line}")

        if result.stderr:
            for line in result.stderr.splitlines():
                logger.error(f"Eval subprocess error: {line}")

        # Check return code
        if result.returncode != 0:
            logger.error(
                f"Evaluation subprocess failed with return code {result.returncode}"
            )
        else:
            logger.info("Evaluation subprocess completed successfully")

        return result.returncode == 0

    except Exception as e:
        logger.error(f"Error running evaluation subprocess: {e}")
        logger.error(traceback.format_exc())
        return False


def train_model(model, timesteps, model_path=MODEL_PATH):
    """Train the model with focus on projectile avoidance and progress"""
    # Create callbacks
    save_callback = BestModelSaveCallback(
        log_freq=1000,
        model_path=model_path,
        save_threshold=0.05,
        checkpoint_freq=100000,
    )

    try:
        logger.info(f"Starting training for {timesteps} timesteps")

        model.learn(
            total_timesteps=timesteps,
            callback=save_callback,
            tb_log_name="kungfu_training",
            progress_bar=True,
        )
        logger.info("Training completed successfully")

        # Final evaluation after training - using subprocess
        logger.info("Performing final evaluation in a separate process...")

        # Make sure the model is saved first
        model.save(model_path)

        # Run evaluation in a separate process
        run_subprocess_eval(model_path, num_episodes=10, render=True)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        logger.error(traceback.format_exc())


def plot_training_metrics(
    metrics_file="logs/training_metrics.csv",
    eval_metrics_file="logs/eval/eval_metrics.csv",
):
    """Plot enhanced training metrics from CSV file"""
    plots_dir = "logs/plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot training metrics
    if os.path.exists(metrics_file):
        try:
            # Load metrics
            metrics = pd.read_csv(metrics_file)

            # Create figure with multiple subplots - expanded to include breakthrough metrics
            fig, axes = plt.subplots(3, 2, figsize=(15, 14))

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

            # Plot breakthrough rewards - NEW
            if "breakthrough_rewards" in metrics.columns:
                axes[2, 0].plot(metrics["steps"], metrics["breakthrough_rewards"])
                axes[2, 0].set_title("Breakthrough Rewards vs. Training Steps")
                axes[2, 0].set_xlabel("Steps")
                axes[2, 0].set_ylabel("Avg Breakthroughs")
                axes[2, 0].grid(True)

            # Plot offensive success rate - NEW
            if "offensive_success_rate" in metrics.columns:
                axes[2, 1].plot(metrics["steps"], metrics["offensive_success_rate"])
                axes[2, 1].set_title("Offensive Success Rate vs. Training Steps")
                axes[2, 1].set_xlabel("Steps")
                axes[2, 1].set_ylabel("Success Rate (%)")
                axes[2, 1].grid(True)

            plt.tight_layout()

            # Save figure
            plt.savefig(os.path.join(plots_dir, "training_metrics.png"))
            plt.close()

            logger.info(
                f"Training metrics plotted and saved to {plots_dir}/training_metrics.png"
            )

            # Create an additional plot to track balance between defense and offense
            if (
                "consecutive_defenses" in metrics.columns
                and "offensive_success_rate" in metrics.columns
            ):
                plt.figure(figsize=(10, 6))

                # Plot consecutive defenses
                plt.plot(
                    metrics["steps"],
                    metrics["consecutive_defenses"],
                    label="Consecutive Defensive Actions",
                    color="blue",
                )

                # Plot offensive success on second y-axis
                plt2 = plt.twinx()
                plt2.plot(
                    metrics["steps"],
                    metrics["offensive_success_rate"],
                    label="Offensive Success Rate",
                    color="red",
                )

                plt.title("Defense vs Offense Balance During Training")
                plt.xlabel("Training Steps")
                plt.ylabel("Consecutive Defenses")
                plt2.set_ylabel("Offensive Success Rate (%)")

                # Add legend
                lines1, labels1 = plt.gca().get_legend_handles_labels()
                lines2, labels2 = plt2.get_legend_handles_labels()
                plt2.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

                plt.grid(True)
                plt.savefig(os.path.join(plots_dir, "defense_offense_balance.png"))
                plt.close()

                logger.info(
                    f"Defense/offense balance plot saved to {plots_dir}/defense_offense_balance.png"
                )

            # Create stagnation plot if available
            if "stagnation" in metrics.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(
                    metrics["steps"],
                    metrics["stagnation"],
                    label="Stagnation Counter",
                    color="red",
                )
                plt.title("Agent Stagnation During Training")
                plt.xlabel("Training Steps")
                plt.ylabel("Stagnation Counter")
                plt.grid(True)
                plt.legend()
                plt.savefig(os.path.join(plots_dir, "stagnation.png"))
                plt.close()

                logger.info(f"Stagnation plot saved to {plots_dir}/stagnation.png")

        except Exception as e:
            logger.error(f"Error plotting training metrics: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.warning(f"Metrics file {metrics_file} not found.")

    # Plot evaluation metrics if they exist
    if os.path.exists(eval_metrics_file):
        try:
            # Load evaluation metrics
            eval_metrics = pd.read_csv(eval_metrics_file)

            # Create evaluation metrics plot
            plt.figure(figsize=(12, 8))

            # Plot reward
            plt.subplot(2, 2, 1)
            plt.plot(eval_metrics["steps"], eval_metrics["mean_reward"], "bo-")
            plt.title("Evaluation Reward")
            plt.xlabel("Steps")
            plt.ylabel("Mean Reward")
            plt.grid(True)

            # Plot max stage
            plt.subplot(2, 2, 2)
            plt.plot(eval_metrics["steps"], eval_metrics["max_stage"], "go-")
            plt.title("Evaluation Max Stage")
            plt.xlabel("Steps")
            plt.ylabel("Stage")
            plt.grid(True)

            # Plot projectile avoidance
            plt.subplot(2, 2, 3)
            plt.plot(
                eval_metrics["steps"], eval_metrics["projectile_avoidance_rate"], "ro-"
            )
            plt.title("Evaluation Projectile Avoidance")
            plt.xlabel("Steps")
            plt.ylabel("Avoidance Rate (%)")
            plt.grid(True)

            # Plot breakthroughs
            plt.subplot(2, 2, 4)
            plt.plot(eval_metrics["steps"], eval_metrics["breakthroughs"], "mo-")
            plt.title("Evaluation Breakthroughs")
            plt.xlabel("Steps")
            plt.ylabel("Avg Breakthroughs")
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "evaluation_metrics.png"))
            plt.close()

            logger.info(
                f"Evaluation metrics plotted and saved to {plots_dir}/evaluation_metrics.png"
            )

        except Exception as e:
            logger.error(f"Error plotting evaluation metrics: {str(e)}")
            logger.error(traceback.format_exc())
    else:
        logger.info(f"Evaluation metrics file {eval_metrics_file} not found.")


def main():
    parser = argparse.ArgumentParser(
        description="Train an AI to play Kung Fu Master with improved projectile avoidance and progression"
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
        "--eval-only", action="store_true", help="Only evaluate the model, no training"
    )
    parser.add_argument(
        "--plot-metrics",
        action="store_true",
        help="Plot training metrics after training",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run a detailed analysis of agent performance",
    )
    args = parser.parse_args()

    # Evaluation only mode - use subprocess
    if args.eval_only:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Error: Model file {MODEL_PATH} not found.")
            return

        logger.info(f"Running evaluation-only mode on model: {MODEL_PATH}")
        run_subprocess_eval(MODEL_PATH, num_episodes=10, render=True)
        return

    # Analysis mode - use subprocess
    if args.analyze:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Error: Model file {MODEL_PATH} not found.")
            return

        logger.info(f"Running analysis mode on model: {MODEL_PATH}")
        # Create a special eval subprocess command for analysis
        cmd = [
            sys.executable,  # Current Python interpreter
            "eval_subprocess.py",  # The evaluation script
            "--model",
            MODEL_PATH,
            "--episodes",
            "20",
            "--analyze",  # Special flag for analysis mode
        ]

        if args.render:
            cmd.append("--render")

        subprocess.run(cmd)
        return

    # Fixed frame stack size for better projectile detection
    frame_stack = 4

    try:
        # Create enhanced environment with projectile features always enabled
        logger.info(
            f"Creating enhanced Kung Fu environment with frame stacking (n_stack={frame_stack}) for projectile detection..."
        )

        # Force garbage collection before creating environment
        gc.collect()
        time.sleep(1)

        env = make_enhanced_kungfu_env(
            is_play_mode=args.render,
            frame_stack=frame_stack,
            use_projectile_features=True,  # Always use projectile features
        )

        # Register environment for cleanup
        active_environments.add(env)

        # Create or load model
        logger.info("Creating model with projectile feature support...")
        model = create_enhanced_kungfu_model(
            env, resume=args.resume, model_path=MODEL_PATH
        )

        # Train model
        train_model(model, args.timesteps, model_path=MODEL_PATH)

        # Plot metrics if requested
        if args.plot_metrics:
            plot_training_metrics()

        # Clean up
        logger.info("Closing training environment")
        env.close()
        active_environments.discard(env)
        logger.info("Training completed")

    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Make sure we clean up all resources
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
        # Make sure we clean up all resources on exit
        cleanup_all_resources()
