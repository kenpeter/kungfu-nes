import os
import argparse
import numpy as np
import torch
import logging
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
import gymnasium as gym
import sys

# Import local modules
try:
    from kung_fu_env import make_kungfu_env, RetroEnvManager, MODEL_PATH
    from train import DFPPolicy, cleanup_all_resources
except ImportError as e:
    print(f"Error importing local modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="kungfu_eval.log",
)
logger = logging.getLogger("kungfu_eval")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


def evaluate_model(model_path, num_episodes=10, render=True, analyze=False):
    """
    Evaluate a trained model on the Kung Fu Master environment.

    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to evaluate
        render: Whether to render the environment during evaluation
        analyze: Whether to analyze agent behavior
    """
    logger.info("Creating evaluation environment")
    env = make_kungfu_env(is_play_mode=render, frame_stack=4, use_dfp=True)

    logger.info(f"Environment observation space: {env.observation_space}")

    try:
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, policy=DFPPolicy)
        logger.info("Model loaded successfully")
        logger.info(f"Model policy observation space: {model.policy.observation_space}")

        # Verify observation space compatibility
        if str(model.policy.observation_space) != str(env.observation_space):
            logger.warning(
                "Observation space mismatch between model and environment. This may cause prediction errors."
            )

        # Initialize metrics
        metrics = {
            "rewards": [],
            "scores": [],
            "damages": [],
            "progresses": [],  # Fixed typo: changed from 'progresss'
            "max_stages": [],
            "steps": [],
            "durations": [],
        }

        max_stage = 0

        for episode in range(1, num_episodes + 1):
            logger.info(f"Starting episode {episode}/{num_episodes}")
            episode_metrics = {
                "reward": 0,
                "score": 0,
                "damage": 0,
                "progress": 0,  # Fixed typo
                "max_stage": 0,
                "steps": 0,
            }

            obs = env.reset()
            # Handle different reset() return types
            if isinstance(obs, tuple):
                obs, info = obs
            else:
                info = {}

            episode_start_time = time.time()
            done = False

            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=True)

                    # Step the environment
                    step_result = env.step(action)

                    # Handle both 4-element and 5-element returns for compatibility
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        # For older gym versions
                        obs, reward, done, info = step_result
                        terminated, truncated = done, False

                    episode_metrics["reward"] += reward
                    episode_metrics["steps"] += 1

                    # Extract info from the environment
                    # Check if info is a dict before accessing
                    if isinstance(info, dict):
                        episode_metrics["score"] = info.get("current_score", 0)
                        episode_metrics["damage"] += info.get("damage_taken", 0)
                        episode_metrics["progress"] += info.get("progress_made", 0)
                        episode_metrics["max_stage"] = max(
                            episode_metrics["max_stage"], info.get("current_stage", 0)
                        )

                except Exception as e:
                    logger.error(f"Error during episode {episode}: {e}")
                    logger.error(f"Traceback: {sys.exc_info()}")
                    break

            # Record episode metrics
            episode_duration = time.time() - episode_start_time
            metrics["rewards"].append(episode_metrics["reward"])
            metrics["scores"].append(episode_metrics["score"])
            metrics["damages"].append(episode_metrics["damage"])
            metrics["progresses"].append(episode_metrics["progress"])
            metrics["max_stages"].append(episode_metrics["max_stage"])
            metrics["steps"].append(episode_metrics["steps"])
            metrics["durations"].append(episode_duration)

            max_stage = max(max_stage, episode_metrics["max_stage"])

            logger.info(
                f"Episode {episode} - "
                f"Reward: {episode_metrics['reward']:.2f}, "
                f"Score: {episode_metrics['score']}, "
                f"Damage: {episode_metrics['damage']}, "
                f"Progress: {episode_metrics['progress']}, "
                f"Stage: {episode_metrics['max_stage']}, "
                f"Steps: {episode_metrics['steps']}, "
                f"Duration: {episode_duration:.2f}s"
            )

        # Calculate and log average metrics
        if len(metrics["rewards"]) > 0:
            logger.info("\nEvaluation Results:")
            logger.info(f"  Average Reward: {np.mean(metrics['rewards']):.2f}")
            logger.info(f"  Average Score: {np.mean(metrics['scores']):.1f}")
            logger.info(f"  Average Damage: {np.mean(metrics['damages']):.1f}")
            logger.info(f"  Average Progress: {np.mean(metrics['progresses']):.1f}")
            logger.info(f"  Average Max Stage: {np.mean(metrics['max_stages']):.1f}")
            logger.info(f"  Max Stage Reached: {max_stage}")
            logger.info(f"  Average Steps: {np.mean(metrics['steps']):.1f}")
            logger.info(f"  Average Duration: {np.mean(metrics['durations']):.2f}s")

            if analyze:
                analyze_performance(metrics)
        else:
            logger.warning("No episodes completed successfully")

    except Exception as e:
        logger.error(f"Error in evaluate_model: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        try:
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

        # Clean up with RetroEnvManager
        RetroEnvManager.get_instance().cleanup_all_envs()
        logger.info("RetroEnvManager cleanup completed")


def analyze_performance(metrics):
    """
    Analyze agent performance metrics
    """
    try:
        # Create a plots directory if it doesn't exist
        os.makedirs("logs/plots", exist_ok=True)

        # Plot reward over episodes
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(range(1, len(metrics["rewards"]) + 1), metrics["rewards"], "b-")
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True)

        # Plot score over episodes
        plt.subplot(2, 2, 2)
        plt.plot(range(1, len(metrics["scores"]) + 1), metrics["scores"], "g-")
        plt.title("Score per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.grid(True)

        # Plot damage over episodes
        plt.subplot(2, 2, 3)
        plt.plot(range(1, len(metrics["damages"]) + 1), metrics["damages"], "r-")
        plt.title("Damage Taken per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Damage")
        plt.grid(True)

        # Plot progress over episodes
        plt.subplot(2, 2, 4)
        plt.plot(range(1, len(metrics["progresses"]) + 1), metrics["progresses"], "m-")
        plt.title("Progress per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Progress")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig("logs/plots/evaluation_metrics.png")
        plt.close()

        logger.info(
            "Performance analysis plots saved to logs/plots/evaluation_metrics.png"
        )
    except Exception as e:
        logger.error(f"Error in analyze_performance: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Kung Fu Master DFP model"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to the model file"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during evaluation"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze agent performance"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        logger.error(f"Model file not found: {args.model}")
        return

    try:
        evaluate_model(args.model, args.episodes, args.render, args.analyze)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        cleanup_all_resources()


if __name__ == "__main__":
    main()
