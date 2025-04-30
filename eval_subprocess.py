import os
import argparse
import numpy as np
import logging
import time
import gc
import traceback
import sys

# Configure logging to both console and file for persistent records
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("kungfu_eval.log", mode="a"),
    ],
)
logger = logging.getLogger("kungfu_eval")

# Import custom environment
try:
    from kung_fu_env import make_kungfu_env, MODEL_PATH, RetroEnvManager
except ImportError as e:
    logger.error(f"Failed to import kung_fu_env: {e}")
    raise

# Import training modules
try:
    from stable_baselines3 import PPO
    from train import DFPPolicy
except ImportError as e:
    logger.error(f"Failed to import training modules: {e}")
    raise


def cleanup_env(env):
    """Clean up the environment and release resources."""
    if env is not None:
        try:
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    # Force garbage collection to prevent memory leaks
    gc.collect()

    # Final cleanup with environment manager
    try:
        RetroEnvManager.get_instance().cleanup_all_envs()
        logger.info("RetroEnvManager cleanup completed")
    except Exception as e:
        logger.warning(f"Failed to cleanup RetroEnvManager: {e}")


def evaluate_model(model_path=MODEL_PATH, num_episodes=10, render=False, analyze=False):
    """
    Evaluate the trained DFP model and identify potential issues.

    Args:
        model_path (str): Path to the trained model file.
        num_episodes (int): Number of episodes to run.
        render (bool): Whether to render the environment.
        analyze (bool): Whether to perform detailed analysis.

    Returns:
        bool: True if evaluation completed successfully, False otherwise.
    """
    env = None

    try:
        # Create environment
        logger.info("Creating evaluation environment")
        env = make_kungfu_env(is_play_mode=render, frame_stack=4, use_dfp=True)
        logger.info(f"Environment observation space: {env.observation_space}")

        # Load model
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, policy=DFPPolicy)
        logger.info("Model loaded successfully")
        logger.info(f"Model policy observation space: {model.policy.observation_space}")

        # Verify observation space compatibility
        if model.policy.observation_space != env.observation_space:
            logger.warning(
                "Observation space mismatch between model and environment. This may cause prediction errors."
            )

        # Initialize metrics
        metrics = {
            "rewards": [],
            "scores": [],
            "damages": [],
            "progress": [],
            "max_stages": [],
            "durations": [],
            "steps": [],
            "actions": [],  # Track actions for analysis
        }

        # Run evaluation episodes
        for episode in range(num_episodes):
            start_time = time.time()
            logger.info(f"Starting episode {episode+1}/{num_episodes}")

            # Handle vectorized environment reset
            obs = env.reset()
            if isinstance(obs, list) or isinstance(obs, np.ndarray):
                obs = obs[0]  # Extract single observation if vectorized
            done = False
            episode_metrics = {
                "reward": 0,
                "score": 0,
                "damage": 0,
                "progress": 0,
                "max_stage": 0,
                "steps": 0,
                "actions": [],
            }

            try:
                while not done:
                    # Predict action
                    action, _ = model.predict(obs, deterministic=True)
                    episode_metrics["actions"].append(action)

                    # Take step in environment
                    step_result = env.step(action)
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    episode_metrics["steps"] += 1

                    # Handle vectorized outputs
                    if isinstance(obs, list) or isinstance(obs, np.ndarray):
                        obs = obs[0]
                    if isinstance(reward, list) or isinstance(reward, np.ndarray):
                        reward = reward[0]
                    if isinstance(info, list):
                        info = info[0]

                    # Validate info format
                    if not isinstance(info, dict):
                        logger.warning(
                            f"Episode {episode+1}: info is not a dict: {info}"
                        )
                        info = {}

                    # Accumulate metrics
                    episode_metrics["reward"] += float(reward)
                    episode_metrics["score"] += info.get("score_increase", 0)
                    episode_metrics["damage"] += info.get("damage_taken", 0)
                    episode_metrics["progress"] += info.get("progress_made", 0)
                    episode_metrics["max_stage"] = max(
                        episode_metrics["max_stage"], info.get("current_stage", 0)
                    )

                    if render:
                        env.render()

                # Record episode results
                duration = time.time() - start_time
                for key in episode_metrics:
                    if (
                        key != "max_stage" or episode_metrics[key] > 0
                    ):  # Only append non-zero max stages
                        metrics[key + "s"].append(episode_metrics[key])
                metrics["durations"].append(duration)

                logger.info(f"Episode {episode+1} completed")
                logger.info(f"  Reward: {episode_metrics['reward']:.2f}")
                logger.info(f"  Score: {episode_metrics['score']:.1f}")
                logger.info(f"  Damage: {episode_metrics['damage']:.1f}")
                logger.info(f"  Progress: {episode_metrics['progress']:.1f}")
                logger.info(f"  Max Stage: {episode_metrics['max_stage']}")
                logger.info(f"  Steps: {episode_metrics['steps']}")
                logger.info(f"  Duration: {duration:.2f}s")

            except Exception as e:
                logger.error(f"Error during episode {episode+1}: {e}")
                logger.error(traceback.format_exc())
                continue

        # Calculate and log averages
        if metrics["rewards"]:
            averages = {
                key: np.mean(values) for key, values in metrics.items() if values
            }
            logger.info("\nEvaluation Results:")
            logger.info(f"  Average Reward: {averages.get('rewards', 0):.2f}")
            logger.info(f"  Average Score: {averages.get('scores', 0):.1f}")
            logger.info(f"  Average Damage: {averages.get('damages', 0):.1f}")
            logger.info(f"  Average Progress: {averages.get('progress', 0):.1f}")
            logger.info(f"  Average Max Stage: {averages.get('max_stages', 0):.1f}")
            logger.info(
                f"  Max Stage Reached: {np.max(metrics['max_stages']) if metrics['max_stages'] else 0}"
            )
            logger.info(f"  Average Steps: {averages.get('steps', 0):.1f}")
            logger.info(f"  Average Duration: {averages.get('durations', 0):.2f}s")

            # Detailed analysis if requested
            if analyze:
                logger.info("\nDetailed Analysis:")
                logger.info(
                    f"  Action Distribution: {np.unique(metrics['actions'], return_counts=True)}"
                )
                if np.std(metrics["rewards"]) > averages.get("rewards", 0) * 0.5:
                    logger.warning(
                        "High variance in rewards; model may be inconsistent."
                    )
                if averages.get("max_stages", 0) < 2:
                    logger.warning("Model struggles to progress beyond early stages.")
        else:
            logger.warning("No episodes completed successfully")

        return True

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return False

    finally:
        cleanup_env(env)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained DFP Kung Fu Master model"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to model file"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--analyze", action="store_true", help="Run detailed analysis")
    args = parser.parse_args()

    # Validate model file
    if not os.path.exists(args.model):
        logger.error(f"Model file {args.model} not found")
        sys.exit(1)

    # Run evaluation
    try:
        success = evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            render=args.render,
            analyze=args.analyze,
        )
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Unhandled exception in evaluation: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
