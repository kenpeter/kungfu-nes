import os
import argparse
import numpy as np
import logging
import time
import gc
import traceback
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("kungfu_eval")

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

# Import other needed modules
try:
    from stable_baselines3 import PPO
    from train import DFPPolicy
except ImportError as e:
    logger.error(f"Failed to import training modules: {e}")
    raise


def cleanup_env(env):
    """Clean up a single environment"""
    if env is not None:
        try:
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    # Force garbage collection
    gc.collect()

    # Final cleanup with environment manager
    try:
        RetroEnvManager.get_instance().cleanup_all_envs()
    except:
        pass


def evaluate_model(model_path=MODEL_PATH, num_episodes=10, render=True, analyze=False):
    """Evaluate the trained model"""
    env = None

    try:
        # Create environment for evaluation
        logger.info("Creating environment for evaluation")
        env = make_kungfu_env(is_play_mode=render, frame_stack=4, use_dfp=True)

        # Load model
        logger.info(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=env, policy=DFPPolicy)
        logger.info("Model loaded successfully")

        # Initialize metrics
        episode_rewards = []
        episode_scores = []
        episode_damages = []
        episode_progress = []
        max_stages = []
        episode_durations = []

        # Run evaluation episodes
        for episode in range(num_episodes):
            start_time = time.time()
            logger.info(f"Starting episode {episode+1}/{num_episodes}")

            obs = env.reset()
            done = False
            episode_reward = 0
            episode_score = 0
            episode_damage = 0
            episode_prog = 0
            current_stage = 0
            step_count = 0

            try:
                # Run episode
                while not done:
                    # Get action
                    action, _ = model.predict(obs, deterministic=True)

                    # Take step
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    step_count += 1

                    # Accumulate metrics
                    episode_reward += reward

                    # If info is a list, get the first element
                    if isinstance(info, list) and len(info) > 0:
                        info = info[0]

                    # Extract measurements
                    if isinstance(info, dict):
                        episode_score += info.get("score_increase", 0)
                        episode_damage += info.get("damage_taken", 0)
                        episode_prog += info.get("progress_made", 0)
                        current_stage = max(current_stage, info.get("current_stage", 0))

                # Record episode results
                duration = time.time() - start_time
                episode_rewards.append(episode_reward)
                episode_scores.append(episode_score)
                episode_damages.append(episode_damage)
                episode_progress.append(episode_prog)
                max_stages.append(current_stage)
                episode_durations.append(duration)

                logger.info(f"Episode {episode+1} completed")
                logger.info(f"  Reward: {episode_reward:.2f}")
                logger.info(f"  Score: {episode_score:.1f}")
                logger.info(f"  Damage: {episode_damage:.1f}")
                logger.info(f"  Progress: {episode_prog:.1f}")
                logger.info(f"  Stage: {current_stage}")
                logger.info(f"  Steps: {step_count}")
                logger.info(f"  Duration: {duration:.2f}s")

            except Exception as e:
                logger.error(f"Error during episode {episode+1}: {e}")
                logger.error(traceback.format_exc())
                # Try to continue with next episode
                continue

        # Calculate averages
        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            avg_score = np.mean(episode_scores)
            avg_damage = np.mean(episode_damages)
            avg_progress = np.mean(episode_progress)
            avg_stage = np.mean(max_stages)
            max_stage = np.max(max_stages)

            logger.info("\nEvaluation results:")
            logger.info(f"  Average reward: {avg_reward:.2f}")
            logger.info(f"  Average score: {avg_score:.1f}")
            logger.info(f"  Average damage: {avg_damage:.1f}")
            logger.info(f"  Average progress: {avg_progress:.1f}")
            logger.info(f"  Average stage: {avg_stage:.1f}")
            logger.info(f"  Max stage reached: {max_stage}")
        else:
            logger.warning("No episodes were successfully completed")

        return True

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return False

    finally:
        # Clean up the environment
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

    # Check if model file exists
    if not os.path.exists(args.model):
        logger.error(f"Model file {args.model} not found")
        return 1

    # Run evaluation
    try:
        success = evaluate_model(
            model_path=args.model,
            num_episodes=args.episodes,
            render=args.render,
            analyze=args.analyze,
        )
        return 0 if success else 1

    except Exception as e:
        logger.error(f"Unhandled exception in evaluation: {e}")
        logger.error(traceback.format_exc())
        return 1


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
