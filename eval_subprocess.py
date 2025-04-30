#!/usr/bin/env python3
"""
Subprocess for evaluating a Kung Fu Master agent.

This script is designed to be run as a separate process to guarantee
clean environment creation without conflicting with any existing retro
emulator instances.
"""

import os
import argparse
import numpy as np
import logging
import time
import sys
import traceback
import gc
import torch
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="eval_subprocess.log",
)
logger = logging.getLogger("eval_subprocess")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

# import env and create model
try:
    from kung_fu_env import (
        make_enhanced_kungfu_env,
        create_enhanced_kungfu_model,
        MODEL_PATH,
        RetroEnvManager,
    )
except ImportError as e:
    logger.error(f"Failed to import kung_fu_env: {e}")
    sys.exit(1)


# we eval our model
def evaluate_model(model, n_eval_episodes=5, env=None, is_analysis=False):
    """Evaluate model with focus on projectile avoidance and stage progress"""
    logger.info(f"Evaluating model over {n_eval_episodes} episodes...")

    # Create a new environment if one wasn't provided
    eval_env = env
    env_created = False

    try:
        if eval_env is None:
            logger.info("Creating environment for evaluation")
            eval_env = make_enhanced_kungfu_env(
                is_play_mode=True, frame_stack=4, use_projectile_features=True
            )
            env_created = True

        # Metrics
        mean_reward = 0
        mean_stages_cleared = 0
        total_projectiles_detected = 0
        total_defensive_actions = 0
        total_successful_defensive_actions = 0
        total_projectile_defensive_actions = 0
        total_successful_projectile_avoidance = 0
        total_breakthrough_rewards = 0
        total_offensive_actions = 0
        total_successful_offensive_actions = 0
        total_stagnation = 0

        episode_stats = []
        stuck_positions = []
        death_positions = []

        # Track actions for analysis
        action_frequencies = {
            name: 0
            for name in ["No-op", "Punch", "Jump", "Crouch", "Left", "Right", "Kick"]
        }

        for i in range(n_eval_episodes):
            try:
                logger.info(f"Starting evaluation episode {i+1}/{n_eval_episodes}")

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
                max_x_position = 0
                min_x_position = 255
                consecutive_same_position = 0
                last_position = None

                # Episode metrics
                episode_projectiles = 0
                episode_defensive_actions = 0
                episode_successful_defensive = 0
                episode_projectile_actions = 0
                episode_successful_projectile = 0
                episode_breakthroughs = 0
                episode_offensive_actions = 0
                episode_successful_offensive = 0
                episode_stagnation = 0

                # Position tracking for analysis
                position_counts = {}

                # Run episode
                while not done:
                    action, _ = model.predict(obs, deterministic=True)

                    # Track action frequency for analysis
                    if is_analysis:
                        action_name = "Unknown"
                        if action < len(
                            [
                                "No-op",
                                "Punch",
                                "Select",
                                "Start",
                                "Jump",
                                "Crouch",
                                "Left",
                                "Right",
                                "Kick",
                            ]
                        ):
                            action_name = [
                                "No-op",
                                "Punch",
                                "Select",
                                "Start",
                                "Jump",
                                "Crouch",
                                "Left",
                                "Right",
                                "Kick",
                            ][action]
                            if action_name in action_frequencies:
                                action_frequencies[action_name] += 1

                    # Take step in environment
                    step_result = eval_env.step(action)

                    # Handle different return formats
                    if len(step_result) == 5:
                        # Gymnasium format
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        # Old gym format
                        obs, reward, done, info = step_result
                    else:
                        # Unexpected format
                        logger.warning(
                            f"Unexpected step result format with {len(step_result)} values"
                        )
                        break

                    # Ensure info is a dictionary, not a list
                    if isinstance(info, list):
                        logger.warning(
                            f"Got info as a list instead of a dict, converting to empty dict"
                        )
                        info = {}

                    # Update metrics
                    total_reward += reward
                    steps += 1

                    # Get current stage and position
                    current_stage = info.get("current_stage", 0)
                    if current_stage > max_stage:
                        max_stage = current_stage

                    # Get player position from info
                    current_position = None
                    if hasattr(eval_env, "get_player_position"):
                        try:
                            current_position = eval_env.get_player_position()
                        except:
                            # Fall back to unwrapped env
                            if hasattr(eval_env.unwrapped, "get_player_position"):
                                current_position = (
                                    eval_env.unwrapped.get_player_position()
                                )

                    # Position tracking for analysis
                    if is_analysis and current_position:
                        x_pos, y_pos = current_position
                        position_key = f"{x_pos}_{y_pos}"
                        position_counts[position_key] = (
                            position_counts.get(position_key, 0) + 1
                        )

                        # Track min/max positions
                        if x_pos > max_x_position:
                            max_x_position = x_pos
                        if x_pos < min_x_position:
                            min_x_position = x_pos

                        # Track if agent is stuck
                        if last_position == current_position:
                            consecutive_same_position += 1
                            if consecutive_same_position > 50:  # Stuck for 50+ frames
                                if position_key not in [
                                    pos[0] for pos in stuck_positions
                                ]:
                                    stuck_positions.append(
                                        (
                                            position_key,
                                            (x_pos, y_pos),
                                            consecutive_same_position,
                                        )
                                    )
                        else:
                            consecutive_same_position = 0

                        last_position = current_position

                    # Track projectile metrics - safely access info dict
                    if isinstance(info, dict):
                        if (
                            "detected_projectiles" in info
                            and info["detected_projectiles"] > 0
                        ):
                            episode_projectiles += 1

                        if "defensive_actions" in info:
                            episode_defensive_actions += info["defensive_actions"]

                        if "successful_defensive_actions" in info:
                            episode_successful_defensive += info[
                                "successful_defensive_actions"
                            ]

                        if "projectile_defensive_actions" in info:
                            episode_projectile_actions += info[
                                "projectile_defensive_actions"
                            ]

                        if "successful_projectile_avoidance" in info:
                            episode_successful_projectile += info[
                                "successful_projectile_avoidance"
                            ]

                        if "breakthrough_rewards_given" in info:
                            episode_breakthroughs += info["breakthrough_rewards_given"]

                        if "offensive_actions" in info:
                            episode_offensive_actions += info["offensive_actions"]

                        if "successful_offensive_actions" in info:
                            episode_successful_offensive += info[
                                "successful_offensive_actions"
                            ]

                        if "stagnation_counter" in info:
                            episode_stagnation = max(
                                episode_stagnation, info["stagnation_counter"]
                            )

                    # Check if the agent died - useful for analysis
                    if done and is_analysis and isinstance(info, dict):
                        current_death_stage = info.get("current_stage", current_stage)
                        if current_death_stage == current_stage and current_position:
                            x_pos, y_pos = current_position
                            death_positions.append((x_pos, y_pos, current_stage))

                # Calculate episode metrics
                episode_length = steps
                episode_reward = total_reward

                # Calculate success rates
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

                offensive_success_rate = 0
                if episode_offensive_actions > 0:
                    offensive_success_rate = (
                        episode_successful_offensive / episode_offensive_actions
                    ) * 100

                # Log episode results
                logger.info(
                    f"Episode {i+1} finished: Reward={episode_reward:.2f}, Steps={episode_length}, "
                    f"Max Stage={max_stage}, Projectiles={episode_projectiles}, "
                    f"Avoidance Rate={projectile_avoidance_rate:.1f}%, "
                    f"Breakthroughs={episode_breakthroughs}"
                )

                # Store episode stats
                episode_stats.append(
                    {
                        "episode": i + 1,
                        "reward": episode_reward,
                        "length": episode_length,
                        "max_stage": max_stage,
                        "projectiles_detected": episode_projectiles,
                        "defensive_actions": episode_defensive_actions,
                        "defensive_success_rate": defensive_success_rate,
                        "projectile_actions": episode_projectile_actions,
                        "projectile_avoidance_rate": projectile_avoidance_rate,
                        "breakthroughs": episode_breakthroughs,
                        "offensive_actions": episode_offensive_actions,
                        "offensive_success_rate": offensive_success_rate,
                        "stagnation": episode_stagnation,
                        "max_x_position": max_x_position,
                        "min_x_position": min_x_position,
                    }
                )

                # Update overall metrics
                mean_reward += episode_reward
                mean_stages_cleared += max_stage
                total_projectiles_detected += episode_projectiles
                total_defensive_actions += episode_defensive_actions
                total_successful_defensive_actions += episode_successful_defensive
                total_projectile_defensive_actions += episode_projectile_actions
                total_successful_projectile_avoidance += episode_successful_projectile
                total_breakthrough_rewards += episode_breakthroughs
                total_offensive_actions += episode_offensive_actions
                total_successful_offensive_actions += episode_successful_offensive
                total_stagnation += episode_stagnation

            except Exception as e:
                logger.error(f"Error during evaluation episode {i+1}: {e}")
                logger.error(traceback.format_exc())

        # Calculate final metrics
        if n_eval_episodes > 0:
            mean_reward /= n_eval_episodes
            mean_stages_cleared /= n_eval_episodes

            defensive_success_rate = 0
            if total_defensive_actions > 0:
                defensive_success_rate = (
                    total_successful_defensive_actions / total_defensive_actions
                ) * 100

            projectile_avoidance_rate = 0
            if total_projectile_defensive_actions > 0:
                projectile_avoidance_rate = (
                    total_successful_projectile_avoidance
                    / total_projectile_defensive_actions
                ) * 100

            offensive_success_rate = 0
            if total_offensive_actions > 0:
                offensive_success_rate = (
                    total_successful_offensive_actions / total_offensive_actions
                ) * 100

            mean_stagnation = total_stagnation / n_eval_episodes

            # Log overall results
            logger.info("=" * 50)
            logger.info(f"Evaluation Results ({n_eval_episodes} episodes):")
            logger.info(f"Mean Reward: {mean_reward:.2f}")
            logger.info(f"Mean Stages Cleared: {mean_stages_cleared:.2f}")
            logger.info(f"Total Projectiles Detected: {total_projectiles_detected}")
            logger.info(f"Defensive Success Rate: {defensive_success_rate:.1f}%")
            logger.info(f"Projectile Avoidance Rate: {projectile_avoidance_rate:.1f}%")
            logger.info(f"Breakthrough Rewards: {total_breakthrough_rewards}")
            logger.info(f"Offensive Success Rate: {offensive_success_rate:.1f}%")
            logger.info(f"Mean Stagnation: {mean_stagnation:.1f}")

            # Create detailed analysis if requested
            if is_analysis:
                logger.info("=" * 50)
                logger.info("Detailed Agent Analysis:")

                # Action distribution
                logger.info("Action Distribution:")
                total_actions = sum(action_frequencies.values())
                if total_actions > 0:
                    for action, count in action_frequencies.items():
                        percentage = (count / total_actions) * 100
                        logger.info(f"  {action}: {count} ({percentage:.1f}%)")

                # Stuck positions analysis
                if stuck_positions:
                    logger.info("Positions where agent got stuck:")
                    for pos_key, pos, duration in sorted(
                        stuck_positions, key=lambda x: x[2], reverse=True
                    )[:10]:
                        logger.info(
                            f"  Position ({pos[0]}, {pos[1]}): Stuck for {duration} frames"
                        )

                # Death positions analysis
                if death_positions:
                    logger.info("Positions where agent died:")
                    stage_deaths = {}
                    for x, y, stage in death_positions:
                        if stage not in stage_deaths:
                            stage_deaths[stage] = []
                        stage_deaths[stage].append((x, y))

                    for stage, positions in stage_deaths.items():
                        logger.info(f"  Stage {stage}: {len(positions)} deaths")
                        # Group by nearby positions
                        position_clusters = {}
                        for x, y in positions:
                            # Round to nearest 10 pixels for clustering
                            cluster_key = (round(x / 10) * 10, round(y / 10) * 10)
                            if cluster_key not in position_clusters:
                                position_clusters[cluster_key] = 0
                            position_clusters[cluster_key] += 1

                        # Show most common death spots
                        for (x, y), count in sorted(
                            position_clusters.items(), key=lambda x: x[1], reverse=True
                        )[:5]:
                            logger.info(
                                f"    Around position ({x}, {y}): {count} deaths"
                            )

                # Save evaluation data for visualization
                try:
                    # Create directories if they don't exist
                    os.makedirs("logs/eval", exist_ok=True)

                    # Save episode stats to CSV
                    episodes_df = pd.DataFrame(episode_stats)
                    episodes_df.to_csv("logs/eval/episode_stats.csv", index=False)

                    # Create basic plots
                    plt.figure(figsize=(12, 10))

                    # Plot rewards
                    plt.subplot(2, 2, 1)
                    plt.plot(
                        [e["episode"] for e in episode_stats],
                        [e["reward"] for e in episode_stats],
                        marker="o",
                    )
                    plt.title("Episode Rewards")
                    plt.xlabel("Episode")
                    plt.ylabel("Reward")
                    plt.grid(True)

                    # Plot max stages
                    plt.subplot(2, 2, 2)
                    plt.plot(
                        [e["episode"] for e in episode_stats],
                        [e["max_stage"] for e in episode_stats],
                        marker="o",
                    )
                    plt.title("Max Stage Reached")
                    plt.xlabel("Episode")
                    plt.ylabel("Stage")
                    plt.grid(True)

                    # Plot projectile avoidance
                    plt.subplot(2, 2, 3)
                    plt.plot(
                        [e["episode"] for e in episode_stats],
                        [e["projectile_avoidance_rate"] for e in episode_stats],
                        marker="o",
                    )
                    plt.title("Projectile Avoidance Rate")
                    plt.xlabel("Episode")
                    plt.ylabel("Avoidance Rate (%)")
                    plt.grid(True)

                    # Plot action distribution
                    plt.subplot(2, 2, 4)
                    actions = list(action_frequencies.keys())
                    counts = list(action_frequencies.values())
                    plt.bar(actions, counts)
                    plt.title("Action Distribution")
                    plt.ylabel("Count")
                    plt.xticks(rotation=45)

                    plt.tight_layout()
                    plt.savefig("logs/eval/evaluation_analysis.png")

                    # Save to evaluation metrics log if it exists
                    eval_metrics_file = "logs/eval/eval_metrics.csv"
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_step = 0  # We don't know the current training step

                    try:
                        # Try to get current step from model path
                        if model_path and "_step_" in model_path:
                            step_part = model_path.split("_step_")[1].split(".")[0]
                            current_step = int(step_part)
                    except:
                        pass

                    eval_data = {
                        "timestamp": timestamp,
                        "steps": current_step,
                        "mean_reward": mean_reward,
                        "max_stage": mean_stages_cleared,
                        "projectile_avoidance_rate": projectile_avoidance_rate,
                        "defensive_success_rate": defensive_success_rate,
                        "offensive_success_rate": offensive_success_rate,
                        "breakthroughs": total_breakthrough_rewards / n_eval_episodes,
                        "stagnation": mean_stagnation,
                    }

                    # Check if file exists
                    file_exists = os.path.exists(eval_metrics_file)

                    # Write to CSV
                    with open(eval_metrics_file, "a") as f:
                        if not file_exists:
                            # Write header
                            f.write(",".join(eval_data.keys()) + "\n")

                        # Write values
                        f.write(",".join(str(val) for val in eval_data.values()) + "\n")

                    logger.info(f"Evaluation analysis saved to logs/eval/ directory")

                except Exception as e:
                    logger.error(f"Error saving evaluation analysis: {e}")
                    logger.error(traceback.format_exc())

            # Return evaluation results as a dictionary
            results = {
                "mean_reward": mean_reward,
                "mean_stages_cleared": mean_stages_cleared,
                "defensive_success_rate": defensive_success_rate,
                "projectile_avoidance_rate": projectile_avoidance_rate,
                "offensive_success_rate": offensive_success_rate,
                "breakthroughs": total_breakthrough_rewards / n_eval_episodes,
                "stagnation": mean_stagnation,
            }

            return results

        else:
            logger.warning("No episodes were evaluated successfully")
            return {
                "mean_reward": 0,
                "mean_stages_cleared": 0,
                "defensive_success_rate": 0,
                "projectile_avoidance_rate": 0,
                "offensive_success_rate": 0,
                "breakthroughs": 0,
                "stagnation": 0,
            }

    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        logger.error(traceback.format_exc())
        return None

    finally:
        # Close the environment if we created it
        if env_created and eval_env is not None:
            try:
                eval_env.close()
                logger.info("Evaluation environment closed")
            except Exception as e:
                logger.error(f"Error closing evaluation environment: {e}")


def main():
    """Main function to run model evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate Kung Fu Master agent")
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to model file"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during evaluation"
    )
    parser.add_argument("--analyze", action="store_true", help="Run detailed analysis")
    args = parser.parse_args()

    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Create clean environment manager for this process
    env_manager = RetroEnvManager.get_instance()

    try:
        # Force garbage collection
        gc.collect()
        time.sleep(1)

        # Create environment with rendering if specified
        logger.info(f"Creating environment with render={args.render}")
        env = make_enhanced_kungfu_env(
            is_play_mode=args.render,
            frame_stack=4,
            use_projectile_features=True,  # Always use features for evaluation
        )

        # Register for cleanup
        env_manager.register_env(env)

        # Verify model file exists
        if not os.path.exists(args.model):
            logger.error(f"Model file not found: {args.model}")
            sys.exit(1)

        # Load the model
        logger.info(f"Loading model from {args.model}")
        model = create_enhanced_kungfu_model(env, resume=True, model_path=args.model)

        # Evaluate
        logger.info(f"Starting evaluation for {args.episodes} episodes")
        results = evaluate_model(
            model, n_eval_episodes=args.episodes, env=env, is_analysis=args.analyze
        )

        if results:
            logger.info("Evaluation completed successfully")
        else:
            logger.error("Evaluation failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    finally:
        # Clean up resources
        logger.info("Cleaning up resources")
        try:
            env_manager.cleanup_all_envs()
        except:
            pass

        # Force garbage collection
        gc.collect()


if __name__ == "__main__":
    main()
