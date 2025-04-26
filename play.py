import os
import argparse
import numpy as np
import time
from stable_baselines3 import PPO
from kungfu_env import make_kungfu_env, MODEL_PATH, KUNGFU_ACTION_NAMES


def play_game(env, model, episodes=5, render_sleep=0.01):
    """
    Play the game using the trained model.

    Args:
        env: The game environment
        model: The trained PPO model
        episodes: Number of game episodes to play
        render_sleep: Sleep time between frames to control game speed (if applicable)
    """
    print(f"Playing Kung Fu Master for {episodes} episodes...")

    for episode in range(episodes):
        print(f"\nEpisode {episode+1}/{episodes}")
        obs, _ = env.reset()
        done = False
        terminated = [False]
        truncated = [False]
        episode_reward = 0
        step_count = 0

        # Game loop
        while not any(terminated) and not any(truncated):
            # Get action from the model
            action, _ = model.predict(obs, deterministic=True)

            # Print the action being taken (for debugging/viewing)
            if step_count % 30 == 0:  # Print every 30 steps to avoid spam
                print(f"Taking action: {KUNGFU_ACTION_NAMES[action[0]]}")

            # Execute action in the environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward[0]
            step_count += 1

            # Optional sleep to slow down rendering for viewing
            if render_sleep > 0:
                time.sleep(render_sleep)

        print(f"Episode {episode+1} finished after {step_count} steps")
        print(f"Total reward: {episode_reward:.2f}")

        if "current_stage" in info[0]:
            print(f"Reached stage: {info[0]['current_stage']}")


def main():
    parser = argparse.ArgumentParser(
        description="Play Kung Fu Master with a trained AI model"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to the trained model"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.01,
        help="Control game speed (sleep time between frames)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Please train a model first using train.py")
        return

    # Create environment in play mode (with rendering)
    env = make_kungfu_env(is_play_mode=True)

    # Load trained model
    print(f"Loading model from {args.model}")
    model = PPO.load(args.model, env=env)

    # Play the game
    try:
        play_game(env, model, episodes=args.episodes, render_sleep=args.speed)
    except KeyboardInterrupt:
        print("\nPlay session interrupted by user")
    finally:
        env.close()
        print("Play session ended")


if __name__ == "__main__":
    main()
