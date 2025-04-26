import os
import argparse
import time
from stable_baselines3 import PPO
from kungfu_env import make_kungfu_env, MODEL_PATH, KUNGFU_ACTION_NAMES


def play_game(model_path=None, human_play=False, num_episodes=5):
    """Play Kung Fu Master with the trained model or human controls"""

    # Create environment with rendering
    env = make_kungfu_env(is_play_mode=True)

    # Load model if not human play
    model = None
    if not human_play and model_path:
        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model = PPO.load(model_path)
        else:
            print(f"Model file {model_path} not found")
            return

    for episode in range(num_episodes):
        print(f"\n--- Starting Episode {episode+1}/{num_episodes} ---")

        # Reset environment
        obs = env.reset()[0]
        done = False
        episode_reward = 0
        step = 0
        start_time = time.time()

        # Game loop
        while not done:
            if not human_play and model:
                # Model play
                action, _ = model.predict(obs, deterministic=True)
                print(f"Step {step}: Action = {KUNGFU_ACTION_NAMES[action[0]]}")
            else:
                # Human play (pauses for input - for testing only)
                print("\nAvailable actions:")
                for i, action_name in enumerate(KUNGFU_ACTION_NAMES):
                    print(f"{i}: {action_name}")
                try:
                    action_idx = int(input("\nEnter action number: "))
                    action = [action_idx]
                except ValueError:
                    print("Invalid input, using No-op")
                    action = [0]

            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            # Display info
            if step % 30 == 0:  # Show info every ~1 second
                elapsed = time.time() - start_time
                if "current_stage" in info:
                    stage = info["current_stage"]
                else:
                    stage = "unknown"

                if "current_score" in info:
                    score = info["current_score"]
                else:
                    score = "unknown"

                if "time_remaining" in info:
                    time_left = info["time_remaining"]
                    print(
                        f"Stage: {stage}, Score: {score}, Time left: {time_left:.1f}s, Reward: {episode_reward:.1f}"
                    )
                else:
                    print(
                        f"Stage: {stage}, Score: {score}, Reward: {episode_reward:.1f}"
                    )

            done = terminated or truncated
            step += 1

            # Add small delay to make it watchable
            time.sleep(0.01)

        # Episode summary
        duration = time.time() - start_time
        print(f"\nEpisode {episode+1} finished after {step} steps")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Duration: {duration:.2f} seconds")

    # Close environment
    env.close()
    print("\nGame closed")


def main():
    parser = argparse.ArgumentParser(description="Play Kung Fu Master with trained AI")
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Path to the trained model"
    )
    parser.add_argument("--human", action="store_true", help="Enable human play mode")
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to play"
    )
    args = parser.parse_args()

    play_game(model_path=args.model, human_play=args.human, num_episodes=args.episodes)


if __name__ == "__main__":
    main()
