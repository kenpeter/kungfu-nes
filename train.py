import os
import argparse
import numpy as np
import torch
import datetime
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os.path
from kungfu_env import make_kungfu_env, MODEL_PATH


class SaveCallback(BaseCallback):
    """Callback for saving the model periodically during training"""

    def __init__(self, check_freq=10000, log_freq=1000, log_dir="logs"):
        super().__init__()
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.best_mean_reward = -float("inf")
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize CSV log file for training metrics
        self.metrics_file = os.path.join(log_dir, "training_metrics.csv")

        # Check if file exists, if not create header
        if not os.path.exists(self.metrics_file):
            with open(self.metrics_file, "w") as f:
                f.write("timestamp,steps,mean_reward,defensive_success_rate\n")

    def _on_step(self):
        # Save model periodically
        if self.n_calls % self.check_freq == 0:
            path = f"{MODEL_PATH.split('.')[0]}_{self.n_calls}.zip"
            self.model.save(path)
            print(f"Model saved to {path}")

        # Log training info periodically
        if self.n_calls % self.log_freq == 0:
            # Get metrics from the training buffer
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)

            # Try to get defensive success rates from episode info buffer
            defensive_success_rate = 0
            defensive_actions = 0
            successful_defensive_actions = 0

            if len(self.model.ep_info_buffer) > 0:
                # Calculate average defensive success rate across episodes
                rates = [
                    ep_info.get("defensive_success_rate", 0)
                    for ep_info in self.model.ep_info_buffer
                    if "defensive_success_rate" in ep_info
                ]
                actions = [
                    ep_info.get("defensive_actions", 0)
                    for ep_info in self.model.ep_info_buffer
                    if "defensive_actions" in ep_info
                ]
                successes = [
                    ep_info.get("successful_defensive_actions", 0)
                    for ep_info in self.model.ep_info_buffer
                    if "successful_defensive_actions" in ep_info
                ]

                if rates:
                    defensive_success_rate = np.mean(rates)
                if actions:
                    defensive_actions = np.mean(actions)
                if successes:
                    successful_defensive_actions = np.mean(successes)

            # Log to console with defensive timing metrics
            print(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Defensive success rate: {defensive_success_rate:.1f}%, "
                f"Avg defensive actions: {defensive_actions:.1f}"
            )

            # Log metrics to CSV file
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.metrics_file, "a") as f:
                f.write(
                    f"{timestamp},{self.n_calls},{mean_reward:.2f},{defensive_success_rate:.1f}\n"
                )

            # Save best model if current reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = f"{MODEL_PATH.split('.')[0]}_best_reward.zip"
                self.model.save(best_path)
                print(
                    f"New best model with reward {mean_reward:.2f} saved to {best_path}"
                )

        return True


def create_model(env, resume=False):
    """Create a new PPO model or load an existing one"""
    # Improved network architecture for better feature extraction
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 128, 64], vf=[256, 128, 64]
        ),  # Updated format for SB3 v1.8.0+
        normalize_images=True,  # Normalize pixel values
    )

    if resume and os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, tensorboard_log="./logs/tensorboard/")
        print("Model loaded successfully")
    else:
        print("Creating new model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0001,  # Slightly lower learning rate for stability
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,  # GAE parameter for advantage estimation
            clip_range=0.2,  # PPO clipping parameter
            ent_coef=0.01,  # Entropy coefficient for exploration
            device=device,
            tensorboard_log="./logs/tensorboard/",
        )

    return model


def train_model(model, timesteps):
    """Train the model for the specified number of timesteps"""
    # Create callbacks
    save_callback = SaveCallback(check_freq=10000, log_freq=1000)

    print(f"Starting training for {timesteps} timesteps")
    model.learn(
        total_timesteps=timesteps,
        callback=save_callback,
        tb_log_name="kungfu_training",
        progress_bar=True,
    )

    # Save final model
    model.save(MODEL_PATH)
    print(f"Training completed. Final model saved to {MODEL_PATH}")


def evaluate_model(model, n_eval_episodes=5):
    """Evaluate model performance with focus on timing abilities"""
    print(f"Evaluating model over {n_eval_episodes} episodes...")
    eval_env = make_kungfu_env(is_play_mode=True)
    mean_reward = 0
    mean_stages_cleared = 0
    total_defensive_actions = 0
    total_successful_defensive_actions = 0

    # Track jump/crouch timing by action
    jump_actions = 0
    crouch_actions = 0
    successful_jumps = 0
    successful_crouches = 0

    for i in range(n_eval_episodes):
        obs = eval_env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        max_stage = 0
        episode_defensive_actions = 0
        episode_successful_defensive_actions = 0

        # Run until episode ends
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            prev_hp = (
                eval_env.get_attr("prev_hp")[0] if hasattr(eval_env, "get_attr") else 0
            )

            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Track maximum stage reached
            if "current_stage" in info and info["current_stage"] > max_stage:
                max_stage = info["current_stage"]

            # Track defensive actions (jumps and crouches)
            if action == 4:  # Jump
                jump_actions += 1
                # Check if HP stayed the same or increased (successful defense)
                current_hp = (
                    eval_env.get_attr("prev_hp")[0]
                    if hasattr(eval_env, "get_attr")
                    else 0
                )
                if current_hp >= prev_hp:
                    successful_jumps += 1

            elif action == 5:  # Crouch
                crouch_actions += 1
                # Check if HP stayed the same or increased (successful defense)
                current_hp = (
                    eval_env.get_attr("prev_hp")[0]
                    if hasattr(eval_env, "get_attr")
                    else 0
                )
                if current_hp >= prev_hp:
                    successful_crouches += 1

        # Get defensive stats from the environment
        if hasattr(eval_env, "get_attr"):
            try:
                episode_defensive_actions = eval_env.get_attr("defensive_actions")[0]
                episode_successful_defensive_actions = eval_env.get_attr(
                    "successful_defensive_actions"
                )[0]
                total_defensive_actions += episode_defensive_actions
                total_successful_defensive_actions += (
                    episode_successful_defensive_actions
                )
            except:
                pass

        mean_reward += total_reward
        mean_stages_cleared += max_stage

        # Calculate defensive success rate for the episode
        defensive_success_rate = 0
        if episode_defensive_actions > 0:
            defensive_success_rate = (
                episode_successful_defensive_actions / episode_defensive_actions
            ) * 100

        print(
            f"Episode {i+1}: Reward: {total_reward:.1f}, Steps: {steps}, Stage: {max_stage}, "
            f"Defensive success: {defensive_success_rate:.1f}%"
        )

    # Calculate averages
    mean_reward /= n_eval_episodes
    mean_stages_cleared /= n_eval_episodes

    # Calculate overall defensive success rate
    overall_defensive_success_rate = 0
    if total_defensive_actions > 0:
        overall_defensive_success_rate = (
            total_successful_defensive_actions / total_defensive_actions
        ) * 100

    # Calculate success rates for jumps and crouches
    jump_success_rate = 0
    if jump_actions > 0:
        jump_success_rate = (successful_jumps / jump_actions) * 100

    crouch_success_rate = 0
    if crouch_actions > 0:
        crouch_success_rate = (successful_crouches / crouch_actions) * 100

    print(f"\nEvaluation results over {n_eval_episodes} episodes:")
    print(f"- Mean reward: {mean_reward:.2f}")
    print(f"- Mean max stage: {mean_stages_cleared:.1f}")
    print(f"- Overall defensive success rate: {overall_defensive_success_rate:.1f}%")
    print(
        f"- Jump success rate: {jump_success_rate:.1f}% ({successful_jumps}/{jump_actions})"
    )
    print(
        f"- Crouch success rate: {crouch_success_rate:.1f}% ({successful_crouches}/{crouch_actions})"
    )

    return mean_reward, mean_stages_cleared, overall_defensive_success_rate


def main():
    parser = argparse.ArgumentParser(description="Train an AI to play Kung Fu Master")
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
        "--frame-stack",
        type=int,
        default=8,
        choices=[4, 8],
        help="Number of frames to stack (4 or 8). Use 8 for better projectile detection.",
    )
    args = parser.parse_args()

    # Determine frame stack size - auto-detect if resuming, otherwise use command line arg
    frame_stack = args.frame_stack  # Default to command line arg

    # If resuming or evaluating, detect the frame stack size of the existing model
    model_path = args.model_path if args.model_path else MODEL_PATH
    if (args.resume or args.eval_only) and os.path.exists(model_path):
        frame_stack = 8
        print(f"Auto-detected frame stack size from model: n_stack={frame_stack}")

    # Create environment with specified frame stacking
    print(
        f"Creating Kung Fu environment with frame stacking (n_stack={frame_stack}) for projectile detection..."
    )
    env = make_kungfu_env(is_play_mode=args.render, frame_stack=frame_stack)

    # Evaluation only mode
    if args.eval_only:
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found.")
            return

        print(f"Loading model from {model_path} for evaluation...")
        model = PPO.load(model_path, env=env)
        evaluate_model(model, n_eval_episodes=5)
        env.close()
        return

    # Create or load model
    model = create_model(env, resume=args.resume)

    # Train model
    train_model(model, args.timesteps)

    # Evaluate model if requested
    if args.eval:
        evaluate_model(model)

    # Clean up
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
