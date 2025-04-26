import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
import os.path
from kungfu_env import make_kungfu_env, MODEL_PATH


class SaveCallback(BaseCallback):
    """Callback for saving the model periodically during training"""

    def __init__(self, check_freq=10000, log_freq=1000):
        super().__init__()
        self.check_freq = check_freq
        self.log_freq = log_freq
        self.best_mean_reward = -float("inf")
        self.best_enemies_defeated = -float("inf")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    def _on_step(self):
        # Save model periodically
        if self.n_calls % self.check_freq == 0:
            path = f"{MODEL_PATH.split('.')[0]}_{self.n_calls}.zip"
            self.model.save(path)
            print(f"Model saved to {path}")

        # Log training info periodically
        if self.n_calls % self.log_freq == 0:
            mean_reward = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)
            enemies_defeated = (
                np.mean(
                    [
                        ep_info.get("defeated_enemies", 0)
                        for ep_info in self.model.ep_info_buffer
                    ]
                )
                if len(self.model.ep_info_buffer) > 0
                else 0
            )

            print(
                f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}, "
                f"Avg enemies defeated: {enemies_defeated:.1f}"
            )

            # Save best model if current reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = f"{MODEL_PATH.split('.')[0]}_best_reward.zip"
                self.model.save(best_path)
                print(
                    f"New best model with reward {mean_reward:.2f} saved to {best_path}"
                )

            # Also track enemies defeated as a metric for saving models
            if enemies_defeated > self.best_enemies_defeated and enemies_defeated > 0:
                self.best_enemies_defeated = enemies_defeated
                best_enemies_path = f"{MODEL_PATH.split('.')[0]}_best_enemies.zip"
                self.model.save(best_enemies_path)
                print(
                    f"New best model with {enemies_defeated:.1f} enemies defeated saved to {best_enemies_path}"
                )

        return True


def create_model(env, resume=False):
    """Create a new PPO model or load an existing one"""
    # Improved network architecture for better feature extraction
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[256, 128, 64], vf=[256, 128, 64])
        ],  # Deeper networks for better performance
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
    callback = SaveCallback(check_freq=10000, log_freq=1000)

    print(f"Starting training for {timesteps} timesteps")
    model.learn(
        total_timesteps=timesteps,
        callback=callback,
        tb_log_name="kungfu_training",
        progress_bar=True,
    )

    # Save final model
    model.save(MODEL_PATH)
    print(f"Training completed. Final model saved to {MODEL_PATH}")


def evaluate_model(model, n_eval_episodes=5):
    """Evaluate model performance"""
    print(f"Evaluating model over {n_eval_episodes} episodes...")
    eval_env = make_kungfu_env(is_play_mode=True)
    mean_reward = 0
    mean_enemies_defeated = 0
    mean_stages_cleared = 0

    for i in range(n_eval_episodes):
        obs = eval_env.reset()[0]
        done = False
        total_reward = 0
        steps = 0
        max_stage = 0

        # Run until episode ends
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

            # Track maximum stage reached
            if "current_stage" in info and info["current_stage"] > max_stage:
                max_stage = info["current_stage"]

        # Get episode statistics
        defeated_enemies = info.get("defeated_enemies", 0)
        mean_reward += total_reward
        mean_enemies_defeated += defeated_enemies
        mean_stages_cleared += max_stage

        print(
            f"Episode {i+1}: Reward: {total_reward:.1f}, Steps: {steps}, "
            f"Stage: {max_stage}, Enemies defeated: {defeated_enemies}"
        )

    # Calculate averages
    mean_reward /= n_eval_episodes
    mean_enemies_defeated /= n_eval_episodes
    mean_stages_cleared /= n_eval_episodes

    print(f"Evaluation results over {n_eval_episodes} episodes:")
    print(f"- Mean reward: {mean_reward:.2f}")
    print(f"- Mean enemies defeated: {mean_enemies_defeated:.1f}")
    print(f"- Mean max stage: {mean_stages_cleared:.1f}")

    return mean_reward, mean_enemies_defeated, mean_stages_cleared


def main():
    parser = argparse.ArgumentParser(description="Train an AI to play Kung Fu Master")
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Number of timesteps to train"
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
    args = parser.parse_args()

    # Create environment
    print("Creating Kung Fu environment...")
    env = make_kungfu_env(is_play_mode=args.render)

    # Evaluation only mode
    if args.eval_only:
        if args.model_path:
            model_path = args.model_path
        else:
            model_path = MODEL_PATH

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
