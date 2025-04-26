import os
import argparse
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
            print(f"Step: {self.n_calls}, Mean reward: {mean_reward:.2f}")

            # Save best model if current reward is better
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_path = f"{MODEL_PATH.split('.')[0]}_best.zip"
                self.model.save(best_path)
                print(
                    f"New best model with reward {mean_reward:.2f} saved to {best_path}"
                )

        return True


def create_model(env, resume=False):
    """Create a new PPO model or load an existing one"""
    # Improved network architecture for better feature extraction
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[128, 64], vf=[128, 64])
        ],  # Separate policy and value networks
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
            learning_rate=0.0003,
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
    args = parser.parse_args()

    # Create environment
    print("Creating Kung Fu environment...")
    env = make_kungfu_env(is_play_mode=args.render)

    # Create or load model
    model = create_model(env, resume=args.resume)

    # Train model
    train_model(model, args.timesteps)

    # Evaluate model if requested
    if args.eval:
        print("Evaluating model...")
        eval_env = make_kungfu_env(is_play_mode=True)
        mean_reward = 0
        n_eval_episodes = 5

        for i in range(n_eval_episodes):
            obs = eval_env.reset()[0]
            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            mean_reward += total_reward
            print(f"Episode {i+1}: Total reward: {total_reward}, Steps: {steps}")

        mean_reward /= n_eval_episodes
        print(f"Mean reward over {n_eval_episodes} episodes: {mean_reward}")

    # Clean up
    env.close()
    print("Environment closed")


if __name__ == "__main__":
    main()
