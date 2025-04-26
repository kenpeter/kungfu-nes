import os
import argparse
import torch
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from kungfu_env import make_kungfu_env, MODEL_PATH, logger as env_logger

# Set up logging for the training script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()],
)
logger = logging.getLogger("KungFuTrainer")


class SaveCallback(BaseCallback):
    """
    Callback for saving the model periodically during training
    """

    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        logger.info(f"SaveCallback initialized with check_freq={check_freq}")

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            logger.info(f"Saving model at step {self.n_calls}")
            self.model.save(MODEL_PATH)
            logger.info(f"Model saved successfully to {MODEL_PATH}")
        return True


def create_model(env, resume=False):
    """Create a new PPO model or load an existing one"""
    policy_kwargs = dict(net_arch=[64, 64])
    if resume and os.path.exists(MODEL_PATH):
        logger.info(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
        logger.info("Model loaded successfully")
    else:
        logger.info("Creating new model")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

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
            device=device,
            tensorboard_log="./tensorboard_logs/",
        )
        logger.info("New model created with the following hyperparameters:")
        logger.info(f"  Policy: CnnPolicy")
        logger.info(f"  Learning rate: 0.0003")
        logger.info(f"  n_steps: 2048")
        logger.info(f"  batch_size: 64")
        logger.info(f"  n_epochs: 10")
        logger.info(f"  gamma: 0.99")

    return model


def train_model(model, timesteps):
    """Train the model for the specified number of timesteps"""
    logger.info(f"Starting training for {timesteps} timesteps")
    callback = SaveCallback(check_freq=10000)
    model.learn(total_timesteps=timesteps, callback=callback)
    model.save(MODEL_PATH)
    logger.info(f"Training completed. Final model saved to {MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train an AI to play Kung Fu Master")
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Set logging level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        env_logger.setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    # Check for CUDA
    if torch.cuda.is_available():
        logger.info("CUDA is available! Training will use GPU.")
    else:
        logger.info("CUDA not available. Training will use CPU.")

    # Create environment
    logger.info("Creating Kung Fu Master environment")
    env = make_kungfu_env(is_play_mode=False)
    logger.info("Environment created successfully")

    # Create or load model
    model = create_model(env, resume=args.resume)

    # Train model
    logger.info(f"Starting training for {args.timesteps} timesteps...")
    train_model(model, args.timesteps)

    # Clean up
    env.close()
    logger.info("Environment closed")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
