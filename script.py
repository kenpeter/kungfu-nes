import os
import argparse
import numpy as np
import retro
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Define the Kung Fu Master action space
KUNGFU_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No-op
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Punch)
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # SELECT
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # START
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # UP (Jump)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN (Crouch)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (Kick)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # B + A (Punch + Kick)
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # UP + RIGHT (Jump + Right)
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # DOWN + A (Crouch Kick)
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN + B (Crouch Punch)
]

KUNGFU_ACTION_NAMES = [
    "No-op",
    "Punch",
    "Select",
    "Start",
    "Jump",
    "Crouch",
    "Left",
    "Right",
    "Kick",
    "Punch + Kick",
    "Jump + Right",
    "Crouch Kick",
    "Crouch Punch",
]

# Set model path
MODEL_PATH = "model/kungfu.zip"


# Create custom environment wrapper for Kung Fu Master with discrete actions
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super(KungFuMasterEnv, self).__init__(env)
        # Override the action space to use our discrete actions
        self.action_space = gym.spaces.Discrete(len(KUNGFU_ACTIONS))

    def step(self, action):
        # Convert our discrete action to the multi-binary format
        converted_action = KUNGFU_ACTIONS[action]
        obs, reward, terminated, truncated, info = self.env.step(converted_action)
        done = terminated or truncated

        # Simple reward shaping to encourage progress
        shaped_reward = reward

        # Use the info dict to enhance rewards
        if "score" in info:
            shaped_reward += info["score"] * 0.01

        # Encourage staying alive
        shaped_reward += 0.1

        # Discourage death
        if done and "lives" in info and info["lives"] == 0:
            shaped_reward -= 10.0

        return obs, shaped_reward, terminated, truncated, info


def make_kungfu_env(num_envs=1, is_play_mode=False):
    """Create vectorized environments for Kung Fu Master"""

    def make_env(rank):
        def _init():
            try:
                # Set render_mode only for play mode
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFu-Nes", render_mode=render_mode)
            except Exception as e:
                # If that fails, try alternative ROM name
                print(f"Failed to load KungFu-Nes, trying alternative ROM name: {e}")
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)

            # Wrap the environment with our custom wrapper
            env = KungFuMasterEnv(env)

            # Add monitoring
            os.makedirs("logs", exist_ok=True)
            env = Monitor(env, os.path.join("logs", f"kungfu_{rank}"))

            return env

        return _init

    if num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Stack 4 frames to give the model some temporal context
    env = VecFrameStack(env, n_stack=4)

    return env


def create_model(env, resume=False):
    """Create or load a PPO model"""
    # Define neural network architecture
    policy_kwargs = dict(
        net_arch=[64, 64]  # Simple network with two hidden layers of 64 units
    )

    if resume and os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model")
        model = PPO(
            "CnnPolicy",  # CNN policy for image inputs
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    return model


def train_model(model, timesteps):
    """Train the model"""
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    # Save the final model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


def play_game(env, model, episodes=5):
    """Use the trained model to play the game with rendering"""
    obs = env.reset()

    for episode in range(episodes):
        done = False
        total_reward = 0
        step = 0

        while not done:
            # Use deterministic actions for gameplay
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Display action being taken
            action_name = KUNGFU_ACTION_NAMES[action[0]]
            print(f"Step: {step}, Action: {action_name}, Reward: {reward}")

            total_reward += reward
            step += 1

            if done[0]:
                print(f"Episode {episode+1} finished with total reward: {total_reward}")
                obs = env.reset()
                break


def main():
    parser = argparse.ArgumentParser(description="Train or play Kung Fu Master with AI")
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play the game with trained agent"
    )

    args = parser.parse_args()

    # Create the environment, specifying if we're in play mode
    env = make_kungfu_env(num_envs=args.num_envs, is_play_mode=args.play)

    # Create or load the model
    model = create_model(env, resume=args.resume)

    if args.play:
        # Play the game with the trained agent
        play_game(env, model)
    else:
        # Train the model
        train_model(model, args.timesteps)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Make sure cuda is enabled by default if available
    if torch.cuda.is_available():
        print("CUDA is available! Training will use GPU.")
    else:
        print("CUDA not available. Training will use CPU.")

    main()
