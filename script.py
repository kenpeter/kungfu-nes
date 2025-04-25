import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces, Wrapper
import retro
from collections import deque
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
N_STACK = 4  # Number of frames to stack for input
VIEWPORT_SIZE = (160, 160)  # Consistent viewport size

# Define actions - complete set
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


class KungFuWrapper(Wrapper):
    def __init__(self, env):
        # Initialize parent class
        super().__init__(env)

        # Reset the environment to get initial observation
        result = env.reset()
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result

        # Get true height and width of the observation
        self.true_height, self.true_width = obs.shape[:2]

        # Set up action space
        self.actions = KUNGFU_ACTIONS
        self.action_names = KUNGFU_ACTION_NAMES
        self.action_space = spaces.Discrete(len(self.actions))

        # Set up observation space for single frame
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(VIEWPORT_SIZE[0], VIEWPORT_SIZE[1], 3),
            dtype=np.uint8,
        )

        # State tracking
        self.last_hp = 0
        self.player_x = 0
        self.last_player_x = 0
        self.game_started = False

        logger.info(f"KungFu wrapper initialized with viewport size {VIEWPORT_SIZE}")

    def reset(self, seed=None, options=None, **kwargs):
        # Reset the environment
        result = self.env.reset(seed=seed, options=options, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Press start to begin the game and move past title screen
        for _ in range(3):
            result = self.env.step([0, 0, 0, 1, 0, 0, 0, 0, 0])  # START action
            if len(result) == 5:
                obs, _, terminated, truncated, _ = result
                if terminated or truncated:
                    obs, info = self.env.reset()
            else:
                obs, _, done, _ = result
                if done:
                    obs, info = self.env.reset()

        # Short delay to let the game initialize
        import time

        time.sleep(0.1)

        # Some additional button presses to ensure we get into gameplay
        for _ in range(3):
            # Press a combination of buttons to try to get past any intro screens
            result = self.env.step([0, 0, 0, 0, 0, 0, 0, 1, 0])  # RIGHT
            if len(result) == 5:
                obs, _, terminated, truncated, _ = result
                if terminated or truncated:
                    obs, info = self.env.reset()
            else:
                obs, _, done, _ = result
                if done:
                    obs, info = self.env.reset()

        self.game_started = True

        # Reset state tracking variables
        ram = self.env.get_ram()
        self.last_hp = float(ram[0x04A6])
        self.player_x = float(ram[0x0094])
        self.last_player_x = self.player_x

        # Resize the observation
        resized_obs = cv2.resize(obs, VIEWPORT_SIZE, interpolation=cv2.INTER_AREA)

        return resized_obs, info

    def step(self, action):
        # Execute action in environment
        result = self.env.step(self.actions[action])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        # Get RAM data for basic reward calculation
        ram = self.env.get_ram()
        hp = float(ram[0x04A6])
        self.player_x = float(ram[0x0094])

        # --- BASIC REWARD CALCULATION ---
        # 1. HP-based reward
        hp_change = hp - self.last_hp
        if hp_change < 0:
            reward = -1.0  # Penalty for taking damage

        # 2. Simple progression reward
        x_change = (
            self.last_player_x - self.player_x
        )  # In Kung Fu, moving left is progress
        progression_reward = x_change * 0.1
        reward += progression_reward

        # 3. Game over penalty
        if done:
            reward -= 5

        # 4. Survival reward
        if not done:
            reward += 0.01

        # Resize observation
        resized_obs = cv2.resize(obs, VIEWPORT_SIZE, interpolation=cv2.INTER_AREA)

        # Update state tracking variables
        self.last_hp = hp
        self.last_player_x = self.player_x

        # Update info dict
        info.update(
            {
                "hp": hp,
                "player_x": self.player_x,
            }
        )

        return resized_obs, reward, terminated, truncated, info


class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)

        # For stacked frames, we need 3 * N_STACK input channels
        n_input_channels = 3 * N_STACK

        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.zeros(1, n_input_channels, VIEWPORT_SIZE[0], VIEWPORT_SIZE[1])
            ).shape[1]

        # Linear layer after CNN
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        # Handle both dictionary and direct observations
        if isinstance(observations, dict):
            observations = observations.get("observation", observations)

        # Convert to float and normalize
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0

        # Channel first conversion if needed (but typically handled by VecTransposeImage)
        if len(observations.shape) == 4 and observations.shape[-1] == 3:
            observations = observations.permute(0, 3, 1, 2)

        return self.linear(self.cnn(observations))


def make_env(render_mode="rgb_array"):
    """Create and wrap the KungFu environment"""
    env = retro.make(
        "KungFu-Nes", use_restricted_actions=retro.Actions.ALL, render_mode=render_mode
    )
    env = Monitor(KungFuWrapper(env))
    return env


class SaveModelCallback(BaseCallback):
    def __init__(self, save_path, save_freq=10000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(f"{self.save_path}_{self.n_calls}")
            if self.verbose > 0:
                print(f"Saved model at step {self.n_calls}")
        return True


def train(args):
    """Train a new model or continue training an existing one"""
    # Create directory for model if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)

    # Create vectorized environment
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env])

    # Stack frames and transpose image
    env = VecFrameStack(env, n_stack=N_STACK)
    # Add this line to handle image format conversion
    from stable_baselines3.common.vec_env import VecTransposeImage

    env = VecTransposeImage(env)

    # Define policy parameters
    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }

    # Load or create model
    if args.resume and os.path.exists(args.model_path + ".zip"):
        print(f"Loading existing model from {args.model_path}")
        model = PPO.load(
            args.model_path, env=env, device="cuda" if args.cuda else "cpu"
        )
    else:
        print("Creating new model")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=0.0003,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu",
        )

    # Setup callback for saving
    callbacks = [SaveModelCallback(save_path=args.model_path, save_freq=args.save_freq)]

    # Train the model
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
    )

    # Save final model
    model.save(args.model_path)
    print(f"Training complete. Final model saved to {args.model_path}")

    # Close environment
    env.close()


def play_direct(args):
    """Play using a trained model with direct rendering"""
    import retro
    import time

    # Create the environment directly without wrappers for rendering
    env = retro.make(
        "KungFu-Nes", use_restricted_actions=retro.Actions.ALL, render_mode="human"
    )

    # Create a separate environment for getting observations and making predictions
    # This helps separate rendering from prediction
    model_env = make_env(render_mode="rgb_array")
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecFrameStack,
        VecTransposeImage,
    )

    vec_env = DummyVecEnv([lambda: model_env])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    vec_env = VecTransposeImage(vec_env)

    # Load the trained model
    model = PPO.load(args.model_path, device="cuda" if args.cuda else "cpu")

    # Initialize both environments
    env.reset()
    obs = vec_env.reset()

    # Press start to begin both games
    for _ in range(3):
        env.step([0, 0, 0, 1, 0, 0, 0, 0, 0])  # START action
        time.sleep(0.1)  # Small delay

    # Play the game
    steps = 0
    dones = [False]

    while not dones[0] and steps < args.max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Execute action in both environments
        obs, _, dones, _ = vec_env.step(action)
        _, _, done, _ = env.step(KUNGFU_ACTIONS[action[0]])

        # Explicit render call with delay
        env.render()
        time.sleep(0.02)  # Adjust delay as needed

        steps += 1

        # Print steps every 100 steps
        if steps % 100 == 0:
            print(f"Steps: {steps}")

    print(f"Game finished after {steps} steps")
    env.close()
    vec_env.close()


def play(args):
    """Play using a trained model with vectorized environment"""
    # Use the direct play method instead for better rendering
    if args.render:
        return play_direct(args)

    # Create environment for non-rendering mode
    env = make_env(render_mode="rgb_array")
    from stable_baselines3.common.vec_env import (
        DummyVecEnv,
        VecFrameStack,
        VecTransposeImage,
    )

    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=N_STACK)
    vec_env = VecTransposeImage(vec_env)

    # Load the trained model
    model = PPO.load(args.model_path, device="cuda" if args.cuda else "cpu")

    # Reset the environment
    obs = vec_env.reset()

    # Play the game
    total_reward = 0
    steps = 0
    dones = [False]

    while not dones[0] and steps < args.max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Execute action in the environment
        obs, rewards, dones, infos = vec_env.step(action)

        # Update stats
        total_reward += rewards[0]
        steps += 1

        # Print info periodically
        if steps % 100 == 0:
            print(f"Steps: {steps}, Total Reward: {total_reward}")
            if "hp" in infos[0]:
                print(f"HP: {infos[0]['hp']}")

    print(f"Game finished after {steps} steps with total reward {total_reward}")
    vec_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train or play Kung Fu with a simplified agent"
    )

    # Common arguments
    parser.add_argument(
        "--model_path",
        default="models/kungfu_simple",
        help="Path to save/load the model",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")

    # Create subparsers for train and play commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument(
        "--timesteps", type=int, default=100000, help="Total timesteps for training"
    )
    train_parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    train_parser.add_argument(
        "--save_freq", type=int, default=10000, help="Save model every N steps"
    )

    # Play command
    play_parser = subparsers.add_parser("play", help="Play using a trained model")
    play_parser.add_argument(
        "--render", action="store_true", help="Render the environment"
    )
    play_parser.add_argument(
        "--max_steps", type=int, default=10000, help="Maximum steps to play"
    )

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
