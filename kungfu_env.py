import os
import numpy as np
import retro  # stable retro
import gymnasium as gym  # not open ai gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("kungfu_env.log"), logging.StreamHandler()],
)
logger = logging.getLogger("KungFuEnv")

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


# Custom environment wrapper for Kung Fu Master with basic rewards
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        filtered_actions = [action for i, action in enumerate(KUNGFU_ACTIONS) if i != 3]
        filtered_action_names = [
            name for i, name in enumerate(KUNGFU_ACTION_NAMES) if i != 3
        ]
        self.KUNGFU_ACTIONS = filtered_actions
        self.KUNGFU_ACTION_NAMES = filtered_action_names
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_stage = 0
        logger.info("KungFuMasterEnv initialized")

    def reset(self, **kwargs):
        # Make sure we return observation and info as required by Gymnasium API
        logger.info("Resetting environment")
        obs_result = self.env.reset(**kwargs)

        # Handle different return types
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs, info = obs_result
        else:
            obs = obs_result
            info = {}

        logger.info("Pressing START button to begin game")
        start_action = KUNGFU_ACTIONS[3]  # START button

        # Log the actual button being pressed
        logger.info(f"START button action: {start_action}")

        for i in range(15):  # Press START for 15 frames
            logger.debug(f"Pressing START button (frame {i+1}/15)")
            step_result = self.env.step(start_action)
            if len(step_result) == 5:
                obs, _, term, trunc, _ = step_result
                if term or trunc:
                    logger.warning(
                        "Environment terminated or truncated while pressing START"
                    )
            else:
                obs, _, done, _ = step_result
                if done:
                    logger.warning("Environment done while pressing START")

        logger.info("START button pressed successfully, waiting for 30 frames")
        for i in range(30):  # Wait for 30 frames
            step_result = self.env.step(KUNGFU_ACTIONS[0])
            if len(step_result) == 5:
                obs, _, term, trunc, _ = step_result
                if term or trunc:
                    logger.warning(
                        "Environment terminated or truncated during wait period"
                    )
            else:
                obs, _, done, _ = step_result
                if done:
                    logger.warning("Environment done during wait period")

        try:
            ram = self.env.get_ram()
            self.prev_score = self._get_score(ram)
            self.prev_hp = int(ram[0x04A6])
            self.prev_x_pos = int(ram[0x0094])
            self.prev_stage = int(ram[0x0058])

            # Log initial game state
            logger.info(
                f"Initial game state - Score: {self.prev_score}, HP: {self.prev_hp}, Stage: {self.prev_stage}"
            )

            # Check if the game has actually started (based on stage or other indicator)
            if self.prev_stage > 0:
                logger.info("Game successfully started - player is on stage")
            else:
                # Try to detect if we're in game by checking other RAM values
                game_active = ram[0x04A6] > 0  # Check if player has HP
                if game_active:
                    logger.info("Game appears to be active (player has HP)")
                else:
                    logger.warning(
                        "Game may not have started properly - no indicators found"
                    )

        except Exception as e:
            logger.error(f"Error accessing RAM: {str(e)}")

        # Always return obs and info
        return obs, info

    def step(self, action):
        if action == 3:  # Filter out START button during gameplay
            logger.debug("Filtering out START button during gameplay")
            action = 0
        converted_action = self.KUNGFU_ACTIONS[action]
        step_result = self.env.step(converted_action)

        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            error_msg = f"Unexpected step result length: {len(step_result)}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if not isinstance(info, dict):
            info = {}

        # Simple reward shaping
        try:
            ram = self.env.get_ram()
            current_score = self._get_score(ram)
            current_hp = int(ram[0x04A6])
            current_x_pos = int(ram[0x0094])
            current_stage = int(ram[0x0058])

            shaped_reward = 0.0
            # Base game reward
            shaped_reward += reward

            # Score increase reward
            score_diff = self._safe_subtract(current_score, self.prev_score)
            if score_diff > 0:
                shaped_reward += score_diff * 0.01

            # HP loss penalty
            hp_diff = self._safe_subtract(current_hp, self.prev_hp)
            if hp_diff < 0:
                shaped_reward += hp_diff * 0.5

            # Stage completion reward
            if current_stage > self.prev_stage:
                logger.info(f"Stage advanced from {self.prev_stage} to {current_stage}")
                shaped_reward += 10.0

            # Update previous values
            self.prev_score = current_score
            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_stage = current_stage

            info["current_stage"] = current_stage

        except Exception as e:
            logger.error(f"Error in reward shaping: {str(e)}")
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info

    def _get_score(self, ram):
        score = 1
        has_score = False
        for addr in [0x0531, 0x0532, 0x0533, 0x0534, 0x0535]:
            if ram[addr] > 0:
                has_score = True
                score += ram[addr]
        if not has_score:
            return 1
        return score

    def _safe_subtract(self, a, b):
        a_int, b_int = int(a), int(b)
        if abs(a_int - b_int) < 128:
            return a_int - b_int
        if a_int == 0 and b_int > 200:
            return -b_int
        if a_int < 50 and b_int > 200:
            return (a_int + 256) - b_int
        return a_int - b_int


def make_kungfu_env(is_play_mode=False):
    """Create a single Kung Fu Master environment wrapped for RL training"""
    try:
        render_mode = "human" if is_play_mode else None
        logger.info(
            f"Attempting to create Kung Fu environment with render_mode={render_mode}"
        )
        env = retro.make(game="KungFu-Nes", render_mode=render_mode)
        logger.info("Successfully created KungFu-Nes environment")
    except Exception as e:
        logger.warning(f"Failed to create KungFu-Nes environment: {str(e)}")
        try:
            render_mode = "human" if is_play_mode else None
            logger.info(
                f"Attempting to create KungFuMaster-Nes environment with render_mode={render_mode}"
            )
            env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)
            logger.info("Successfully created KungFuMaster-Nes environment")
        except Exception as e:
            logger.error(f"Failed to create KungFuMaster-Nes environment: {str(e)}")
            raise

    env = KungFuMasterEnv(env)
    os.makedirs("logs", exist_ok=True)
    logger.info("Created logs directory")
    env = Monitor(env, os.path.join("logs", "kungfu"))
    logger.info("Wrapped environment with Monitor")

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: env])
    logger.info("Wrapped environment with DummyVecEnv")

    # Frame stacking for better learning with visual inputs
    n_stack = 4
    env = VecFrameStack(env, n_stack=n_stack)
    logger.info(f"Wrapped environment with VecFrameStack (n_stack={n_stack})")

    return env
