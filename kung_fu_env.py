import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3 import PPO
from gymnasium import spaces
from typing import Dict, List, Tuple, Type, Union, Optional
import tempfile
import atexit
import gc
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="kungfu_env.log",
)
logger = logging.getLogger("kungfu_env")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# import the projectile detector - use try/except to handle import errors gracefully
try:
    from projectile import (
        ImprovedProjectileDetector,
        enhance_observation_with_projectiles,
    )

    PROJECTILE_DETECTOR_AVAILABLE = True
except ImportError:
    logger.warning("Failed to import projectile detector. Using fallback detection.")
    PROJECTILE_DETECTOR_AVAILABLE = False

    # Define fallback projectile detection if module isn't available
    class ImprovedProjectileDetector:
        def __init__(self, debug=False):
            self.debug = debug
            logger.info("Using fallback projectile detector")

        def reset(self):
            pass

        def detect_projectiles(self, frame):
            return []

    def enhance_observation_with_projectiles(obs, detector, player_position):
        return {"projectiles": [], "recommended_action": 0}


# Singleton for retro environment management
class RetroEnvManager:
    _instance = None
    _active_envs = set()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = RetroEnvManager()
        return cls._instance

    def __init__(self):
        self._active_envs = set()
        # Register cleanup on exit
        atexit.register(self.cleanup_all_envs)

    def register_env(self, env):
        self._active_envs.add(env)

    def unregister_env(self, env):
        if env in self._active_envs:
            self._active_envs.remove(env)

    def cleanup_all_envs(self):
        logger.info(f"Cleaning up {len(self._active_envs)} environments")
        for env in list(self._active_envs):
            try:
                if hasattr(env, "close"):
                    env.close()
            except Exception as e:
                logger.error(f"Error closing environment: {e}")
        self._active_envs.clear()
        # Force garbage collection
        gc.collect()


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

# Action names
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

# Critical memory addresses
MEMORY = {
    "current_stage": 0x0058,  # Current Stage
    "player_hp": 0x04A6,  # Hero HP
    "player_x": 0x0094,  # Hero Screen Pos X
    "player_y": 0x00B6,  # Hero Screen Pos Y
    "game_mode": 0x0051,  # Game Mode
    "boss_hp": 0x04A5,  # Boss HP
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes

# Set default model path - using a single model path
MODEL_PATH = "model/kungfu.zip"


# Custom Monitor class that's more resilient
class ResilientMonitor(Monitor):
    """A more fault-tolerant version of Monitor that won't crash on I/O errors"""

    def __init__(self, env, filename, **kwargs):
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Add unique timestamp to prevent conflicts
        timestamp = int(time.time())
        filename_with_timestamp = f"{filename}.{timestamp}"

        super().__init__(env, filename_with_timestamp, **kwargs)
        logger.info(
            f"ResilientMonitor initialized with file: {filename_with_timestamp}"
        )

    def step(self, action):
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, info
        """
        try:
            return super().step(action)
        except ValueError as e:
            if "I/O operation on closed file" in str(e):
                logger.warning("Monitor file I/O error. Reopening file.")
                # Create a new results writer
                self._setup_results_writer()
                # Proceed with step, but don't try to log
                obs, reward, terminated, truncated, info = self.env.step(action)
                return obs, reward, terminated, truncated, info
            else:
                raise e

    def _setup_results_writer(self):
        """Set up the results writer with required columns."""
        try:
            # Close the old writer if exists
            if hasattr(self, "results_writer") and hasattr(
                self.results_writer, "close"
            ):
                try:
                    self.results_writer.close()
                except:
                    pass

            # Create a new directory if needed
            os.makedirs(os.path.dirname(self.file_handler.name), exist_ok=True)

            # Create the results writer
            from stable_baselines3.common.monitor import ResultsWriter

            self.results_writer = ResultsWriter(
                self.file_handler.name,
                header={"t_start": self.t_start, **self.metadata},
                extra_keys=self.info_keywords,
            )
        except Exception as e:
            logger.error(f"Failed to set up results writer: {e}")
            # Use a dummy file as a fallback
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".monitor.csv")
            logger.info(f"Using temporary monitor file: {temp_file.name}")
            self.file_handler = temp_file
            from stable_baselines3.common.monitor import ResultsWriter

            self.results_writer = ResultsWriter(
                self.file_handler.name,
                header={"t_start": self.t_start, **self.metadata},
                extra_keys=self.info_keywords,
            )

    def close(self):
        """Closes the environment"""
        super().close()
        # Make sure file handlers are closed
        if hasattr(self, "file_handler") and hasattr(self.file_handler, "close"):
            try:
                self.file_handler.close()
            except:
                pass
        if hasattr(self, "results_writer") and hasattr(self.results_writer, "close"):
            try:
                self.results_writer.close()
            except:
                pass


# Enhanced environment wrapper with improved reward shaping and action buffering
class EnhancedKungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Filter out START action from regular gameplay
        filtered_actions = [action for i, action in enumerate(KUNGFU_ACTIONS) if i != 3]
        filtered_action_names = [
            name for i, name in enumerate(KUNGFU_ACTION_NAMES) if i != 3
        ]
        self.KUNGFU_ACTIONS = filtered_actions
        self.KUNGFU_ACTION_NAMES = filtered_action_names
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))

        # Flag to track whether reset has been called
        self.reset_called = False

        # State tracking
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_stage = 0
        self.n_steps = 0
        self.episode_steps = 0
        self.total_training_steps = 0  # Added for curriculum learning

        # Enhanced metrics for projectile avoidance
        self.detected_projectiles = 0
        self.defensive_actions = 0
        self.projectile_defensive_actions = 0
        self.successful_defensive_actions = 0
        self.successful_projectile_avoidance = 0

        # Track offensive actions and success
        self.offensive_actions = 0
        self.successful_offensive_actions = 0
        self.consecutive_defenses = 0  # Track consecutive defensive actions

        # Track progression through the game
        self.passed_projectile_positions = set()  # Store sections we've passed
        # break through?
        self.breakthrough_rewards_given = 0
        self.last_progress_time = 0  # Track when significant progress was last made
        self.stagnation_counter = 0  # Count steps without progress

        # Initialize projectile detector
        if PROJECTILE_DETECTOR_AVAILABLE:
            self.projectile_detector = ImprovedProjectileDetector(debug=True)
            logger.info("Initialized improved projectile detector")
        else:
            self.projectile_detector = ImprovedProjectileDetector(debug=False)
            logger.warning("Using fallback projectile detector")

        # Raw observation buffer for projectile detection
        self.raw_observation_buffer = []
        self.max_buffer_size = 4  # Store the last 4 frames

        # Action buffer for consistent defensive actions
        self.action_buffer = []
        self.action_buffer_size = 5  # Hold actions for 5 frames
        self.last_defensive_action = 0
        self.last_defensive_action_time = 0
        self.defensive_cooldown = 20  # Frames between defensive actions

        logger.info(
            "Enhanced KungFuMasterEnv initialized with improved projectile detection"
        )

    def _buffer_raw_observation(self, obs):
        """Store raw observations for projectile detection"""
        # If buffer is at max size, remove oldest frame
        if len(self.raw_observation_buffer) >= self.max_buffer_size:
            self.raw_observation_buffer.pop(0)

        # Add new observation
        self.raw_observation_buffer.append(obs)

    def get_ram(self):
        """Safely get RAM from the environment, compatible with Stable Retro"""
        try:
            # For Stable Retro, try these common access methods
            if hasattr(self.env, "get_ram"):
                return self.env.get_ram()
            elif hasattr(self.env.unwrapped, "get_ram"):
                return self.env.unwrapped.get_ram()

            # Direct data access methods
            elif hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
                return self.env.data.memory
            elif hasattr(self.env.unwrapped, "data") and hasattr(
                self.env.unwrapped.data, "memory"
            ):
                return self.env.unwrapped.data.memory

            # Last resort - try internal attribute if it exists
            elif hasattr(self.env, "_memory") and self.env._memory is not None:
                return self.env._memory

            # If all else fails, return dummy RAM
            logger.warning("Unable to access RAM through known methods")
            return np.zeros(0x10000, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error accessing RAM: {e}")
            return np.zeros(0x10000, dtype=np.uint8)

    def get_ram_value(self, address):
        """Get a value from RAM at the specified address"""
        try:
            ram = self.get_ram()
            if ram is not None and len(ram) > address:
                return int(ram[address])
            return 0
        except Exception as e:
            logger.error(f"Error accessing RAM at address 0x{address:04x}: {e}")
            return 0

    def get_stage(self):
        """Get current stage"""
        return self.get_ram_value(MEMORY["current_stage"])

    def get_hp(self):
        """Get current HP"""
        return self.get_ram_value(MEMORY["player_hp"])

    def get_player_position(self):
        """Get player position as (x, y) tuple"""
        try:
            x = self.get_ram_value(MEMORY["player_x"])
            y = self.get_ram_value(MEMORY["player_y"])
            return (x, y)
        except Exception as e:
            logger.error(f"Error getting player position: {e}")
            return (0, 0)  # Return default position

    def get_score(self):
        """Get score from RAM"""
        try:
            score = 0
            for i, addr in enumerate(MEMORY["score"]):
                digit = self.get_ram_value(addr)
                score += digit * (10 ** (4 - i))
            return score
        except Exception as e:
            logger.error(f"Error getting score: {e}")
            return 0

    def reset(self, **kwargs):
        """Reset the environment with Stable Retro compatibility"""
        # Set flag that reset has been called
        self.reset_called = True

        # Reset the environment
        try:
            # For gymnasium/Stable Retro
            obs_result = self.env.reset(**kwargs)

            # Handle different return types
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                obs, info = obs_result
            else:
                obs = obs_result
                info = {}
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            # Create a dummy observation and info
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            info = {}

        # Reset step counter and metrics
        self.episode_steps = 0
        self.detected_projectiles = 0
        self.defensive_actions = 0
        self.projectile_defensive_actions = 0
        self.successful_defensive_actions = 0
        self.successful_projectile_avoidance = 0

        # Reset offensive metrics
        self.offensive_actions = 0
        self.successful_offensive_actions = 0
        self.consecutive_defenses = 0
        self.passed_projectile_positions = set()
        # break through
        self.breakthrough_rewards_given = 0
        self.last_progress_time = 0

        # Reset action buffer
        self.action_buffer = []
        self.last_defensive_action = 0
        self.last_defensive_action_time = 0

        # Log defensive timing stats from previous episode
        if self.n_steps > 0:  # Only log if not the first episode
            projectile_avoidance_rate = 0
            if self.projectile_defensive_actions > 0:
                projectile_avoidance_rate = (
                    self.successful_projectile_avoidance
                    / self.projectile_defensive_actions
                ) * 100
                logger.info(
                    f"Episode projectile stats - Detected: {self.detected_projectiles}, "
                    f"Defensive actions: {self.projectile_defensive_actions}, "
                    f"Successful avoidance: {self.successful_projectile_avoidance}, "
                    f"Avoidance rate: {projectile_avoidance_rate:.1f}%, "
                    f"Breakthroughs: {self.breakthrough_rewards_given}"
                )

        # Clear observation buffer
        self.raw_observation_buffer = []

        # Reset projectile detector
        self.projectile_detector.reset()

        # Get initial state
        self.prev_hp = self.get_hp()
        self.prev_x_pos, self.prev_y_pos = self.get_player_position()
        self.prev_stage = self.get_stage()
        self.prev_boss_hp = self.get_ram_value(MEMORY["boss_hp"])
        self.prev_score = self.get_score()

        logger.info(
            f"Initial state - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
            f"Pos: ({self.prev_x_pos}, {self.prev_y_pos}), Score: {self.prev_score}"
        )

        # Buffer the initial observation
        self._buffer_raw_observation(obs)

        # Return based on gym version
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            return obs, info
        else:
            return obs

    def step(self, action):
        """Take a step in the environment with Stable Retro compatibility"""
        # Check if reset has been called
        if not self.reset_called:
            logger.warning(
                "step() called before reset(). Attempting to reset the environment."
            )
            self.reset()

        # Increment total training steps counter
        self.total_training_steps += 1

        # Identify offensive actions
        is_offensive_action = action in [1, 8, 9, 11, 12]  # Punch, Kick, combinations

        # Check action buffer for defensive actions - MODIFIED FOR PROGRESSIVE REDUCTION
        if self.action_buffer:
            buffered_action = self.action_buffer.pop(0)

            # Only use buffer if there's an immediate projectile threat
            # And reduce probability of override as training progresses
            override_chance = max(0.5, 1.0 - (self.total_training_steps / 2000000))

            if np.random.rand() < override_chance and buffered_action in [4, 5]:
                # Don't override if we're trying to do an offensive action after many defenses
                if is_offensive_action and self.consecutive_defenses > 5:
                    logger.info(
                        f"ALLOWING OFFENSIVE ACTION: {self.KUNGFU_ACTION_NAMES[action]} after many defenses"
                    )
                else:
                    # Otherwise, use the buffered defensive action
                    logger.debug(
                        f"ACTION OVERRIDE ({override_chance:.2f}): Using buffered defensive action {self.KUNGFU_ACTION_NAMES[buffered_action]} instead of {self.KUNGFU_ACTION_NAMES[action]}"
                    )
                    action = buffered_action

        # Convert to actual action
        try:
            converted_action = self.KUNGFU_ACTIONS[action]
        except Exception as e:
            logger.error(f"Error converting action {action}: {e}")
            converted_action = self.KUNGFU_ACTIONS[0]  # Default to no-op

        # Get current hp and current position before step
        current_hp = self.get_hp()
        player_position = self.get_player_position()
        if not isinstance(player_position, tuple) or len(player_position) != 2:
            logger.warning(f"Invalid player position format: {player_position}")
            player_position = (0, 0)

        # Initialize projectile variables
        image_projectiles = []
        projectile_info = None
        recommended_action = 0

        # Detect projectiles if we have enough frames
        if len(self.raw_observation_buffer) >= 2:
            try:
                # Get the latest frame for detection
                current_frame = self.raw_observation_buffer[-1]

                # Detect projectiles using OpenCV
                projectile_info = enhance_observation_with_projectiles(
                    current_frame, self.projectile_detector, player_position
                )

                image_projectiles = projectile_info.get("projectiles", [])
                recommended_action = projectile_info.get("recommended_action", 0)

                # Fill action buffer if a defensive action is recommended
                if recommended_action in [4, 5]:  # Jump or Crouch
                    action_name = self.KUNGFU_ACTION_NAMES[recommended_action]

                    # Adaptive cooldown based on consecutive defenses
                    adaptive_cooldown = self.defensive_cooldown + (
                        self.consecutive_defenses * 5
                    )

                    # Only buffer defensive action if we're not in cooldown period
                    if (
                        self.episode_steps - self.last_defensive_action_time
                        > adaptive_cooldown
                    ):
                        logger.info(
                            f"⚠️ BUFFERING DEFENSIVE ACTION: {action_name} - {len(image_projectiles)} projectiles detected!"
                        )
                        # Clear existing buffer and fill with the new action
                        self.action_buffer = [
                            recommended_action
                        ] * self.action_buffer_size
                        self.last_defensive_action = recommended_action
                        self.last_defensive_action_time = self.episode_steps
            except Exception as e:
                logger.error(f"Error in projectile detection: {e}")

        # Determine if there's an active projectile threat
        projectile_threat = len(image_projectiles) > 0

        if projectile_threat:
            self.detected_projectiles += 1

        # Track defensive actions
        is_defensive_action = action == 4 or action == 5  # Jump or Crouch

        if is_defensive_action:
            self.defensive_actions += 1
            self.consecutive_defenses += 1
            if projectile_threat:
                self.projectile_defensive_actions += 1
        elif is_offensive_action:
            self.offensive_actions += 1
            # Reset consecutive defenses when offensive action is taken
            self.consecutive_defenses = max(0, self.consecutive_defenses - 2)

        # Take step in environment with error handling for different gym versions
        try:
            # Try step with action
            step_result = self.env.step(converted_action)

            # Handle different return formats (gym vs gymnasium)
            if len(step_result) == 5:
                # Gymnasium format
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Old gym format
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                # Unexpected format
                logger.warning(
                    f"Unexpected step result format with {len(step_result)} values"
                )
                obs = (
                    np.zeros_like(self.raw_observation_buffer[-1])
                    if self.raw_observation_buffer
                    else np.zeros((224, 240, 3), dtype=np.uint8)
                )
                reward = 0
                terminated = True
                truncated = True
                info = {}

        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            # Fall back to a safe return value format
            obs = (
                np.zeros_like(self.raw_observation_buffer[-1])
                if self.raw_observation_buffer
                else np.zeros((224, 240, 3), dtype=np.uint8)
            )
            reward = -1.0  # Penalty for error
            terminated = True
            truncated = True
            info = {"error": str(e)}

        # Ensure info is a dictionary
        if not isinstance(info, dict):
            info = {}

        # Buffer the new observation
        self._buffer_raw_observation(obs)

        # Increment episode step counter
        self.episode_steps += 1

        # Check for timeout
        if self.episode_steps >= MAX_EPISODE_STEPS:
            truncated = True
            reward -= 5.0

        # Enhanced reward shaping
        try:
            # Get new state
            current_hp = self.get_hp()
            current_x_pos, current_y_pos = self.get_player_position()
            current_stage = self.get_stage()
            current_boss_hp = self.get_ram_value(MEMORY["boss_hp"])
            current_score = self.get_score()

            # Log status occasionally
            if self.n_steps % 100 == 0:
                time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30  # in seconds
                logger.info(
                    f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                    f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s"
                )

            # Increment step counter
            self.n_steps += 1

            # Shape the reward
            shaped_reward = reward

            # Score reward - IMPROVED WITH EMPHASIS ON ENEMY DEFEAT
            score_diff = current_score - self.prev_score
            if score_diff > 0:
                # Basic score rewards
                shaped_reward += score_diff * 0.05

                # Special bonus for significant score increases (enemy defeats)
                if score_diff >= 100:
                    shaped_reward += 10.0  # Significant reward for defeating enemies
                    self.successful_offensive_actions += 1

                    # Reset defensive bias after successful offense
                    self.consecutive_defenses = max(0, self.consecutive_defenses - 3)

                    logger.info(
                        f"ENEMY DEFEAT: Score increased by {score_diff}! +10.0 reward. Current: {current_score}"
                    )

            # HP loss penalty
            hp_diff = current_hp - self.prev_hp
            if hp_diff < 0 and current_hp < 200:  # Normal health loss
                shaped_reward += hp_diff * 0.5  # Penalize health loss

            # Stage completion bonus - INCREASED
            if current_stage > self.prev_stage:
                logger.info(f"STAGE UP! {self.prev_stage} -> {current_stage}")
                shaped_reward += 100.0  # DOUBLED from 50 to 100
                self.episode_steps = max(0, self.episode_steps - 600)

                # Reset passed positions for new stage
                self.passed_projectile_positions = set()

            # Boss damage bonus
            boss_hp_diff = (
                self.prev_boss_hp - current_boss_hp
                if hasattr(self, "prev_boss_hp")
                else 0
            )
            if boss_hp_diff > 0:
                shaped_reward += boss_hp_diff * 0.5  # Increased from 0.3 to 0.5

            # Movement rewards based on stage - IMPROVED WITH PROGRESS MULTIPLIER
            x_diff = current_x_pos - self.prev_x_pos
            progress_multiplier = 1.0 + (
                0.1 * self.episode_steps / 100
            )  # Grows over time

            if current_stage in [1, 3]:  # Move right in stages 1, 3
                if x_diff > 0:
                    # Increased from 0.1 to 0.5 and add progress multiplier
                    shaped_reward += x_diff * 0.5 * progress_multiplier
            else:  # Move left in stages 0, 2, 4
                if x_diff < 0:
                    shaped_reward += abs(x_diff) * 0.5 * progress_multiplier

            # NEW - BREAKTHROUGH REWARDS FOR MAKING PROGRESS WITH PROJECTILES
            # This approach detects when an agent makes progress while projectiles are present
            # We track progress in fixed-width sections of the stage

            # 30 pixel for each game section
            section_width = 30  # Width of each game section to track
            # current section
            current_section = current_x_pos // section_width
            # prev section
            prev_section = self.prev_x_pos // section_width

            # we track whether we move from one section to another section
            if current_section != prev_section and self.detected_projectiles > 0:
                # we need to move in the right direction
                moving_correctly = (
                    current_stage in [1, 3] and current_section > prev_section
                ) or (  # Moving right in stages 1,3
                    current_stage not in [1, 3] and current_section < prev_section
                )  # Moving left in stages 0,2,4

                if moving_correctly:
                    section_key = f"{current_stage}_{current_section}"
                    if section_key not in self.passed_projectile_positions:
                        # we progress the new section, give big reward
                        self.passed_projectile_positions.add(section_key)
                        self.breakthrough_rewards_given += 1

                        # Progressive reward scaling with more breakthroughs
                        breakthrough_reward = 15.0 * (
                            1 + 0.3 * self.breakthrough_rewards_given
                        )
                        shaped_reward += breakthrough_reward

                        logger.info(
                            f"BREAKTHROUGH: Agent made progress to section {current_section} with projectiles present! +{breakthrough_reward:.1f} reward"
                        )

                        # Reset consecutive defenses to encourage more progress
                        self.consecutive_defenses = 0

            # if we move a lot, reward a lot (even without proj)
            progress_distance = abs(current_x_pos - self.prev_x_pos)
            # reward at > 20 pixel
            if progress_distance > 20:
                # Check if we're moving in the correct direction for the stage
                correct_direction = (
                    current_stage in [1, 3] and current_x_pos > self.prev_x_pos
                ) or (  # Right in stages 1,3
                    current_stage not in [1, 3] and current_x_pos < self.prev_x_pos
                )  # Left in stages 0,2,4

                if correct_direction:
                    progress_reward = 5.0 + (0.1 * progress_distance)
                    shaped_reward += progress_reward
                    logger.debug(
                        f"SIGNIFICANT PROGRESS: Moved {progress_distance} pixels in the right direction! +{progress_reward:.1f} reward"
                    )

            # ENHANCED PROJECTILE AVOIDANCE REWARDS - WITH DYNAMIC SCALING
            # Calculate defensive bias that decreases over time through curriculum learning
            defensive_bias = max(0.5, 1.0 - (self.total_training_steps / 2000000))

            if is_defensive_action and projectile_threat:
                # If we didn't lose health during a defensive action against a projectile
                if hp_diff >= 0:
                    self.successful_defensive_actions += 1
                    self.successful_projectile_avoidance += 1

                    # REDUCED reward for successful projectile avoidance that scales down over time
                    base_reward = 2.0  # Reduced from 5.0 to 2.0

                    # Add decay factor based on consecutive successful avoidances
                    decay_factor = max(
                        0.5, 1.0 - (0.01 * self.successful_projectile_avoidance)
                    )

                    # Apply the defensive bias from curriculum learning
                    final_reward = base_reward * decay_factor * defensive_bias
                    shaped_reward += final_reward

                    logger.debug(
                        f"SUCCESS: {self.KUNGFU_ACTION_NAMES[action]} avoided projectile! "
                        f"Reward +{final_reward:.2f} (bias: {defensive_bias:.2f}, decay: {decay_factor:.2f})"
                    )

                    # Log successful projectile avoidance
                    if self.n_steps % 20 == 0:
                        avoidance_rate = (
                            self.successful_projectile_avoidance
                            / max(1, self.projectile_defensive_actions)
                        ) * 100
                        logger.info(
                            f"Projectile avoidance success rate: {avoidance_rate:.1f}% "
                            f"({self.successful_projectile_avoidance}/{self.projectile_defensive_actions})"
                        )
                else:
                    # Keep penalty for incorrect timing of defensive action
                    shaped_reward -= 1.0
                    logger.debug(
                        f"FAILURE: Defensive action {self.KUNGFU_ACTION_NAMES[action]} failed - took damage. Penalty -1.0"
                    )
            elif is_defensive_action and not projectile_threat:
                # Increased penalty for unnecessary defensive actions as training progresses
                unnecessary_penalty = 0.5 * (
                    1.0 + (self.total_training_steps / 5000000)
                )
                shaped_reward -= unnecessary_penalty

                if self.n_steps % 50 == 0:
                    logger.debug(
                        f"UNNECESSARY: Defensive action {self.KUNGFU_ACTION_NAMES[action]} with no projectile detected. "
                        f"Penalty -{unnecessary_penalty:.2f}"
                    )
            elif is_defensive_action and hp_diff >= 0:
                # Regular defensive action that maintained health
                self.successful_defensive_actions += 1
                shaped_reward += 0.3  # Reduced from 0.5 to 0.3

            # Extra reward for following recommended defensive actions - SCALED DOWN OVER TIME
            if action == recommended_action and recommended_action in [4, 5]:
                recommendation_reward = 1.0 * defensive_bias
                shaped_reward += recommendation_reward
                logger.debug(
                    f"GOOD DECISION: Agent followed recommended defensive action {self.KUNGFU_ACTION_NAMES[action]}. "
                    f"Reward +{recommendation_reward:.2f}"
                )

            # NEW - OFFENSIVE SUCCESS REWARDS
            if is_offensive_action and score_diff > 0:
                # Reward offensive actions that increase score
                shaped_reward += 1.0 + (
                    0.01 * score_diff
                )  # Base reward plus score-based bonus
                logger.debug(
                    f"OFFENSIVE SUCCESS: {self.KUNGFU_ACTION_NAMES[action]} led to score increase! Reward +{1.0 + (0.01 * score_diff):.2f}"
                )

            # Additional stagnation penalty - detect when agent is stuck
            self.stagnation_counter += 1

            # Consider significant movement as progress
            if progress_distance > 10 and correct_direction:
                self.stagnation_counter = 0
                self.last_progress_time = self.episode_steps

            # Apply increasing penalty if agent hasn't made progress in a while
            if self.stagnation_counter > 200:  # No significant progress for 200 steps
                stagnation_penalty = min(5.0, 0.01 * (self.stagnation_counter - 200))
                shaped_reward -= stagnation_penalty

                if self.stagnation_counter % 100 == 0:
                    logger.info(
                        f"STAGNATION PENALTY: No progress for {self.stagnation_counter} steps. Penalty: -{stagnation_penalty:.2f}"
                    )

                # If extremely stuck and doing defensive actions, apply larger penalty
                if self.stagnation_counter > 500 and is_defensive_action:
                    shaped_reward -= 1.0
                    if self.stagnation_counter % 100 == 0:
                        logger.info(
                            "SEVERE STAGNATION: Penalizing defensive actions to encourage exploration"
                        )

            # Time urgency - More aggressive as episode progresses
            time_factor = min(3.0, 1.0 + (self.episode_steps / MAX_EPISODE_STEPS * 2))
            time_penalty = -0.001 * time_factor
            shaped_reward += time_penalty

            # Update previous values
            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_y_pos = current_y_pos
            self.prev_stage = current_stage
            self.prev_boss_hp = current_boss_hp
            self.prev_score = current_score

            # Update info dictionary with enhanced metrics
            info.update(
                {
                    "current_stage": current_stage,
                    "current_score": current_score,
                    "time_remaining": (MAX_EPISODE_STEPS - self.episode_steps) / 30,
                    "defensive_actions": self.defensive_actions,
                    "successful_defensive_actions": self.successful_defensive_actions,
                    "detected_projectiles": self.detected_projectiles,
                    "projectile_defensive_actions": self.projectile_defensive_actions,
                    "successful_projectile_avoidance": self.successful_projectile_avoidance,
                    "recommended_action": recommended_action,
                    "offensive_actions": self.offensive_actions,
                    "successful_offensive_actions": self.successful_offensive_actions,
                    "breakthrough_rewards_given": self.breakthrough_rewards_given,
                    "consecutive_defenses": self.consecutive_defenses,
                    "stagnation_counter": self.stagnation_counter,
                }
            )

            # Calculate success rates
            if self.defensive_actions > 0:
                info["defensive_success_rate"] = (
                    self.successful_defensive_actions / self.defensive_actions
                ) * 100
            else:
                info["defensive_success_rate"] = 0

            if self.projectile_defensive_actions > 0:
                info["projectile_avoidance_rate"] = (
                    self.successful_projectile_avoidance
                    / self.projectile_defensive_actions
                ) * 100
            else:
                info["projectile_avoidance_rate"] = 0

            if self.offensive_actions > 0:
                info["offensive_success_rate"] = (
                    self.successful_offensive_actions / self.offensive_actions
                ) * 100
            else:
                info["offensive_success_rate"] = 0

        except Exception as e:
            logger.error(f"Error in reward shaping: {e}")
            shaped_reward = reward

        # Return with gymnasium API by default
        return obs, shaped_reward, terminated, truncated, info

    def close(self):
        """Properly close environment resources"""
        try:
            # Attempt to close the underlying environment
            if hasattr(self.env, "close"):
                self.env.close()

            # Clear buffers to free memory
            self.raw_observation_buffer = []
            self.action_buffer = []

            # Log closure
            logger.info("EnhancedKungFuMasterEnv closed successfully")
        except Exception as e:
            logger.error(f"Error closing EnhancedKungFuMasterEnv: {e}")

        # Unregister from the environment manager
        try:
            RetroEnvManager.get_instance().unregister_env(self)
        except Exception as e:
            logger.error(f"Error unregistering from RetroEnvManager: {e}")


# a wrapper add proj info (or anything) to obs (network)
class ProjectileAwareWrapper(gym.Wrapper):
    """An improved wrapper that adds projectile information to the observation space"""

    def __init__(self, env, max_projectiles=5):
        super().__init__(env)

        # Original observation space (image)
        self.image_obs_space = self.observation_space

        # Additional features for each projectile:
        # [relative_x, relative_y, velocity_x, velocity_y, size, distance, time_to_impact,
        #  will_hit, target_area, recommended_action]
        self.projectile_features = 10  # Increased from 7 to 10

        # Maximum number of projectiles to track
        self.max_projectiles = max_projectiles

        # Create a hybrid observation space with both images and vector data
        self.observation_space = spaces.Dict(
            {
                "image": self.image_obs_space,
                "projectiles": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_projectiles, self.projectile_features),
                    dtype=np.float32,
                ),
                "projectile_mask": spaces.Box(
                    low=0, high=1, shape=(max_projectiles,), dtype=np.int8
                ),
                "recommended_action": spaces.Box(
                    low=0,
                    high=13,  # Number of actions in Kung Fu Master
                    shape=(1,),
                    dtype=np.int32,
                ),
                "projectile_threat": spaces.Box(  # New: binary indicator of projectile threat
                    low=0,
                    high=1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "time_since_last_threat": spaces.Box(  # New: frames since last threat
                    low=0,
                    high=100,
                    shape=(1,),
                    dtype=np.float32,
                ),
                # NEW: Information about agent progression for better decision making
                "progress_info": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),  # [stage, x_position, stage_progress]
                    dtype=np.float32,
                ),
                # NEW: Defensive vs offensive state tracking
                "action_balance": spaces.Box(
                    low=0,
                    high=100,
                    shape=(2,),  # [consecutive_defenses, offensive_success_rate]
                    dtype=np.float32,
                ),
            }
        )

        # Track additional state for enhanced observations
        self.frames_since_last_threat = 100  # Initialize high
        self.last_recommended_action = 0
        self.projectile_threat_active = False
        self.stage_progress = 0.0  # Track progress through current stage

        # Register with environment manager
        RetroEnvManager.get_instance().register_env(self)

        logger.info(
            f"ProjectileAwareWrapper initialized with max_projectiles={max_projectiles}"
        )

    def reset(self, **kwargs):
        """Reset the environment and return enhanced observation with projectile data"""
        # Reset the underlying environment
        try:
            # For gymnasium/Stable Retro
            obs_result = self.env.reset(**kwargs)

            # Handle different return types
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                obs, info = obs_result
            else:
                obs = obs_result
                info = {}
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            # Create a dummy observation and info
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            info = {}

        # Reset tracking state
        self.frames_since_last_threat = 100
        self.last_recommended_action = 0
        self.projectile_threat_active = False
        self.stage_progress = 0.0

        # Create empty projectile vector data
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)
        projectile_threat = np.zeros(1, dtype=np.int8)
        time_since_last_threat = np.array(
            [self.frames_since_last_threat], dtype=np.float32
        )

        # Get stage and position info
        current_stage = 0
        x_position = 0

        if hasattr(self.env, "get_stage") and callable(self.env.get_stage):
            current_stage = self.env.get_stage()

        if hasattr(self.env, "get_player_position") and callable(
            self.env.get_player_position
        ):
            player_pos = self.env.get_player_position()
            if isinstance(player_pos, tuple) and len(player_pos) == 2:
                x_position = player_pos[0]

        # Initialize progress info
        progress_info = np.array(
            [float(current_stage), float(x_position), 0.0],  # Initial progress is 0
            dtype=np.float32,
        )

        # Initialize action balance
        action_balance = np.array(
            [
                0.0,  # consecutive_defenses
                50.0,  # start with neutral offensive success rate
            ],
            dtype=np.float32,
        )

        # Create enhanced observation
        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
            "projectile_threat": projectile_threat,
            "time_since_last_threat": time_since_last_threat,
            "progress_info": progress_info,
            "action_balance": action_balance,
        }

        # Return based on what we received
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            return enhanced_obs, info
        else:
            return enhanced_obs

    def step(self, action):
        """Step the environment and return enhanced observation with projectile data"""
        # Take step in environment with error handling for different gym versions
        try:
            # Try with gymnasium API (5 return values)
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                # Old gym API
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                # Unexpected format
                raise ValueError(
                    f"Unexpected step result format with {len(step_result)} values"
                )
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            # Return default values as fallback
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            reward = -1.0
            terminated = True
            truncated = True
            info = {"error": str(e)}

        # Ensure info is a dictionary
        if not isinstance(info, dict):
            info = {}

        # Get current player position
        player_position = (
            self.env.get_player_position()
            if hasattr(self.env, "get_player_position")
            else (0, 0)
        )

        # Initialize projectile data with improved features
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)

        # Get stage and progress information
        current_stage = info.get("current_stage", 0)
        if hasattr(self.env, "get_stage") and callable(self.env.get_stage):
            current_stage = self.env.get_stage()

        x_position = (
            player_position[0]
            if isinstance(player_position, tuple) and len(player_position) == 2
            else 0
        )

        # Calculate stage progress (0-100%)
        # Assuming right stages (1,3) progress from 0 to 255, left stages from 255 to 0
        if current_stage in [1, 3]:  # Right-moving stages
            self.stage_progress = min(100.0, (x_position / 255.0) * 100.0)
        else:  # Left-moving stages
            self.stage_progress = min(100.0, ((255.0 - x_position) / 255.0) * 100.0)

        # Create progress info array
        progress_info = np.array(
            [float(current_stage), float(x_position), self.stage_progress],
            dtype=np.float32,
        )

        # Get action balance information
        consecutive_defenses = info.get("consecutive_defenses", 0)
        offensive_success_rate = info.get("offensive_success_rate", 50.0)

        action_balance = np.array(
            [float(consecutive_defenses), float(offensive_success_rate)],
            dtype=np.float32,
        )

        # If we have projectile detector, use it
        projectile_threat_detected = False
        if hasattr(self.env, "projectile_detector"):
            try:
                # Get projectile info
                projectile_info = enhance_observation_with_projectiles(
                    obs, self.env.projectile_detector, player_position
                )

                # Safely get projectiles and recommended action
                projectiles = projectile_info.get("projectiles", [])
                recommended_action[0] = projectile_info.get("recommended_action", 0)

                # Store recommended action
                self.last_recommended_action = recommended_action[0]

                # Check if there's an active threat
                projectile_threat_detected = len(
                    projectiles
                ) > 0 and recommended_action[0] in [4, 5]

                # Update threat timing
                if projectile_threat_detected:
                    self.frames_since_last_threat = 0
                    self.projectile_threat_active = True
                else:
                    self.frames_since_last_threat = min(
                        100, self.frames_since_last_threat + 1
                    )
                    if self.frames_since_last_threat > 10:  # Reset after 10 frames
                        self.projectile_threat_active = False

                # Convert projectile info to feature vectors with enhanced features
                for i, proj in enumerate(projectiles[: self.max_projectiles]):
                    try:
                        # Safely extract projectile position and info
                        position = proj.get("position", (0, 0))
                        if not isinstance(position, tuple) or len(position) != 2:
                            position = (0, 0)

                        velocity = proj.get("velocity", (0, 0))
                        if not isinstance(velocity, tuple) or len(velocity) != 2:
                            velocity = (0, 0)

                        # Extract components safely
                        x, y = position
                        vel_x, vel_y = velocity

                        # Get size safely
                        size = proj.get("size", 0)
                        confidence = proj.get("confidence", 0.5)

                        # Calculate relative position to player
                        player_x, player_y = player_position
                        rel_x = x - player_x
                        rel_y = y - player_y

                        # Calculate distance and estimated time to impact
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        vel_magnitude = max(np.sqrt(vel_x**2 + vel_y**2), 1e-6)
                        time_to_impact = distance / vel_magnitude

                        # Determine if projectile will hit player (new feature)
                        will_hit = 0.0
                        target_area = 0.0  # 0 = none, -1 = upper body, 1 = lower body
                        optimal_action = 0.0  # 0 = none, 4 = jump, 5 = crouch

                        # Check for collision course
                        if (x < player_x and vel_x > 0) or (x > player_x and vel_x < 0):
                            # Calculate time to reach player x-coordinate
                            time_to_x = (
                                abs((player_x - x) / vel_x)
                                if vel_x != 0
                                else float("inf")
                            )

                            # Predict y position at collision
                            future_y = y + (vel_y * time_to_x)

                            # Player hitbox estimation
                            player_height = 40
                            player_top = player_y - player_height
                            player_middle = player_y - player_height / 2
                            player_bottom = player_y

                            # Check if projectile will hit player
                            if (
                                time_to_x < 20
                                and abs(future_y - player_middle)
                                < player_height / 2 + 10
                            ):
                                will_hit = 1.0

                                # Determine which part of the body would be hit
                                if future_y < player_middle:
                                    target_area = -1.0  # Upper body
                                    optimal_action = 5.0  # Crouch
                                else:
                                    target_area = 1.0  # Lower body
                                    optimal_action = 4.0  # Jump

                        # Store enhanced features
                        projectile_data[i] = [
                            rel_x,
                            rel_y,
                            vel_x,
                            vel_y,
                            size,
                            distance,
                            time_to_impact,
                            will_hit,  # New: binary hit prediction
                            target_area,  # New: which part of body would be hit
                            optimal_action,  # New: recommended defensive action
                        ]
                        projectile_mask[i] = 1  # Mark this projectile as valid

                    except Exception as e:
                        logger.error(f"Error processing projectile {i}: {e}")
                        # Skip this projectile
                        continue

            except Exception as e:
                logger.error(f"Error in projectile detection: {e}")

        # Create enhanced observation with threat information
        projectile_threat = np.array(
            [1 if self.projectile_threat_active else 0], dtype=np.int8
        )
        time_since_last_threat = np.array(
            [min(self.frames_since_last_threat, 100)], dtype=np.float32
        )

        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
            "projectile_threat": projectile_threat,
            "time_since_last_threat": time_since_last_threat,
            "progress_info": progress_info,
            "action_balance": action_balance,
        }

        return enhanced_obs, reward, terminated, truncated, info

    def close(self):
        """Properly close the environment"""
        try:
            # Close the wrapped environment
            if hasattr(self.env, "close"):
                self.env.close()

            # Unregister from environment manager
            RetroEnvManager.get_instance().unregister_env(self)
            logger.info("ProjectileAwareWrapper closed successfully")
        except Exception as e:
            logger.error(f"Error closing ProjectileAwareWrapper: {e}")


# CNN for projectile detection with enhanced feature importance
class ProjectileAwareCNN(BaseFeaturesExtractor):
    """
    Improved CNN for processing both game frames and projectile features.

    Gives more weight to projectile information for better defensive actions.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        # First call parent constructor with the provided features_dim
        super().__init__(observation_space, features_dim)

        # Extract dimensions from the image observation space
        if isinstance(observation_space, spaces.Dict):
            image_space = observation_space["image"]
            projectile_space = observation_space["projectiles"]
            n_input_channels = image_space.shape[
                0
            ]  # Number of input channels (e.g., 4 for frame stack)
            image_height, image_width = image_space.shape[1], image_space.shape[2]

            # Calculate projectile feature dimensions
            self.projectile_dim = (
                projectile_space.shape[0] * projectile_space.shape[1]
                if len(projectile_space.shape) > 1
                else projectile_space.shape[0]
            )

            # Add dimensions for additional threat features
            if "projectile_threat" in observation_space.keys():
                self.threat_dim = 1
            else:
                self.threat_dim = 0

            if "time_since_last_threat" in observation_space.keys():
                self.time_dim = 1
            else:
                self.time_dim = 0

            # NEW: Add dimensions for progress and action balance features
            if "progress_info" in observation_space.keys():
                self.progress_dim = observation_space["progress_info"].shape[0]
            else:
                self.progress_dim = 0

            if "action_balance" in observation_space.keys():
                self.balance_dim = observation_space["action_balance"].shape[0]
            else:
                self.balance_dim = 0

            logger.info(
                f"Image space: {image_space.shape}, Projectile space: {projectile_space.shape}"
            )
            logger.info(
                f"Projectile dim: {self.projectile_dim}, Threat dim: {self.threat_dim}, Time dim: {self.time_dim}"
            )
            logger.info(
                f"Progress dim: {self.progress_dim}, Balance dim: {self.balance_dim}"
            )
        else:
            # Fallback for standard observation spaces
            n_input_channels = observation_space.shape[0]
            image_height, image_width = (
                observation_space.shape[1],
                observation_space.shape[2],
            )
            self.projectile_dim = 0
            self.threat_dim = 0
            self.time_dim = 0
            self.progress_dim = 0
            self.balance_dim = 0
            logger.info(f"Standard observation space: {observation_space.shape}")

        # CNN for processing game frames
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output shape by doing one forward pass
        with torch.no_grad():
            try:
                # Try with regular dimensions first
                test_tensor = torch.zeros(
                    1, n_input_channels, image_height, image_width
                )
                n_flatten = self.cnn(test_tensor).shape[1]
                logger.info(
                    f"CNN output shape determined with dimensions {image_height}x{image_width}"
                )
            except RuntimeError:
                try:
                    # If that fails, try with transposed dimensions
                    logger.info(
                        f"Trying transposed dimensions for CNN input: {image_width}x{image_height}"
                    )
                    test_tensor = torch.zeros(
                        1, n_input_channels, image_width, image_height
                    )
                    n_flatten = self.cnn(test_tensor).shape[1]
                    logger.info(
                        f"CNN output shape determined with transposed dimensions"
                    )
                except RuntimeError:
                    # If both fail, use a hardcoded value that's likely to work
                    logger.warning(
                        "Could not determine CNN output shape, using estimated value"
                    )
                    n_flatten = 39936  # Common value for frame stacked observations

        logger.info(f"CNN output features: {n_flatten}")

        # Create separate networks for processing different feature types

        # 1. Network for processing projectile features
        self.projectile_network = (
            nn.Sequential(
                nn.Linear(self.projectile_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            if self.projectile_dim > 0
            else None
        )

        # 2. Network for processing threat indicators
        self.threat_network = (
            nn.Sequential(nn.Linear(self.threat_dim + self.time_dim, 16), nn.ReLU())
            if (self.threat_dim + self.time_dim) > 0
            else None
        )

        # 3. NEW: Network for processing progress information
        self.progress_network = (
            nn.Sequential(
                nn.Linear(self.progress_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
            )
            if self.progress_dim > 0
            else None
        )

        # 4. NEW: Network for processing action balance
        self.balance_network = (
            nn.Sequential(
                nn.Linear(self.balance_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
            )
            if self.balance_dim > 0
            else None
        )

        # Calculate the total input size for final linear layer
        total_features = n_flatten
        if self.projectile_network is not None:
            total_features += 64  # Output from projectile network
        if self.threat_network is not None:
            total_features += 16  # Output from threat network
        if self.progress_network is not None:
            total_features += 16  # Output from progress network
        if self.balance_network is not None:
            total_features += 8  # Output from balance network

        logger.info(f"Total input features to linear layer: {total_features}")

        # Final linear layer to combine all features
        self.linear = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
        )

        # Save dimensions for forward pass
        self.n_input_channels = n_input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.n_flatten = n_flatten

        # Initialize training steps counter for feature weight adjustment
        self.training_steps = 0

        # Register device for consistency
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"ProjectileAwareCNN using device: {self.device}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process image and projectile observations with dynamic weight adjustment.

        :param observations: Dict containing 'image' and other feature tensors
        :return: Tensor of extracted features
        """
        # Increment training steps counter
        self.training_steps += 1

        # Dynamic weight factors based on training progress
        # As training progresses, we reduce emphasis on defensive features
        projectile_weight = max(1.0, 1.3 - (self.training_steps / 2000000 * 0.3))
        threat_weight = max(1.0, 1.5 - (self.training_steps / 2000000 * 0.5))
        progress_weight = min(1.5, 1.0 + (self.training_steps / 2000000 * 0.5))

        # Check if observations is a dict or a tensor
        try:
            if isinstance(observations, dict):
                image_tensor = observations["image"]
            else:
                # If not a dict, assume it's the image tensor directly
                image_tensor = observations

            # Ensure image tensor is not None
            if image_tensor is None:
                raise ValueError("Image tensor is None")

            # Get batch size safely
            if len(image_tensor.shape) >= 1:
                batch_size = image_tensor.shape[0]
            else:
                batch_size = 1

            # Process image features
            try:
                # Try first with standard channel order
                if len(image_tensor.shape) == 4:
                    # Shape could be [batch, channels, height, width] or [batch, height, width, channels]
                    if image_tensor.shape[3] in [3, 12]:  # RGB or stacked RGB frames
                        # Likely [batch, height, width, channels]
                        image_features = self.cnn(
                            image_tensor.permute(0, 3, 1, 2) / 255.0
                        )
                    else:
                        # Already in expected format [batch, channels, height, width]
                        image_features = self.cnn(image_tensor / 255.0)
                elif len(image_tensor.shape) == 3:
                    # Add batch dimension if missing
                    image_features = self.cnn(image_tensor.unsqueeze(0) / 255.0)
                else:
                    # Unexpected shape
                    logger.warning(
                        f"Unexpected image tensor shape: {image_tensor.shape}"
                    )
                    # Create placeholder features
                    image_features = torch.zeros(
                        (batch_size, self.n_flatten), device=image_tensor.device
                    )

            except Exception as e:
                logger.error(f"Error processing image: {e}")
                # Fall back to just image features
                image_features = torch.zeros(
                    (batch_size, self.n_flatten), device=image_tensor.device
                )

            # Initialize combined features with image features
            combined_features = [image_features]

            # Process projectile features if available
            if (
                self.projectile_network is not None
                and isinstance(observations, dict)
                and "projectiles" in observations
                and observations["projectiles"] is not None
            ):
                try:
                    projectile_features = observations["projectiles"]

                    # Apply projectile mask if available
                    if (
                        "projectile_mask" in observations
                        and observations["projectile_mask"] is not None
                    ):
                        mask = observations["projectile_mask"]
                        if len(mask.shape) == 2:  # [batch, num_projectiles]
                            # Expand mask to match projectile features
                            proj_shape = projectile_features.shape
                            if (
                                len(proj_shape) == 3
                            ):  # [batch, num_projectiles, features]
                                expanded_mask = mask.unsqueeze(-1).expand(
                                    -1, -1, proj_shape[2]
                                )
                                # Apply mask (zero out invalid projectiles)
                                projectile_features = (
                                    projectile_features * expanded_mask
                                )

                    # Flatten projectile features if needed
                    if (
                        len(projectile_features.shape) == 3
                    ):  # [batch, num_projectiles, features]
                        projectile_features = projectile_features.reshape(
                            batch_size, -1
                        )
                    elif len(projectile_features.shape) == 1:  # [features]
                        projectile_features = projectile_features.unsqueeze(
                            0
                        )  # Add batch dimension

                    # Process through projectile network
                    # MODIFIED: Dynamic weight based on training progress
                    projectile_output = (
                        self.projectile_network(projectile_features) * projectile_weight
                    )
                    combined_features.append(projectile_output)

                except Exception as e:
                    logger.error(f"Error processing projectile features: {e}")
                    # Create placeholder features
                    placeholder = torch.zeros(
                        (batch_size, 64), device=image_tensor.device
                    )
                    combined_features.append(placeholder)

            # Process threat information if available
            if self.threat_network is not None:
                try:
                    threat_inputs = []

                    if (
                        isinstance(observations, dict)
                        and "projectile_threat" in observations
                        and observations["projectile_threat"] is not None
                    ):
                        threat = observations["projectile_threat"]
                        if len(threat.shape) == 1:
                            threat = threat.unsqueeze(0)
                        threat_inputs.append(threat)

                    if (
                        isinstance(observations, dict)
                        and "time_since_last_threat" in observations
                        and observations["time_since_last_threat"] is not None
                    ):
                        time_since = observations["time_since_last_threat"]
                        if len(time_since.shape) == 1:
                            time_since = time_since.unsqueeze(0)
                        # Normalize time to 0-1 range
                        time_since = time_since / 100.0
                        threat_inputs.append(time_since)

                    if threat_inputs:
                        # Combine threat feature tensors
                        threat_tensor = torch.cat(threat_inputs, dim=1)

                        # Process through threat network with dynamic weight
                        threat_output = (
                            self.threat_network(threat_tensor) * threat_weight
                        )
                        combined_features.append(threat_output)
                    else:
                        # If no threat inputs, create a placeholder
                        placeholder = torch.zeros(
                            (batch_size, 16), device=image_tensor.device
                        )
                        combined_features.append(placeholder)

                except Exception as e:
                    logger.error(f"Error processing threat features: {e}")
                    placeholder = torch.zeros(
                        (batch_size, 16), device=image_tensor.device
                    )
                    combined_features.append(placeholder)

            # NEW: Process progress information if available
            if (
                self.progress_network is not None
                and isinstance(observations, dict)
                and "progress_info" in observations
                and observations["progress_info"] is not None
            ):
                try:
                    progress_features = observations["progress_info"]

                    # Ensure proper shape
                    if len(progress_features.shape) == 1:
                        progress_features = progress_features.unsqueeze(0)

                    # Process through progress network with increasing weight over time
                    progress_output = (
                        self.progress_network(progress_features) * progress_weight
                    )
                    combined_features.append(progress_output)
                except Exception as e:
                    logger.error(f"Error processing progress features: {e}")
                    placeholder = torch.zeros(
                        (batch_size, 16), device=image_tensor.device
                    )
                    combined_features.append(placeholder)

            # NEW: Process action balance information if available
            if (
                self.balance_network is not None
                and isinstance(observations, dict)
                and "action_balance" in observations
                and observations["action_balance"] is not None
            ):
                try:
                    balance_features = observations["action_balance"]

                    # Ensure proper shape
                    if len(balance_features.shape) == 1:
                        balance_features = balance_features.unsqueeze(0)

                    # Process through balance network - weight increases with training
                    balance_output = (
                        self.balance_network(balance_features) * progress_weight
                    )
                    combined_features.append(balance_output)
                except Exception as e:
                    logger.error(f"Error processing action balance features: {e}")
                    placeholder = torch.zeros(
                        (batch_size, 8), device=image_tensor.device
                    )
                    combined_features.append(placeholder)

            # Combine all feature outputs
            try:
                final_combined = torch.cat(combined_features, dim=1)
            except Exception as e:
                logger.error(f"Error combining features: {e}")
                # Fall back to just image features
                final_combined = image_features

            # Process through final layers
            return self.linear(final_combined)

        except Exception as e:
            logger.error(f"Critical error in feature extraction: {e}")
            # Return a zero tensor as fallback
            return torch.zeros((1, self.features_dim), device=self.device)


# Policy that uses the enhanced ProjectileAwareCNN
class ProjectileAwarePolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        *args,
        **kwargs,
    ):
        # Use our enhanced CNN feature extractor
        features_extractor_class = ProjectileAwareCNN
        features_extractor_kwargs = dict(features_dim=256)

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )


def create_enhanced_kungfu_model(env, resume=False, model_path=None):
    """Create a custom PPO model with enhanced projectile awareness"""
    # Default model path if none provided
    if model_path is None:
        model_path = MODEL_PATH

    # Resume from existing model if requested
    if resume and os.path.exists(model_path):
        logger.info(f"Loading existing projectile-aware model from {model_path}")
        try:
            model = PPO.load(model_path, env=env, policy=ProjectileAwarePolicy)
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.info("Creating new model instead")

    # Create new model with custom policy
    logger.info("Creating new projectile-aware model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Enhanced parameters for better learning
    try:
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        model = PPO(
            policy=ProjectileAwarePolicy,
            env=env,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # INCREASED from 0.01 to 0.05 to encourage exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device=device,
            policy_kwargs=dict(
                net_arch=[dict(pi=[64, 64], vf=[64, 64])],  # Deeper networks
                optimizer_class=torch.optim.Adam,
                optimizer_kwargs=dict(eps=1e-5),
            ),
        )
    except TypeError as e:
        logger.error(f"Error with initial PPO parameters: {e}")
        # Try with fewer parameters as a fallback
        model = PPO(
            policy=ProjectileAwarePolicy,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
        )

    return model


def make_enhanced_kungfu_env(
    is_play_mode=False, frame_stack=4, use_projectile_features=True
):
    """Create a Kung Fu Master environment with enhanced projectile detection for Stable Retro"""
    # Get environment manager instance for tracking
    env_manager = RetroEnvManager.get_instance()

    # Environment creation attempts limiter
    max_attempts = 3

    # Create base environment
    env = None
    for attempt in range(max_attempts):
        try:
            import retro

            # Force garbage collection before creating new environment
            gc.collect()

            logger.info(
                f"Attempt {attempt+1}/{max_attempts}: Creating Stable Retro environment"
            )

            # First ensure we have the render mode set correctly
            render_mode = "human" if is_play_mode else None
            logger.info(
                f"Attempting to create Stable Retro environment with render_mode={render_mode}"
            )

            # Create the environment
            if render_mode:
                env = retro.make(
                    game="KungFu-Nes",
                    render_mode=render_mode,
                    inttype=retro.data.Integrations.STABLE,
                )
            else:
                # Try without render mode if not in play mode
                env = retro.make(
                    game="KungFu-Nes", inttype=retro.data.Integrations.STABLE
                )

            # If we reach here, environment creation was successful
            logger.info("Successfully created KungFu-Nes environment")

            # Make sure we register the environment for cleanup
            env_manager.register_env(env)

            # Explicitly reset the environment once before wrapping to test it
            logger.info("Performing initial reset of base environment")
            env.reset()
            logger.info("Initial reset successful")

            # Break out of the loop since we created the environment successfully
            break

        except Exception as e:
            logger.error(f"Error creating environment (attempt {attempt+1}): {e}")

            # Try to clean up
            if env is not None:
                try:
                    env.close()
                except:
                    pass
                env = None

            # Force garbage collection
            gc.collect()

            # Wait before retry
            time.sleep(1)

            # If last attempt, raise the error
            if attempt == max_attempts - 1:
                logger.error("All environment creation attempts failed")
                raise RuntimeError("Could not initialize Stable Retro environment")

    # Check if we have a valid environment
    if env is None:
        raise RuntimeError("Failed to create base environment")

    # Print environment info for debugging
    try:
        logger.info(f"Environment type: {type(env)}")
        logger.info(f"Observation space: {env.observation_space}")
        logger.info(f"Action space: {env.action_space}")
    except Exception as e:
        logger.warning(f"Could not print environment info: {e}")

    # Apply our enhanced wrapper with extra error handling
    logger.info("Applying EnhancedKungFuMasterEnv wrapper")
    try:
        env = EnhancedKungFuMasterEnv(env)
    except Exception as e:
        logger.error(f"Error applying EnhancedKungFuMasterEnv: {e}")
        raise

    # Set up monitoring with our resilient monitor
    try:
        os.makedirs("logs", exist_ok=True)
        monitor_path = os.path.join("logs", "kungfu")
        logger.info(f"Setting up monitoring at {monitor_path}")
        env = ResilientMonitor(env, monitor_path)
    except Exception as e:
        logger.warning(f"Could not set up monitoring: {e}")

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    try:
        logger.info("Wrapping with DummyVecEnv")
        env = DummyVecEnv([lambda: env])
        # Test the wrapped environment
        logger.info("Testing DummyVecEnv wrapper")
        test_obs = env.reset()
        logger.info(
            f"DummyVecEnv reset successful, observation shape: {test_obs.shape if hasattr(test_obs, 'shape') else 'N/A'}"
        )
    except Exception as e:
        logger.error(f"Error wrapping with DummyVecEnv: {e}")
        raise

    # Apply frame stacking - we need more frames for better projectile detection
    try:
        logger.info(
            f"Using enhanced frame stacking with n_stack={frame_stack} for improved projectile detection"
        )
        env = VecFrameStack(env, n_stack=frame_stack)

        # Test the frame-stacked environment
        logger.info("Testing VecFrameStack wrapper")
        test_obs = env.reset()
        logger.info(
            f"VecFrameStack reset successful, observation shape: {test_obs.shape if hasattr(test_obs, 'shape') else 'N/A'}"
        )
    except Exception as e:
        logger.error(f"Error applying frame stacking: {e}")
        raise

    # Always add projectile features wrapper
    if use_projectile_features:
        try:
            logger.info(
                "Adding projectile feature wrapper for explicit projectile information"
            )
            env = wrap_projectile_aware(env, max_projectiles=5)

            # Test the projectile-aware environment
            logger.info("Testing ProjectileAwareWrapper")
            test_obs = env.reset()
            if isinstance(test_obs, dict):
                logger.info(
                    f"ProjectileAwareWrapper reset successful, observation keys: {test_obs.keys()}"
                )
            else:
                logger.info(
                    f"ProjectileAwareWrapper reset returned unexpected type: {type(test_obs)}"
                )
        except Exception as e:
            logger.warning(f"Could not add projectile features wrapper: {e}")
            logger.info("Continuing without projectile features")

    return env


# Helper function to wrap environment with projectile awareness
def wrap_projectile_aware(env, max_projectiles=5):
    """Wrap an environment to add projectile awareness"""
    # Handle VecEnv case - unwrap to get the base env
    if isinstance(env, VecEnv):
        # Get the base environment without closing the VecEnv
        base_env = env.envs[0]

        # Apply our wrapper to the unwrapped base environment
        logger.info(f"Wrapping base environment of type {type(base_env)}")
        wrapped_env = ProjectileAwareWrapper(base_env, max_projectiles=max_projectiles)

        # Create a new VecEnv with our wrapped environment
        logger.info("Creating new DummyVecEnv with the ProjectileAwareWrapper")
        new_env = DummyVecEnv([lambda: wrapped_env])

        return new_env
    else:
        # Apply wrapper directly
        logger.info(f"Directly wrapping environment of type {type(env)}")
        return ProjectileAwareWrapper(env, max_projectiles=max_projectiles)
