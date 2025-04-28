import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3 import PPO
from gymnasium import spaces
from typing import Dict, List, Tuple, Type, Union, Optional
import tempfile

# Import projectile detector
from projectile import OpenCVProjectileDetector, enhance_observation_with_projectiles

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

# Set default model path
MODEL_PATH = "model/kungfu.zip"


# Custom Monitor class that's more resilient
class ResilientMonitor(Monitor):
    """A more fault-tolerant version of Monitor that won't crash on I/O errors"""

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
                print("Warning: Monitor file I/O error. Reopening file.")
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
            print(f"Warning: Failed to set up results writer: {e}")
            # Use a dummy file as a fallback
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".monitor.csv")
            print(f"Using temporary monitor file: {temp_file.name}")
            self.file_handler = temp_file
            from stable_baselines3.common.monitor import ResultsWriter

            self.results_writer = ResultsWriter(
                self.file_handler.name,
                header={"t_start": self.t_start, **self.metadata},
                extra_keys=self.info_keywords,
            )


# Enhanced environment wrapper
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

        # Enhanced metrics for projectile avoidance
        self.detected_projectiles = 0
        self.defensive_actions = 0
        self.projectile_defensive_actions = 0
        self.successful_defensive_actions = 0
        self.successful_projectile_avoidance = 0

        # OpenCV-based projectile detector with adjusted parameters
        self.projectile_detector = OpenCVProjectileDetector(min_size=4, max_size=30)

        # Raw observation buffer for projectile detection
        self.raw_observation_buffer = []
        self.max_buffer_size = 4  # Store the last 4 frames

        print("Enhanced KungFuMasterEnv initialized with OpenCV projectile detection")

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
            print("Warning: Unable to access RAM through known methods")
            return np.zeros(0x10000, dtype=np.uint8)
        except Exception as e:
            print(f"Error accessing RAM: {e}")
            return np.zeros(0x10000, dtype=np.uint8)

    def get_ram_value(self, address):
        """Get a value from RAM at the specified address"""
        try:
            ram = self.get_ram()
            if ram is not None and len(ram) > address:
                return int(ram[address])
            return 0
        except Exception as e:
            print(f"Error accessing RAM at address 0x{address:04x}: {e}")
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
            print(f"Error getting player position: {e}")
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
            print(f"Error getting score: {e}")
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
            print(f"Error in reset: {e}")
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

        # Log defensive timing stats from previous episode
        if self.n_steps > 0:  # Only log if not the first episode
            projectile_avoidance_rate = 0
            if self.projectile_defensive_actions > 0:
                projectile_avoidance_rate = (
                    self.successful_projectile_avoidance
                    / self.projectile_defensive_actions
                ) * 100
                print(
                    f"Episode projectile stats - Detected: {self.detected_projectiles}, "
                    f"Defensive actions: {self.projectile_defensive_actions}, "
                    f"Successful avoidance: {self.successful_projectile_avoidance}, "
                    f"Avoidance rate: {projectile_avoidance_rate:.1f}%"
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

        print(
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
            print(
                "Warning: step() called before reset(). Attempting to reset the environment."
            )
            self.reset()

        # Convert to actual action
        try:
            converted_action = self.KUNGFU_ACTIONS[action]
        except Exception as e:
            print(f"Error converting action {action}: {e}")
            converted_action = self.KUNGFU_ACTIONS[0]  # Default to no-op

        # Get current hp and current position before step
        current_hp = self.get_hp()
        player_position = self.get_player_position()
        if not isinstance(player_position, tuple) or len(player_position) != 2:
            print(f"Warning: Invalid player position format: {player_position}")
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

                # Enhanced logging for projectile threats
                if recommended_action in [4, 5]:  # Jump or Crouch
                    action_name = self.KUNGFU_ACTION_NAMES[recommended_action]
                    print(
                        f"⚠️ DEFENSIVE ACTION REQUIRED: {action_name} - {len(image_projectiles)} projectiles detected!"
                    )
            except Exception as e:
                print(f"Error in projectile detection: {e}")

        # Determine if there's an active projectile threat
        projectile_threat = len(image_projectiles) > 0

        if projectile_threat:
            self.detected_projectiles += 1

        # Track defensive actions
        is_defensive_action = action == 4 or action == 5  # Jump or Crouch

        if is_defensive_action:
            self.defensive_actions += 1
            if projectile_threat:
                self.projectile_defensive_actions += 1

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
                print(
                    f"Warning: Unexpected step result format with {len(step_result)} values"
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
            print(f"Error during environment step: {e}")
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
                print(
                    f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                    f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s"
                )

            # Increment step counter
            self.n_steps += 1

            # Shape the reward
            shaped_reward = reward

            # Score reward
            score_diff = current_score - self.prev_score
            if score_diff > 0:
                shaped_reward += score_diff * 0.05
                if score_diff >= 100:
                    print(f"Score increased by {score_diff}! Current: {current_score}")

            # HP loss penalty
            hp_diff = current_hp - self.prev_hp
            if hp_diff < 0 and current_hp < 200:  # Normal health loss
                shaped_reward += hp_diff * 0.5

            # Stage completion bonus
            if current_stage > self.prev_stage:
                print(f"Stage up! {self.prev_stage} -> {current_stage}")
                shaped_reward += 50.0
                self.episode_steps = max(0, self.episode_steps - 600)

            # Boss damage bonus
            boss_hp_diff = (
                self.prev_boss_hp - current_boss_hp
                if hasattr(self, "prev_boss_hp")
                else 0
            )
            if boss_hp_diff > 0:
                shaped_reward += boss_hp_diff * 0.3

            # Movement rewards based on stage
            x_diff = current_x_pos - self.prev_x_pos

            if current_stage in [1, 3]:  # Move right in stages 1, 3
                if x_diff > 0:
                    shaped_reward += x_diff * 0.1
            else:  # Move left in stages 0, 2, 4
                if x_diff < 0:
                    shaped_reward += abs(x_diff) * 0.1

            # ENHANCED PROJECTILE AVOIDANCE REWARDS - MORE STRICT VALIDATION
            if is_defensive_action and projectile_threat:
                # If we didn't lose health during a defensive action against a projectile
                if hp_diff >= 0:
                    self.successful_defensive_actions += 1
                    self.successful_projectile_avoidance += 1

                    # Significant reward for successful projectile avoidance
                    shaped_reward += 1.0

                    # Log successful projectile avoidance occasionally
                    if self.n_steps % 50 == 0:
                        avoidance_rate = (
                            self.successful_projectile_avoidance
                            / max(1, self.projectile_defensive_actions)
                        ) * 100
                        print(
                            f"Projectile avoidance success rate: {avoidance_rate:.1f}% "
                            f"({self.successful_projectile_avoidance}/{self.projectile_defensive_actions})"
                        )
                else:
                    # Penalty for incorrect timing of defensive action
                    shaped_reward -= 0.2
                    print("Defensive action against projectile failed - took damage")
            elif is_defensive_action and not projectile_threat:
                # Penalty for defensive action when no threat exists
                shaped_reward -= 0.1
                if self.n_steps % 100 == 0:
                    print("Defensive action with no projectile detected")
            elif is_defensive_action and hp_diff >= 0:
                # Regular defensive action that maintained health
                self.successful_defensive_actions += 1
                shaped_reward += 0.3

            # Death penalty
            if current_hp == 0 and self.prev_hp > 0:
                shaped_reward -= 10.0
                print("Agent died! Applying penalty.")

            # Time urgency
            time_penalty = -0.001 * (self.episode_steps / MAX_EPISODE_STEPS)
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

        except Exception as e:
            print(f"Error in reward shaping: {e}")
            shaped_reward = reward

        # Return with gymnasium API by default
        return obs, shaped_reward, terminated, truncated, info


# a wrapper add proj info (or anything) to obs (network)
class ProjectileAwareWrapper(gym.Wrapper):
    """A wrapper that adds projectile information to the observation space"""

    def __init__(self, env, max_projectiles=5):
        super().__init__(env)

        # Original observation space (image)
        self.image_obs_space = self.observation_space

        # Additional features for each projectile:
        # [relative_x, relative_y, velocity_x, velocity_y, size, distance, time_to_impact]
        self.projectile_features = 7

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
            }
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
            print(f"Error in reset: {e}")
            # Create a dummy observation and info
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            info = {}

        # Create empty projectile vector data
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)

        # Create enhanced observation
        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
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
            try:
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
            except ValueError as e:
                print(f"Step format error: {e}")
                obs = np.zeros((224, 240, 3), dtype=np.uint8)
                reward = -1.0
                terminated = True
                truncated = True
                info = {"error": str(e)}

        except Exception as e:
            print(f"Error during environment step: {e}")
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

        # Initialize projectile data
        projectile_data = np.zeros(
            (self.max_projectiles, self.projectile_features), dtype=np.float32
        )
        projectile_mask = np.zeros(self.max_projectiles, dtype=np.int8)
        recommended_action = np.zeros(1, dtype=np.int32)

        # If we have projectile detector, use it
        if hasattr(self.env, "projectile_detector"):
            try:
                # Get projectile info
                projectile_info = enhance_observation_with_projectiles(
                    obs, self.env.projectile_detector, player_position
                )

                # Safely get projectiles and recommended action
                projectiles = projectile_info.get("projectiles", [])
                recommended_action[0] = projectile_info.get("recommended_action", 0)

                # Convert projectile info to feature vectors
                # Each projectile: [relative_x, relative_y, velocity_x, velocity_y, size, distance, time_to_impact]
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

                        # Calculate relative position to player
                        player_x, player_y = player_position
                        rel_x = x - player_x
                        rel_y = y - player_y

                        # Calculate distance and estimated time to impact
                        distance = np.sqrt(rel_x**2 + rel_y**2)
                        vel_magnitude = max(np.sqrt(vel_x**2 + vel_y**2), 1e-6)
                        time_to_impact = distance / vel_magnitude

                        # Store features
                        projectile_data[i] = [
                            rel_x,
                            rel_y,
                            vel_x,
                            vel_y,
                            size,
                            distance,
                            time_to_impact,
                        ]
                        projectile_mask[i] = 1  # Mark this projectile as valid

                    except Exception as e:
                        print(
                            f"Error processing projectile {i}: {e}, projectile data: {proj}"
                        )
                        # Skip this projectile
                        continue

            except Exception as e:
                print(f"Error in projectile detection: {e}")

        # Create enhanced observation
        enhanced_obs = {
            "image": obs,
            "projectiles": projectile_data,
            "projectile_mask": projectile_mask,
            "recommended_action": recommended_action,
        }

        return enhanced_obs, reward, terminated, truncated, info


# CNN for projectile detection
class ProjectileAwareCNN(BaseFeaturesExtractor):
    """
    CNN for processing both game frames and projectile features.

    :param observation_space: Observation space
    :param features_dim: Number of features to extract
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
            self.projectile_dim = (
                projectile_space.shape[0] * projectile_space.shape[1]
                if len(projectile_space.shape) > 1
                else projectile_space.shape[0]
            )
            print(
                f"Image space: {image_space.shape}, Projectile space: {projectile_space.shape}"
            )
            print(f"Projectile dim calculated: {self.projectile_dim}")
        else:
            # Fallback for standard observation spaces
            n_input_channels = observation_space.shape[0]
            image_height, image_width = (
                observation_space.shape[1],
                observation_space.shape[2],
            )
            self.projectile_dim = 0
            print(f"Standard observation space: {observation_space.shape}")

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
                print(
                    f"CNN output shape determined with dimensions {image_height}x{image_width}"
                )
            except RuntimeError:
                try:
                    # If that fails, try with transposed dimensions
                    print(
                        f"Trying transposed dimensions for CNN input: {image_width}x{image_height}"
                    )
                    test_tensor = torch.zeros(
                        1, n_input_channels, image_width, image_height
                    )
                    n_flatten = self.cnn(test_tensor).shape[1]
                    print(f"CNN output shape determined with transposed dimensions")
                except RuntimeError:
                    # If both fail, use a hardcoded value that's likely to work
                    print("Could not determine CNN output shape, using estimated value")
                    n_flatten = 39936  # Common value for frame stacked observations

        print(f"CNN output features: {n_flatten}")

        # Linear layer for combining CNN features with projectile features
        # Calculate the total input size
        total_features = n_flatten + self.projectile_dim
        print(f"Total input features to linear layer: {total_features}")

        # Create the linear layer to combine features
        self.linear = nn.Sequential(
            nn.Linear(total_features, features_dim),
            nn.ReLU(),
        )

        # Save dimensions for forward pass
        self.n_input_channels = n_input_channels
        self.image_height = image_height
        self.image_width = image_width
        self.n_flatten = n_flatten

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Process image and projectile observations

        :param observations: Dict containing 'image' and 'projectiles' tensors
        :return: Tensor of extracted features
        """
        # Check if observations is a dict or a tensor
        if isinstance(observations, dict):
            image_tensor = observations["image"]
        else:
            # If not a dict, assume it's the image tensor directly
            image_tensor = observations

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
                if image_tensor.shape[3] < image_tensor.shape[1]:
                    # Likely [batch, height, width, channels]
                    image_features = self.cnn(image_tensor.permute(0, 3, 1, 2) / 255.0)
                else:
                    # Already in expected format [batch, channels, height, width]
                    image_features = self.cnn(image_tensor / 255.0)
            elif len(image_tensor.shape) == 3:
                # Add batch dimension if missing
                image_features = self.cnn(image_tensor.unsqueeze(0) / 255.0)
            else:
                # Unexpected shape
                print(f"Warning: Unexpected image tensor shape: {image_tensor.shape}")
                # Create placeholder features
                image_features = torch.zeros(
                    (batch_size, self.n_flatten), device=image_tensor.device
                )

        except Exception as e:
            print(f"Error processing image: {e}")
            # Create placeholder features
            image_features = torch.zeros(
                (batch_size, self.n_flatten), device=image_tensor.device
            )

        # Handle projectile features or create zeros if not available
        try:
            if isinstance(observations, dict) and "projectiles" in observations:
                projectile_features = observations["projectiles"]

                # Flatten projectile features if needed
                if len(projectile_features.shape) == 3:  # [batch, time, features]
                    projectile_features = projectile_features.reshape(batch_size, -1)
                elif len(projectile_features.shape) == 1:  # [features]
                    projectile_features = projectile_features.unsqueeze(
                        0
                    )  # Add batch dimension
            else:
                # Create zero tensor for projectile features if not available
                projectile_features = torch.zeros(
                    (batch_size, self.projectile_dim), device=image_tensor.device
                )

        except Exception as e:
            print(f"Error processing projectiles: {e}")
            projectile_features = torch.zeros(
                (batch_size, self.projectile_dim), device=image_tensor.device
            )

        # Ensure both tensors have the same batch dimension
        if image_features.shape[0] != projectile_features.shape[0]:
            print(
                f"Batch size mismatch: image={image_features.shape[0]}, projectile={projectile_features.shape[0]}"
            )
            # Fix batch size mismatch
            if image_features.shape[0] > projectile_features.shape[0]:
                projectile_features = projectile_features.expand(
                    image_features.shape[0], -1
                )
            else:
                image_features = image_features.expand(projectile_features.shape[0], -1)

        # Check if projectile_features needs resizing to match expected dimensions
        if projectile_features.shape[1] != self.projectile_dim:
            print(
                f"Projectile feature dim mismatch: got {projectile_features.shape[1]}, expected {self.projectile_dim}"
            )
            # Resize to match expected dimensions
            if projectile_features.shape[1] < self.projectile_dim:
                # Pad with zeros if smaller
                padding = torch.zeros(
                    (batch_size, self.projectile_dim - projectile_features.shape[1]),
                    device=projectile_features.device,
                )
                projectile_features = torch.cat([projectile_features, padding], dim=1)
            else:
                # Truncate if larger
                projectile_features = projectile_features[:, : self.projectile_dim]

        # Combine features
        combined_features = torch.cat([image_features, projectile_features], dim=1)

        # Verify the combined shape matches what our linear layer expects
        expected_input_size = self.linear[0].in_features
        actual_input_size = combined_features.shape[1]

        if expected_input_size != actual_input_size:
            print(
                f"Linear input size mismatch: got {actual_input_size}, expected {expected_input_size}"
            )
            if actual_input_size < expected_input_size:
                # Pad with zeros if too small
                padding = torch.zeros(
                    (batch_size, expected_input_size - actual_input_size),
                    device=combined_features.device,
                )
                combined_features = torch.cat([combined_features, padding], dim=1)
            else:
                # Truncate if too large
                combined_features = combined_features[:, :expected_input_size]

        # Process through final layers
        return self.linear(combined_features)


# Policy that uses the ProjectileAwareCNN
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
        # Use our custom CNN feature extractor
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
    """Create a custom PPO model with projectile awareness"""
    # Default model path if none provided
    if model_path is None:
        model_path = "model/kungfu_projectile_model.zip"

    # Resume from existing model if requested
    if resume and os.path.exists(model_path):
        print(f"Loading existing projectile-aware model from {model_path}")
        try:
            model = PPO.load(model_path, env=env, policy=ProjectileAwarePolicy)
            print("Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating new model instead")

    # Create new model with custom policy
    print("Creating new projectile-aware model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Simplified initialization - compatible with older versions of stable-baselines3
    try:
        # Try with the most common parameters first
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
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
            device=device,
        )
    except TypeError as e:
        print(f"Error with initial PPO parameters: {e}")
        # Try with even fewer parameters as a fallback
        model = PPO(
            policy=ProjectileAwarePolicy,
            env=env,
            learning_rate=0.0001,
            tensorboard_log="./logs/tensorboard/",
            verbose=1,
        )

    return model


def make_enhanced_kungfu_env(
    is_play_mode=False, frame_stack=4, use_projectile_features=False
):
    """Create a Kung Fu Master environment with enhanced projectile detection for Stable Retro"""

    # Create base environment
    env = None
    try:
        import retro

        print("Using Stable Retro library")

        # First ensure we have the render mode set correctly
        render_mode = "human" if is_play_mode else None
        print(
            f"Attempting to create Stable Retro environment with render_mode={render_mode}"
        )

        # Try with full error handling
        try:
            # Try with explicit render mode first
            if render_mode:
                env = retro.make(game="KungFu-Nes", render_mode=render_mode)
            else:
                # Try without render mode if not in play mode
                env = retro.make(game="KungFu-Nes")
            print("Successfully created KungFu-Nes environment")
        except Exception as e1:
            print(f"Failed to create KungFu-Nes environment: {e1}")
            try:
                # Try alternate game name
                if render_mode:
                    env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)
                else:
                    env = retro.make(game="KungFuMaster-Nes")
                print("Successfully created KungFuMaster-Nes environment")
            except Exception as e2:
                print(f"Failed to create KungFuMaster-Nes environment: {e2}")

                # Last resort - try with minimal parameters
                try:
                    # No render mode specified
                    env = retro.make("KungFu-Nes")
                    if is_play_mode and hasattr(env, "render"):
                        print("Using manual rendering mode")
                        env.render()
                    print("Created environment with basic parameters")
                except Exception as e3:
                    print(f"All environment creation attempts failed: {e1}, {e2}, {e3}")
                    raise RuntimeError("Could not initialize Stable Retro environment")

        # Explicitly reset the environment once before wrapping to test it
        if env is not None:
            print("Performing initial reset of base environment")
            try:
                env.reset()
                print("Initial reset successful")
            except Exception as e:
                print(f"Warning: Initial reset failed but continuing: {e}")

    except Exception as e:
        print(f"Fatal error creating environment: {e}")
        raise

    # Check if we have a valid environment
    if env is None:
        raise RuntimeError("Failed to create base environment")

    # Print environment info for debugging
    try:
        print(f"Environment type: {type(env)}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    except Exception as e:
        print(f"Warning: Could not print environment info: {e}")

    # Apply our enhanced wrapper with extra error handling
    print("Applying EnhancedKungFuMasterEnv wrapper")
    try:
        env = EnhancedKungFuMasterEnv(env)
    except Exception as e:
        print(f"Error applying EnhancedKungFuMasterEnv: {e}")
        raise

    # Set up monitoring with our resilient monitor
    try:
        os.makedirs("logs", exist_ok=True)
        monitor_path = os.path.join("logs", "kungfu")
        print(f"Setting up monitoring at {monitor_path}")
        env = ResilientMonitor(env, monitor_path)
    except Exception as e:
        print(f"Warning: Could not set up monitoring: {e}")

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    try:
        print("Wrapping with DummyVecEnv")
        env = DummyVecEnv([lambda: env])
        # Test the wrapped environment
        print("Testing DummyVecEnv wrapper")
        test_obs = env.reset()
        print(
            f"DummyVecEnv reset successful, observation shape: {test_obs.shape if hasattr(test_obs, 'shape') else 'N/A'}"
        )
    except Exception as e:
        print(f"Error wrapping with DummyVecEnv: {e}")
        raise

    # Apply frame stacking - we need more frames for better projectile detection
    try:
        print(
            f"Using enhanced frame stacking with n_stack={frame_stack} for improved projectile detection"
        )
        env = VecFrameStack(env, n_stack=frame_stack)

        # Test the frame-stacked environment
        print("Testing VecFrameStack wrapper")
        test_obs = env.reset()
        print(
            f"VecFrameStack reset successful, observation shape: {test_obs.shape if hasattr(test_obs, 'shape') else 'N/A'}"
        )
    except Exception as e:
        print(f"Error applying frame stacking: {e}")
        raise

    # Optionally add projectile features wrapper
    if use_projectile_features:
        try:
            print(
                "Adding projectile feature wrapper for explicit projectile information"
            )
            env = wrap_projectile_aware(env, max_projectiles=5)

            # Test the projectile-aware environment
            print("Testing ProjectileAwareWrapper")
            test_obs = env.reset()
            if isinstance(test_obs, dict):
                print(
                    f"ProjectileAwareWrapper reset successful, observation keys: {test_obs.keys()}"
                )
            else:
                print(
                    f"ProjectileAwareWrapper reset returned unexpected type: {type(test_obs)}"
                )
        except Exception as e:
            print(f"Warning: Could not add projectile features wrapper: {e}")
            print("Continuing without projectile features")

    return env


# Helper function to wrap environment with projectile awareness
def wrap_projectile_aware(env, max_projectiles=5):
    """Wrap an environment to add projectile awareness"""
    # Handle VecEnv case - unwrap to get the base env
    if isinstance(env, VecEnv):
        # Get the base environment without closing the VecEnv
        base_env = env.envs[0]

        # Apply our wrapper to the unwrapped base environment
        print(f"Wrapping base environment of type {type(base_env)}")
        wrapped_env = ProjectileAwareWrapper(base_env, max_projectiles=max_projectiles)

        # Create a new VecEnv with our wrapped environment
        print("Creating new DummyVecEnv with the ProjectileAwareWrapper")
        new_env = DummyVecEnv([lambda: wrapped_env])

        return new_env
    else:
        # Apply wrapper directly
        print(f"Directly wrapping environment of type {type(env)}")
        return ProjectileAwareWrapper(env, max_projectiles=max_projectiles)
