import numpy as np
import gymnasium as gym
import retro  # Use stable_retro instead of retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os
import tempfile

# projectile detector
from projectile_detection import (
    ProjectileDetector,
    enhance_observation_with_projectiles,
)

# Define the Kung Fu Master action space (same as original)
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

# Action names (same as original)
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

# Critical memory addresses - REMOVED incorrect projectile addresses
MEMORY = {
    "current_stage": 0x0058,  # Current Stage
    "player_hp": 0x04A6,  # Hero HP
    "player_x": 0x0094,  # Hero Screen Pos X
    "player_y": 0x0097,  # Hero Screen Pos Y
    "game_mode": 0x0051,  # Game Mode
    "boss_hp": 0x04A5,  # Boss HP
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes


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
        self.projectile_defensive_actions = (
            0  # Defensive actions taken when projectile detected
        )
        self.successful_defensive_actions = 0
        self.successful_projectile_avoidance = (
            0  # Successful avoidance of detected projectiles
        )

        # projectile detector
        self.projectile_detector = ProjectileDetector()

        # Raw observation buffer for projectile detection
        self.raw_observation_buffer = []
        self.max_buffer_size = 8  # Store the last 8 frames

        print("Enhanced KungFuMasterEnv initialized with projectile detection")

    def _buffer_raw_observation(self, obs):
        """Store raw observations for projectile detection"""
        # If buffer is at max size, remove oldest frame
        if len(self.raw_observation_buffer) >= self.max_buffer_size:
            self.raw_observation_buffer.pop(0)

        # Add new observation
        self.raw_observation_buffer.append(obs)

    def get_ram_value(self, address):
        """Get a value from RAM at the specified address"""
        try:
            ram = self.env.get_ram()
            return int(ram[address])
        except:
            return 0

    def get_stage(self):
        """Get current stage"""
        return self.get_ram_value(MEMORY["current_stage"])

    def get_hp(self):
        """Get current HP"""
        return self.get_ram_value(MEMORY["player_hp"])

    def get_player_position(self):
        """Get player position as (x, y) tuple"""
        x = self.get_ram_value(MEMORY["player_x"])
        y = self.get_ram_value(MEMORY["player_y"])
        return (x, y)

    def get_score(self):
        """Get score from RAM"""
        try:
            ram = self.env.get_ram()
            score = 0
            for i, addr in enumerate(MEMORY["score"]):
                digit = int(ram[addr])
                score += digit * (10 ** (4 - i))
            return score
        except:
            return 0

    def reset(self, **kwargs):
        # Set flag that reset has been called
        self.reset_called = True

        # Reset the environment - specifically for Stable Retro
        try:
            obs, info = self.env.reset(**kwargs)
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
        projectile_avoidance_rate = 0
        if self.projectile_defensive_actions > 0:
            projectile_avoidance_rate = (
                self.successful_projectile_avoidance / self.projectile_defensive_actions
            ) * 100
            print(
                f"Episode projectile stats - Detected: {self.detected_projectiles}, "
                f"Defensive actions: {self.projectile_defensive_actions}, "
                f"Successful avoidance: {self.successful_projectile_avoidance}, "
                f"Avoidance rate: {projectile_avoidance_rate:.1f}%"
            )

        # Clear observation buffer
        self.raw_observation_buffer = []

        # Simple game start - press START a few times
        try:
            # Only press START if reset() worked correctly
            # (this prevents the error we were seeing)
            for _ in range(5):
                # Only try to press START if the environment seems ready
                if hasattr(self.env, "step"):
                    try:
                        self.env.step(KUNGFU_ACTIONS[3])  # Press START
                    except Exception as step_err:
                        print(
                            f"Warning: Could not press START button during reset: {step_err}"
                        )
                        break
        except Exception as e:
            print(f"Warning: Could not press START button during reset: {str(e)}")

        # Get initial state
        try:
            ram = self.env.get_ram()
            self.prev_hp = int(ram[MEMORY["player_hp"]])
            self.prev_x_pos = int(ram[MEMORY["player_x"]])
            self.prev_y_pos = int(ram[MEMORY["player_y"]])
            self.prev_stage = int(ram[MEMORY["current_stage"]])
            self.prev_boss_hp = int(ram[MEMORY["boss_hp"]])
            self.prev_score = self.get_score()

            print(
                f"Initial state - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
                f"Pos: ({self.prev_x_pos}, {self.prev_y_pos}), Score: {self.prev_score}"
            )
        except Exception as e:
            print(f"Error getting initial state: {str(e)}")
            # Initialize with default values if we can't read RAM
            self.prev_hp = 0
            self.prev_x_pos = 0
            self.prev_y_pos = 0
            self.prev_stage = 0
            self.prev_boss_hp = 0
            self.prev_score = 0

        # Buffer the initial observation
        self._buffer_raw_observation(obs)

        return obs, info

    def step(self, action):
        # Check if reset has been called
        if not self.reset_called:
            print(
                "Warning: step() called before reset(). Attempting to reset the environment."
            )
            self.reset()

        # convet to actual action
        converted_action = self.KUNGFU_ACTIONS[action]

        # get current hp and current position
        current_hp = self.get_hp()
        player_position = self.get_player_position()

        # many frames inside image projectile
        image_projectiles = []
        # projectile info
        projectile_info = None
        # recomend action
        recommended_action = 0

        # obs >= 2
        if len(self.raw_observation_buffer) >= 2:
            # Convert buffer to numpy array for projectile detection
            frame_stack = np.array(self.raw_observation_buffer)
            # Detect projectiles from frame differences
            projectile_info = enhance_observation_with_projectiles(
                frame_stack, self.projectile_detector, player_position
            )
            image_projectiles = projectile_info["projectiles"]
            recommended_action = projectile_info["recommended_action"]

        # Determine if there's an active projectile threat - now based only on image detection
        projectile_threat = len(image_projectiles) > 0

        if projectile_threat:
            self.detected_projectiles += 1

        # Track defensive actions
        is_defensive_action = action == 4 or action == 5  # Jump or Crouch

        if is_defensive_action:
            self.defensive_actions += 1
            if projectile_threat:
                self.projectile_defensive_actions += 1

                # Check if the action matches the recommended action
                if recommended_action > 0 and action == recommended_action:
                    # Bonus for taking the specifically recommended defensive action
                    action_bonus = 0.5
                else:
                    action_bonus = 0.1

        # Take step in environment
        try:
            obs, reward, terminated, truncated, info = self.env.step(converted_action)
        except Exception as e:
            print(f"Error during environment step: {str(e)}")
            # Return default values as fallback
            return (
                (
                    np.zeros_like(self.raw_observation_buffer[-1])
                    if self.raw_observation_buffer
                    else np.zeros((224, 240, 3), dtype=np.uint8)
                ),
                -1.0,  # Negative reward for error
                True,  # Terminate episode
                True,  # Truncate episode
                {"error": str(e)},
            )

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
            ram = self.env.get_ram()
            current_hp = int(ram[MEMORY["player_hp"]])
            current_x_pos = int(ram[MEMORY["player_x"]])
            current_y_pos = int(ram[MEMORY["player_y"]])
            current_stage = int(ram[MEMORY["current_stage"]])
            current_boss_hp = int(ram[MEMORY["boss_hp"]])
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

            # ENHANCED PROJECTILE AVOIDANCE REWARDS
            if is_defensive_action and projectile_threat:
                # If we didn't lose health during a defensive action against a projectile
                if hp_diff >= 0:
                    self.successful_defensive_actions += 1
                    self.successful_projectile_avoidance += 1

                    # Significant reward for successful projectile avoidance
                    shaped_reward += 1.0

                    # Log successful projectile avoidance occasionally
                    if self.n_steps % 100 == 0:
                        avoidance_rate = (
                            self.successful_projectile_avoidance
                            / max(1, self.projectile_defensive_actions)
                        ) * 100
                        print(
                            f"Projectile avoidance success rate: {avoidance_rate:.1f}% "
                            f"({self.successful_projectile_avoidance}/{self.projectile_defensive_actions})"
                        )
                else:
                    # Small penalty for incorrect timing of defensive action
                    shaped_reward -= 0.2
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
            print(f"Error in reward shaping: {str(e)}")
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info


def make_enhanced_kungfu_env(
    is_play_mode=False, frame_stack=8, use_projectile_features=False
):
    """Create a Kung Fu Master environment with enhanced projectile detection

    Args:
        is_play_mode: Whether to render the environment
        frame_stack: Number of frames to stack (8 recommended for projectile detection)
        use_projectile_features: Whether to use explicit projectile features
    """
    try:
        # Stable Retro uses a different API
        render_mode = "human" if is_play_mode else None
        env = retro.make(game="KungFu-Nes", render_mode=render_mode)
    except Exception as e:
        print(f"First attempt failed: {str(e)}")
        try:
            render_mode = "human" if is_play_mode else None
            env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)
        except Exception as e:
            print(f"Second attempt failed: {str(e)}")
            # Last resort - try without render_mode parameter
            try:
                env = retro.make(game="KungFu-Nes")
                if is_play_mode:
                    env.render_mode = "human"
            except Exception as e:
                print(f"Third attempt failed: {str(e)}")
                try:
                    env = retro.make(game="KungFuMaster-Nes")
                    if is_play_mode:
                        env.render_mode = "human"
                except Exception as e:
                    raise Exception(f"Could not create Kung Fu environment: {str(e)}")

    # Apply our enhanced wrapper
    env = EnhancedKungFuMasterEnv(env)

    # Set up monitoring with our resilient monitor
    os.makedirs("logs", exist_ok=True)
    monitor_path = os.path.join("logs", "kungfu")
    env = ResilientMonitor(env, monitor_path)

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Apply frame stacking - we need more frames for better projectile detection
    print(
        f"Using enhanced frame stacking with n_stack={frame_stack} for improved projectile detection"
    )
    env = VecFrameStack(env, n_stack=frame_stack)

    # Optionally add projectile features wrapper
    if use_projectile_features:
        print("Adding projectile feature wrapper for explicit projectile information")
        # Import our projectile aware wrapper
        from projectile_aware_wrapper import wrap_projectile_aware

        env = wrap_projectile_aware(env, max_projectiles=5)

    return env


# Set model path (same as original)
MODEL_PATH = "model/kungfu.zip"
