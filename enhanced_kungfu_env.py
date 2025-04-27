import numpy as np
import gymnasium as gym
import retro
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
import os

# Import the ProjectileDetector
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

# Critical memory addresses
MEMORY = {
    "current_stage": 0x0058,  # Current Stage
    "player_hp": 0x04A6,  # Hero HP
    "player_x": 0x0094,  # Hero Screen Pos X
    "player_y": 0x0097,  # Hero Screen Pos Y
    "game_mode": 0x0051,  # Game Mode
    "boss_hp": 0x04A5,  # Boss HP
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
    "enemy_projectile": 0x0420,  # Enemy projectile state (active/inactive)
    "projectile_x": 0x0422,  # Projectile X position
    "projectile_y": 0x0424,  # Projectile Y position
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes


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

        # Create projectile detector
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

    def detect_active_projectiles_from_ram(self):
        """Detect active projectiles directly from RAM"""
        try:
            projectile_state = self.get_ram_value(MEMORY["enemy_projectile"])
            if projectile_state > 0:  # Projectile is active
                x = self.get_ram_value(MEMORY["projectile_x"])
                y = self.get_ram_value(MEMORY["projectile_y"])
                return [{"x": x, "y": y, "active": True}]
            return []
        except:
            return []

    def reset(self, **kwargs):
        # Reset the environment
        obs_result = self.env.reset(**kwargs)

        # Handle different return types
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs, info = obs_result
        else:
            obs = obs_result
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
        for _ in range(5):
            self.env.step(KUNGFU_ACTIONS[3])  # Press START

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

        # Buffer the initial observation
        self._buffer_raw_observation(obs)

        return obs, info

    def step(self, action):
        # Convert to actual action
        converted_action = self.KUNGFU_ACTIONS[action]

        # Get current state before taking action
        current_hp = self.get_hp()
        player_position = self.get_player_position()

        # Check for projectiles before action (using RAM and/or image processing)
        ram_projectiles = self.detect_active_projectiles_from_ram()

        # Image-based projectile detection if we have enough frames buffered
        image_projectiles = []
        projectile_info = None
        recommended_action = 0

        if len(self.raw_observation_buffer) >= 2:
            # Convert buffer to numpy array for projectile detection
            frame_stack = np.array(self.raw_observation_buffer)
            # Detect projectiles from frame differences
            projectile_info = enhance_observation_with_projectiles(
                frame_stack, self.projectile_detector, player_position
            )
            image_projectiles = projectile_info["projectiles"]
            recommended_action = projectile_info["recommended_action"]

        # Determine if there's an active projectile threat
        projectile_threat = len(ram_projectiles) > 0 or len(image_projectiles) > 0

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
        step_result = self.env.step(converted_action)

        # Handle different return types
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")

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


def make_enhanced_kungfu_env(is_play_mode=False, frame_stack=8):
    """Create a Kung Fu Master environment with enhanced projectile detection

    Args:
        is_play_mode: Whether to render the environment
        frame_stack: Number of frames to stack (8 recommended for projectile detection)
    """
    try:
        render_mode = "human" if is_play_mode else None
        env = retro.make(game="KungFu-Nes", render_mode=render_mode)
    except Exception:
        try:
            render_mode = "human" if is_play_mode else None
            env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)
        except Exception as e:
            raise e

    # Apply our enhanced wrapper
    env = EnhancedKungFuMasterEnv(env)

    # Set up monitoring
    os.makedirs("logs", exist_ok=True)
    env = Monitor(env, os.path.join("logs", "kungfu"))

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Apply frame stacking - we need more frames for better projectile detection
    print(
        f"Using enhanced frame stacking with n_stack={frame_stack} for improved projectile detection"
    )
    env = VecFrameStack(env, n_stack=frame_stack)

    return env


# Set model path (same as original)
MODEL_PATH = "model/kungfu.zip"
