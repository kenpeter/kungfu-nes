import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import FrameStack
import logging
import gc
import atexit
import traceback
from collections import deque
import retro
import cv2
import time
from threat_detection import (
    ThreatDetection,
    AgentActionType,
    ThreatType,
    ThreatDirection,
)

logger = logging.getLogger("kungfu_env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Changed to INFO for less verbose logs
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="kungfu_env.log",
)
logger = logging.getLogger("kungfu_env")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

# Define the Kung Fu Master action space - simplified
KUNGFU_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No-op
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Punch)
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # UP (Jump)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN (Crouch)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (Kick)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # B + A (Punch + Kick)
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # UP + RIGHT (Jump + Right)
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # DOWN + A (Crouch Kick)
    [1, 0, 0, 0, 0, 0, 0, 1, 0],  # B + RIGHT (Punch + Right)
    [0, 0, 0, 0, 0, 0, 1, 0, 1],  # LEFT + A (Left + Kick)
]

# Action names
KUNGFU_ACTION_NAMES = [
    "No-op",
    "Punch",
    "Jump",
    "Crouch",
    "Left",
    "Right",
    "Kick",
    "Punch + Kick",
    "Jump + Right",
    "Crouch Kick",
    "Punch + Right",
    "Left + Kick",
]

# Critical memory addresses for NES Kung Fu Master
MEMORY = {
    "current_stage": 0x0058,
    "player_hp": 0x04A6,
    "player_x": 0x0094,
    "player_y": 0x00B6,
    "game_mode": 0x0051,
    "player_action": 0x0069,
    "player_action_timer": 0x0021,
    "player_air_mode": 0x036A,
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],
    "frame_counter": 0x0049,
    "screen_scroll1": 0x00E5,
    "screen_scroll2": 0x00D4,
    "game_submode": 0x0008,
    "grab_counter": 0x0374,
    "shrug_counter": 0x0378,
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes

# Set default model path
MODEL_PATH = "model/kungfu_model.zip"

# Global config for environment behavior with simplified weights
ENV_CONFIG = {
    "progression_weight": 2.0,  # Increased to emphasize progression
    "combat_engagement_weight": 1.5,  # Adjusted for balance
    "enemy_detection_range": 120,
    "proactive_combat_bonus": 0.5,  # Reduced to avoid over-aggression
    "defensive_bonus": 0.1,  # Significantly reduced to prevent over-defensive play
    "strategic_retreat_bonus": 0.01,  # Minimal retreat bonus
}


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
        gc.collect()


# Simplified environment wrapper
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.KUNGFU_ACTIONS = KUNGFU_ACTIONS
        self.KUNGFU_ACTION_NAMES = KUNGFU_ACTION_NAMES

        # Create an action space that matches the KUNGFU_ACTIONS format
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))

        # Initialize threat detection system
        self.threat_detector = ThreatDetection(frame_size=(84, 84))

        # Track last observation for threat detection
        self.last_observation = None

        # State tracking - initialize all with proper types
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_stage = 0
        self.episode_steps = 0
        self.standing_still_count = 0
        self.no_progress_count = 0
        self.enemy_distance_history = deque(maxlen=10)

        # Track time since last damage taken for defensive strategy evaluation
        self.frames_since_damage = 0

        # Track stage time to adjust rewards based on progression
        self.current_stage_start_time = 0
        self.stage_time_limit = 1200  # 40 seconds per stage

        # Tracking for threat-based decisions
        self.last_recommended_action = None
        self.recommended_action_taken = False

        # Flag to track whether reset has been called
        self.reset_called = False

        logger.info("KungFuMasterEnv initialized with simplified action space")
        logger.info(f"Environment configuration: {ENV_CONFIG}")

    def get_ram(self):
        try:
            if hasattr(self.env, "get_ram"):
                return self.env.get_ram()
            elif hasattr(self.env.unwrapped, "get_ram"):
                return self.env.unwrapped.get_ram()
            elif hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
                return self.env.data.memory
            elif hasattr(self.env.unwrapped, "data") and hasattr(
                self.env.unwrapped.data, "memory"
            ):
                return self.env.unwrapped.data.memory
            elif hasattr(self.env, "_memory") and self.env._memory is not None:
                return self.env._memory
            logger.warning("Unable to access RAM through known methods")
            return np.zeros(0x10000, dtype=np.uint8)
        except Exception as e:
            logger.error(f"Error accessing RAM: {e}")
            return np.zeros(0x10000, dtype=np.uint8)

    def get_ram_value(self, address):
        try:
            ram = self.get_ram()
            if ram is not None and len(ram) > address:
                return int(ram[address])
            return 0
        except Exception as e:
            logger.error(f"Error accessing RAM at address 0x{address:04x}: {e}")
            return 0

    def get_stage(self):
        return self.get_ram_value(MEMORY["current_stage"])

    def get_hp(self):
        return self.get_ram_value(MEMORY["player_hp"])

    def get_player_position(self):
        try:
            x = self.get_ram_value(MEMORY["player_x"])
            y = self.get_ram_value(MEMORY["player_y"])
            return (x, y)
        except Exception as e:
            logger.error(f"Error getting player position: {e}")
            return (0, 0)

    def get_score(self):
        try:
            score = 0
            for i, addr in enumerate(MEMORY["score"]):
                digit = self.get_ram_value(addr)
                score += digit * (10 ** (4 - i))
            return int(score)
        except Exception as e:
            logger.error(f"Error getting score: {e}")
            return 0

    def reset(self, **kwargs):
        self.reset_called = True
        obs = np.zeros((84, 84, 1), dtype=np.uint8)
        info = {}
        try:
            obs_result = self.env.reset(**kwargs)
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                obs, info = obs_result
            else:
                obs = obs_result
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            info["error"] = str(e)

        # Explicitly ensure all tracking variables are correctly typed
        self.episode_steps = 0
        self.standing_still_count = 0
        self.no_progress_count = 0
        self.last_recommended_action = None
        self.recommended_action_taken = False
        self.frames_since_damage = 0
        self.current_stage_start_time = 0

        # Check if enemy_distance_history is a deque, recreate if not
        if not isinstance(self.enemy_distance_history, deque):
            logger.warning("enemy_distance_history was not a deque. Recreating.")
            self.enemy_distance_history = deque(maxlen=10)
        else:
            self.enemy_distance_history.clear()

        self.prev_hp = self.get_hp()
        self.prev_x_pos, self.prev_y_pos = self.get_player_position()
        self.prev_stage = self.get_stage()
        self.prev_score = self.get_score()

        # Reset threat detector
        self.last_observation = obs

        logger.info(
            f"Reset - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
            f"Pos: ({self.prev_x_pos}, {self.prev_y_pos}), Score: {self.prev_score}"
        )

        return obs, info

    def step(self, action):
        if not self.reset_called:
            logger.warning(
                "step() called before reset(). Attempting to reset the environment."
            )
            self.reset()

        # Ensure episode_steps is an integer
        if not isinstance(self.episode_steps, int):
            logger.error(
                f"self.episode_steps is not an integer at step start: type={type(self.episode_steps)}, value={self.episode_steps}. Resetting to 0."
            )
            self.episode_steps = 0

        # Increment steps
        self.episode_steps = int(self.episode_steps) + 1

        # Increment frames since damage
        self.frames_since_damage += 1

        # Get the actual button combination for logging/info purpose only
        button_combination = None
        try:
            # Handle action index safely using modulo operation to prevent out-of-bounds errors
            if isinstance(action, (int, np.int32, np.int64)):
                # Use modulo to ensure action is within bounds
                safe_action = action % len(self.KUNGFU_ACTIONS)
                if safe_action != action:
                    logger.warning(
                        f"Action {action} out of bounds, using {safe_action} instead"
                    )
                button_combination = self.KUNGFU_ACTIONS[safe_action]
                actual_action = safe_action
            else:
                logger.error(f"Received non-integer action: {action}, using No-op")
                button_combination = self.KUNGFU_ACTIONS[0]  # No-op
                actual_action = 0  # Use No-op action index
        except Exception as e:
            logger.error(f"Error converting action {action}: {e}")
            button_combination = self.KUNGFU_ACTIONS[0]  # Default to No-op
            actual_action = 0

        # Check if this action matches the last recommended action
        try:
            if self.last_recommended_action is not None:
                action_index = self.last_recommended_action.value
                self.recommended_action_taken = action_index == actual_action
        except Exception as e:
            logger.error(f"Error checking recommended action: {e}")
            self.recommended_action_taken = False

        try:
            # Use the integer action index with the retro environment
            step_result = self.env.step(actual_action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            elif len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            else:
                logger.warning(
                    f"Unexpected step result format with {len(step_result)} values"
                )
                obs = np.zeros((84, 84, 1), dtype=np.uint8)
                reward = 0
                terminated = True
                truncated = True
                info = {}
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            logger.error(traceback.format_exc())
            obs = np.zeros((84, 84, 1), dtype=np.uint8)
            reward = -1.0
            terminated = True
            truncated = True
            info = {"error": str(e)}

        if self.episode_steps >= MAX_EPISODE_STEPS:
            truncated = True

        try:
            current_hp = self.get_hp()
            current_x_pos, current_y_pos = self.get_player_position()
            current_stage = self.get_stage()
            current_score = self.get_score()

            # Process observation with threat detector
            player_pos = (current_x_pos, current_y_pos)
            current_time = time.time()

            # Process frame for threats - simplified
            highest_threat, all_threats = self.threat_detector.process_frame(
                obs, player_pos, current_time
            )

            # Store threat information
            if highest_threat:
                self.last_recommended_action = highest_threat.recommended_action
                threat_info = {
                    "threat_detected": True,
                    "threat_type": highest_threat.threat_type.name,
                    "threat_direction": highest_threat.direction.name,
                    "threat_priority": highest_threat.priority,
                    "recommended_action": highest_threat.recommended_action.name,
                    "recommended_action_index": highest_threat.recommended_action.value,
                    "threat_distance": highest_threat.distance_to_player,
                    "num_threats": len(all_threats),
                    "recommended_action_taken": self.recommended_action_taken,
                }
            else:
                self.last_recommended_action = None
                threat_info = {
                    "threat_detected": False,
                    "num_threats": 0,
                    "recommended_action_taken": False,
                }

            # Save observation for next frame
            self.last_observation = obs

            try:
                score_diff = int(current_score - self.prev_score)
            except Exception as e:
                logger.error(f"Error calculating score_diff: {e}. Using 0.")
                score_diff = 0

            damage_taken = self.prev_hp - current_hp
            if damage_taken < 0:
                damage_taken = 0

            # If damage was taken, reset frames_since_damage counter
            if damage_taken > 0:
                self.frames_since_damage = 0

            # Check for stage change
            stage_changed = current_stage > self.prev_stage
            if stage_changed:
                self.current_stage_start_time = self.episode_steps

            # SIMPLIFIED: Stage progression logic
            # Odd stages (1,3,5) - progress is measured by moving RIGHT (x increases)
            # Even stages (2,4,6) - progress is measured by moving LEFT (x decreases)
            if current_stage in [2, 4, 6]:  # Even stages - progress left
                progress = self.prev_x_pos - current_x_pos
                # Don't penalize for being at left boundary
                if current_x_pos < 20:
                    progress = max(0, progress)
                if current_stage > self.prev_stage:
                    progress += 100
            else:  # Odd stages (1,3,5) - progress right
                progress = current_x_pos - self.prev_x_pos
                # Don't penalize for being at right boundary
                if current_x_pos > 230:
                    progress = max(0, progress)
                if current_stage > self.prev_stage:
                    progress += 100

            # Track standing still and no progress
            if abs(progress) < 2:
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0

            if (
                abs(current_x_pos - self.prev_x_pos) < 2
                and abs(current_y_pos - self.prev_y_pos) < 2
            ):
                self.standing_still_count += 1
            else:
                self.standing_still_count = 0

            recorded_progress = max(0, progress)

            attack_attempt = action in [1, 6, 7, 9, 10, 11]
            defensive_action = action in [2, 3, 4]  # Jump, crouch, move left

            # SIMPLIFIED: Combat engagement reward - focused on attacking
            combat_engagement_reward = 0

            # Add threat-based combat rewards - simplified
            if highest_threat and highest_threat.threat_type == ThreatType.REGULAR:
                # Reward for detecting threats
                combat_engagement_reward += 0.2 * ENV_CONFIG["combat_engagement_weight"]

                # Reward attacking threats that are close
                if highest_threat.distance_to_player < 40 and attack_attempt:
                    combat_engagement_reward += (
                        0.5 * ENV_CONFIG["combat_engagement_weight"]
                    )

                # Bonus for recommended action
                if self.recommended_action_taken:
                    combat_engagement_reward += (
                        0.3 * ENV_CONFIG["combat_engagement_weight"]
                    )

            # SIMPLIFIED: Defensive reward - minimal to prevent over-defensive play
            defensive_reward = 0
            if (
                self.frames_since_damage > 60 and attack_attempt
            ):  # Only reward avoiding damage if also attacking
                defensive_reward += 0.1 * ENV_CONFIG["defensive_bonus"]

            # Log status every 100 steps
            if self.episode_steps % 100 == 0:
                try:
                    time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30
                    logger.info(
                        f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                        f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s"
                    )
                except Exception as e:
                    logger.error(f"Error in episode step logging: {e}")

            # SIMPLIFIED: Reward calculation - emphasize progression and combat
            reward = score_diff * 0.3

            # Stage progression bonus - big reward
            if stage_changed:
                stage_bonus = 25.0 * ENV_CONFIG["progression_weight"]  # Increased bonus
                reward += stage_bonus

            # Progress reward - major component
            if progress > 0:
                progress_reward = progress * 0.1 * ENV_CONFIG["progression_weight"]
                reward += progress_reward

            # Damage penalty - moderate
            if damage_taken > 0:
                reward -= damage_taken * 0.1

            # Vertical positioning reward - minor
            optimal_y = 160
            y_distance = abs(current_y_pos - optimal_y)
            if y_distance < 30:
                reward += 0.05

            # Penalties for standing still or making no progress
            if self.standing_still_count > 20:  # Increased threshold
                reward -= 0.05 * (self.standing_still_count - 20)  # Reduced penalty
            if self.no_progress_count > 30:  # Increased threshold
                reward -= 0.1 * (self.no_progress_count - 30)  # Reduced penalty

            # Add combat engagement reward
            reward += combat_engagement_reward

            # Add defensive reward - minimal
            reward += defensive_reward

            # Small survival reward
            reward += 0.01

            enemies_defeated = 0
            try:
                if isinstance(score_diff, (int, float)) and score_diff >= 100:
                    enemies_defeated = int(score_diff) // 100
            except Exception as e:
                logger.error(f"Error calculating enemies_defeated: {e}")
                enemies_defeated = 0

            try:
                time_remaining = (MAX_EPISODE_STEPS - self.episode_steps) / 30
                info.update(
                    {
                        "current_stage": int(current_stage),
                        "current_score": int(current_score),
                        "current_hp": int(current_hp),
                        "player_x": int(current_x_pos),
                        "player_y": int(current_y_pos),
                        "score_increase": int(score_diff),
                        "damage_taken": int(damage_taken),
                        "progress_made": float(recorded_progress),
                        "time_remaining": float(time_remaining),
                        "enemies_defeated": int(enemies_defeated),
                        "attack_attempt": int(attack_attempt),
                        "strategic_position": int(abs(current_y_pos - 160) < 30),
                        "combat_engagement_reward": float(combat_engagement_reward),
                        "defensive_reward": float(defensive_reward),
                        "frames_without_damage": int(self.frames_since_damage),
                    }
                )

                # Add threat information to info dictionary
                info.update(threat_info)

            except Exception as e:
                logger.error(f"Error updating info dictionary: {e}")
                info.update(
                    {
                        "current_stage": 0,
                        "current_score": 0,
                        "score_increase": 0,
                        "error": str(e),
                    }
                )

            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_y_pos = current_y_pos
            self.prev_stage = current_stage
            self.prev_score = current_score

        except Exception as e:
            logger.error(f"Error in measurement calculation: {e}")
            logger.error(traceback.format_exc())

        return obs, reward, terminated, truncated, info

    def render_with_threats(self):
        """Render the game with threat visualization overlay"""
        # This assumes the environment has a render method
        frame = None
        try:
            frame = self.env.render(mode="rgb_array")
        except Exception as e:
            logger.error(f"Error in render: {e}")
            # Create a blank frame if render fails
            frame = np.zeros((210, 160, 3), dtype=np.uint8)
        return frame


def make_kungfu_env(is_play_mode=False, frame_stack=4, use_dfp=False):
    try:
        from stable_baselines3.common.atari_wrappers import (
            ClipRewardEnv,
            WarpFrame,
        )
        from gymnasium.wrappers import FrameStack
        from stable_baselines3.common.monitor import Monitor

        logger.info("Creating Kung Fu Master Retro environment")

        env = retro.make(
            game="KungFu-Nes",
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode="human" if is_play_mode else None,
        )

        RetroEnvManager.get_instance().register_env(env)
        env = KungFuMasterEnv(env)

        # Use a wrapper that converts to channel-first format (what SB3 expects)
        class ChannelFirstWrapper(gym.ObservationWrapper):
            def __init__(self, env):
                super().__init__(env)
                # Update observation space to be channel-first
                old_shape = self.observation_space.shape
                # If last dimension is 1 (grayscale images), convert to (1, H, W)
                if len(old_shape) == 3 and old_shape[-1] == 1:
                    new_shape = (1, old_shape[0], old_shape[1])
                    self.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=new_shape, dtype=np.uint8
                    )
                elif len(old_shape) == 4 and old_shape[-1] == 1:
                    # Handle case when already stacked (N, H, W, 1)
                    new_shape = (old_shape[0], old_shape[1], old_shape[2])
                    self.observation_space = gym.spaces.Box(
                        low=0, high=255, shape=new_shape, dtype=np.uint8
                    )

            def observation(self, observation):
                # Check the shape to determine proper handling
                if len(observation.shape) == 3 and observation.shape[-1] == 1:
                    # For (H, W, 1) -> (1, H, W)
                    return np.transpose(observation, (2, 0, 1))
                elif len(observation.shape) == 4 and observation.shape[-1] == 1:
                    # For (N, H, W, 1) -> (N, H, W)
                    return observation.squeeze(-1)
                elif len(observation.shape) == 3 and observation.shape[0] == 1:
                    # Already in (1, H, W) format
                    return observation
                elif len(observation.shape) == 2:
                    # For (H, W) -> (1, H, W)
                    return observation[np.newaxis, ...]
                # Default fallback
                return observation

        class CustomWarpFrame(WarpFrame):
            def __init__(self, env, width=84, height=84):
                super().__init__(env, width=width, height=height)

            def observation(self, obs):
                if len(obs.shape) == 3 and obs.shape[-1] == 1:
                    resized = cv2.resize(
                        obs.squeeze(-1),
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    )
                    return resized[:, :, np.newaxis]
                else:
                    return super().observation(obs)

        env = CustomWarpFrame(env)
        env = ClipRewardEnv(env)

        # Apply frame stacking - using standard consecutive frames
        class CustomFrameStack(FrameStack):
            def __init__(self, env, num_stack):
                super().__init__(env, num_stack)

            def observation(self, observation):
                obs = super().observation(observation)
                return np.array(obs)

        if frame_stack > 1:
            env = CustomFrameStack(env, frame_stack)

        # Apply channel-first conversion after frame stacking
        env = ChannelFirstWrapper(env)

        env = Monitor(env)
        logger.info(f"Observation space after wrapping: {env.observation_space}")
        logger.info(f"Observation space shape: {env.observation_space.shape}")
        logger.info("Environment setup complete")
        return env
    except Exception as e:
        logger.error(f"Error creating Kung Fu environment: {e}")
        logger.error(traceback.format_exc())
        raise
