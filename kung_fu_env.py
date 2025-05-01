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
from collections import deque
import retro


logger = logging.getLogger("kungfu_env")

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
    "Punch + Right",  # Added for aggressive forward progress
    "Left + Kick",  # Added for aggressive back attack
]

# Critical memory addresses for NES Kung Fu Master
MEMORY = {
    "current_stage": 0x0058,  # Current Stage
    "player_hp": 0x04A6,  # Hero HP
    "player_x": 0x0094,  # Hero Screen Pos X
    "player_y": 0x00B6,  # Hero Pos Y
    "game_mode": 0x0051,  # Game Mode
    "boss_hp": 0x04A5,  # Boss HP
    "boss_x": 0x0093,  # Boss Pos X
    "boss_action": 0x004E,  # Boss Action
    "player_action": 0x0069,  # Hero Action
    "player_action_timer": 0x0021,  # Hero Action Timer
    "player_air_mode": 0x036A,  # Hero Air Mode
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
    "frame_counter": 0x0049,  # Frame Counter
    "screen_scroll1": 0x00E5,  # Screen Scroll 1
    "screen_scroll2": 0x00D4,  # Screen Scroll 2
    "game_submode": 0x0008,  # Game Submode
    "enemy1_x": 0x008E,  # Enemy 1 Pos X
    "enemy2_x": 0x008F,  # Enemy 2 Pos X
    "enemy3_x": 0x0090,  # Enemy 3 Pos X
    "enemy4_x": 0x0091,  # Enemy 4 Pos X
    "enemy1_action": 0x0080,  # Enemy 1 Action
    "enemy2_action": 0x0081,  # Enemy 2 Action
    "enemy3_action": 0x0082,  # Enemy 3 Action
    "enemy4_action": 0x0083,  # Enemy 4 Action
    "enemy1_timer": 0x002B,  # Enemy 1 Action Timer
    "enemy2_timer": 0x002C,  # Enemy 2 Action Timer
    "enemy3_timer": 0x002D,  # Enemy 3 Action Timer
    "enemy4_timer": 0x002E,  # Enemy 4 Action Timer
    "grab_counter": 0x0374,  # Grab Counter
    "shrug_counter": 0x0378,  # Shrug Counter
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes

# Set default model path
MODEL_PATH = "model/kungfu_dfp.zip"

# Global config for environment behavior (fixed with aggressive values by default)
ENV_CONFIG = {
    "progression_weight": 1.5,  # Weight for progression reward
    "combat_engagement_weight": 1.2,  # Weight for combat engagement reward
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
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))

        # State tracking
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_y_pos = 0
        self.prev_stage = 0
        self.episode_steps = 0
        self.standing_still_count = 0
        self.no_progress_count = 0
        self.enemy_distance_history = deque(maxlen=10)

        # Enemy tracking
        self.prev_enemy_positions = {}
        self.time_since_last_attack = 0
        self.enemies_in_range_count = 0
        self.last_engagement_reward = 0

        # Track for DFP measurements
        self.total_steps = 0
        self.scores_by_step = []
        self.damage_by_step = []
        self.progress_by_step = []
        self.combat_engagement_by_step = []

        # Flag to track whether reset has been called
        self.reset_called = False

        logger.info("KungFuMasterEnv initialized with enhanced action space")
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
            return score
        except Exception as e:
            logger.error(f"Error getting score: {e}")
            return 0

    def get_boss_hp(self):
        return self.get_ram_value(MEMORY["boss_hp"])

    def get_enemy_positions(self):
        """Get positions and actions of all enemies on screen"""
        enemies = []
        try:
            player_x = self.get_ram_value(MEMORY["player_x"])

            # Get fixed y positions based on game mechanics
            # In Kung Fu, enemies typically appear at certain fixed Y heights
            enemy_y_positions = [80, 120, 160, 200]

            for i in range(1, 5):  # Check 4 potential enemies
                x = self.get_ram_value(MEMORY[f"enemy{i}_x"])
                action = self.get_ram_value(MEMORY[f"enemy{i}_action"])
                timer = self.get_ram_value(MEMORY[f"enemy{i}_timer"])

                # Only consider active enemies (x > 0)
                if x > 0:
                    # Use an approximated y position based on index
                    # This is a simplification since the actual y positions aren't in memory map
                    y = enemy_y_positions[min(i - 1, len(enemy_y_positions) - 1)]

                    # Calculate relative position to player (negative = left of player)
                    relative_x = x - player_x

                    enemies.append(
                        {
                            "id": i,
                            "x": x,
                            "y": y,
                            "action": action,
                            "timer": timer,
                            "relative_x": relative_x,
                        }
                    )
            return enemies
        except Exception as e:
            logger.error(f"Error getting enemy positions: {e}")
            return []

    def get_boss_info(self):
        """Get boss information if present"""
        try:
            boss_x = self.get_ram_value(MEMORY["boss_x"])
            boss_hp = self.get_ram_value(MEMORY["boss_hp"])
            boss_action = self.get_ram_value(MEMORY["boss_action"])

            # If boss is active
            if boss_hp > 0 and boss_x > 0:
                return {
                    "x": boss_x,
                    "hp": boss_hp,
                    "action": boss_action,
                    # Approximate y position for the boss (mid-screen)
                    "y": 160,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting boss info: {e}")
            return None

    def calculate_nearest_enemy_distance(self, enemies, player_x, player_y):
        """Calculate distance to nearest enemy"""
        if not enemies:
            return 999, None  # No enemies found

        distances = []
        for enemy in enemies:
            dx = abs(enemy["x"] - player_x)
            dy = abs(enemy["y"] - player_y)
            distance = (dx**2 + dy**2) ** 0.5  # Euclidean distance
            distances.append((distance, enemy))

        distances.sort()  # Sort by distance
        return distances[0] if distances else (999, None)

    def calculate_engagement_status(self, enemies, player_x, player_y):
        """Calculate engagement metrics with enemies"""
        close_enemies = 0
        distance_enemies = 0

        for enemy in enemies:
            dx = abs(enemy["x"] - player_x)
            dy = abs(enemy["y"] - player_y)

            # Close combat range
            if dx < 30 and dy < 20:
                close_enemies += 1
            # Distance enemy
            elif dx < 100:
                distance_enemies += 1

        return {
            "close_enemies": close_enemies,
            "distance_enemies": distance_enemies,
        }

    def reset(self, **kwargs):
        self.reset_called = True
        try:
            obs_result = self.env.reset(**kwargs)
            if isinstance(obs_result, tuple) and len(obs_result) == 2:
                obs, info = obs_result
            else:
                obs = obs_result
                info = {}
        except Exception as e:
            logger.error(f"Error in reset: {e}")
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            info = {}

        self.episode_steps = 0
        self.scores_by_step = []
        self.damage_by_step = []
        self.progress_by_step = []
        self.combat_engagement_by_step = []
        self.standing_still_count = 0
        self.no_progress_count = 0
        self.time_since_last_attack = 0
        self.enemy_distance_history.clear()
        self.prev_enemy_positions = {}

        self.prev_hp = self.get_hp()
        self.prev_x_pos, self.prev_y_pos = self.get_player_position()
        self.prev_stage = self.get_stage()
        self.prev_score = self.get_score()

        # Initialize enemy tracking
        enemies = self.get_enemy_positions()
        for enemy in enemies:
            self.prev_enemy_positions[enemy["id"]] = (enemy["x"], enemy["y"])

        logger.info(
            f"Reset - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
            f"Pos: ({self.prev_x_pos}, {self.prev_y_pos}), Score: {self.prev_score}"
        )

        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            return obs, info
        else:
            return obs

    def step(self, action):
        if not self.reset_called:
            logger.warning(
                "step() called before reset(). Attempting to reset the environment."
            )
            self.reset()

        self.total_steps += 1
        self.episode_steps += 1
        self.time_since_last_attack += 1

        try:
            converted_action = self.KUNGFU_ACTIONS[action]
        except Exception as e:
            logger.error(f"Error converting action {action}: {e}")
            converted_action = self.KUNGFU_ACTIONS[0]

        try:
            step_result = self.env.step(converted_action)
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
                obs = np.zeros((224, 240, 3), dtype=np.uint8)
                reward = 0
                terminated = True
                truncated = True
                info = {}
        except Exception as e:
            logger.error(f"Error during environment step: {e}")
            obs = np.zeros((224, 240, 3), dtype=np.uint8)
            reward = -1.0
            terminated = True
            truncated = True
            info = {"error": str(e)}

        if self.episode_steps >= MAX_EPISODE_STEPS:
            truncated = True

        try:
            # Get basic game state
            current_hp = self.get_hp()
            current_x_pos, current_y_pos = self.get_player_position()
            current_stage = self.get_stage()
            current_score = self.get_score()

            # Track enemies
            current_enemies = self.get_enemy_positions()

            # Calculate engagement metrics
            engagement = self.calculate_engagement_status(
                current_enemies, current_x_pos, current_y_pos
            )
            nearest_enemy_data = self.calculate_nearest_enemy_distance(
                current_enemies, current_x_pos, current_y_pos
            )
            nearest_enemy_distance = (
                nearest_enemy_data[0] if isinstance(nearest_enemy_data, tuple) else 999
            )
            nearest_enemy = (
                nearest_enemy_data[1] if isinstance(nearest_enemy_data, tuple) else None
            )

            # Track if we're making progress
            self.enemy_distance_history.append(nearest_enemy_distance)

            # Calculate base metrics
            score_diff = current_score - self.prev_score
            self.scores_by_step.append(score_diff)

            damage_taken = self.prev_hp - current_hp
            if damage_taken < 0:  # Health restored
                damage_taken = 0
            self.damage_by_step.append(damage_taken)

            # Calculate progress differently based on stage (odd stages move right, even move left)
            if current_stage in [1, 3, 5]:
                progress = current_x_pos - self.prev_x_pos
                if current_stage > self.prev_stage:
                    progress += 100  # Big bonus for stage transition
            else:
                progress = self.prev_x_pos - current_x_pos  # Moving left in even stages
                if current_stage > self.prev_stage:
                    progress += 100  # Big bonus for stage transition

            # Detect if player is stuck without making progress
            if abs(progress) < 2:
                self.no_progress_count += 1
            else:
                self.no_progress_count = 0

            # Standing still too long gets negative reward (encourage movement)
            if (
                abs(current_x_pos - self.prev_x_pos) < 2
                and abs(current_y_pos - self.prev_y_pos) < 2
            ):
                self.standing_still_count += 1
            else:
                self.standing_still_count = 0

            # Ensure progress is non-negative for the progress tracker
            recorded_progress = max(0, progress)
            self.progress_by_step.append(recorded_progress)

            # Track combat engagement
            attack_attempt = action in [
                1,
                6,
                7,
                9,
                10,
                11,
            ]  # Action indices for attack moves
            if attack_attempt:
                self.time_since_last_attack = 0

            # Calculate combat engagement reward
            combat_engagement_reward = 0

            # Reward for attacking when enemies are close
            if attack_attempt and engagement["close_enemies"] > 0:
                combat_engagement_reward += (
                    0.5
                    * engagement["close_enemies"]
                    * ENV_CONFIG["combat_engagement_weight"]
                )

            # Reward for moving toward distant enemies (closing the distance)
            if nearest_enemy and len(self.enemy_distance_history) > 3:
                avg_prev_distance = sum(list(self.enemy_distance_history)[:-1]) / (
                    len(self.enemy_distance_history) - 1
                )
                if nearest_enemy_distance < avg_prev_distance:
                    combat_engagement_reward += (
                        0.3 * ENV_CONFIG["combat_engagement_weight"]
                    )

            self.combat_engagement_by_step.append(combat_engagement_reward)

            if self.episode_steps % 100 == 0:
                time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30
                logger.info(
                    f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                    f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s, "
                    f"Enemies: {len(current_enemies)}"
                )

            # Enhanced reward structure

            # Score-based rewards - kept from original
            reward = score_diff * 0.3

            # Stage progression rewards - increased for aggressive progression
            if current_stage > self.prev_stage:
                stage_bonus = 15.0 * ENV_CONFIG["progression_weight"]
                reward += stage_bonus

            # Forward progress rewards - enhanced for aggressive movement
            if progress > 0:
                progress_reward = progress * 0.05 * ENV_CONFIG["progression_weight"]
                reward += progress_reward

            # Damage penalties - adjusted based on configuration
            if damage_taken > 0:
                reward -= damage_taken * 0.05

            # Position-based rewards for strategic positioning
            optimal_y = 160
            y_distance = abs(current_y_pos - optimal_y)
            if y_distance < 30:
                reward += 0.02

            # Penalties for non-movement to avoid getting stuck
            if self.standing_still_count > 20:
                reward -= 0.05 * (self.standing_still_count - 20)

            if self.no_progress_count > 30:
                reward -= 0.1 * (self.no_progress_count - 30)

            # Add combat engagement reward
            reward += combat_engagement_reward

            # Reward for staying alive
            reward += 0.01

            # Track enemy defeats based on score increases
            enemies_defeated = 0
            if score_diff >= 100:
                enemies_defeated = score_diff // 100

            # Update info dictionary with enhanced metrics
            info.update(
                {
                    "current_stage": current_stage,
                    "current_score": current_score,
                    "current_hp": current_hp,
                    "player_x": current_x_pos,
                    "player_y": current_y_pos,
                    "score_increase": score_diff,
                    "damage_taken": damage_taken,
                    "progress_made": recorded_progress,
                    "time_remaining": (MAX_EPISODE_STEPS - self.episode_steps) / 30,
                    "enemies_defeated": enemies_defeated,
                    "attack_attempt": int(attack_attempt),
                    "strategic_position": int(abs(current_y_pos - 160) < 30),
                    "enemies_close": engagement["close_enemies"],
                    "enemies_distance": engagement["distance_enemies"],
                    "combat_engagement_reward": combat_engagement_reward,
                    "nearest_enemy_distance": (
                        nearest_enemy_distance if nearest_enemy_distance != 999 else 0
                    ),
                }
            )

            # Update previous state
            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_y_pos = current_y_pos
            self.prev_stage = current_stage
            self.prev_score = current_score

            # Update enemy tracking
            self.prev_enemy_positions = {
                enemy["id"]: (enemy["x"], enemy["y"]) for enemy in current_enemies
            }

        except Exception as e:
            logger.error(f"Error in measurement calculation: {e}")

        return obs, reward, terminated, truncated, info

    def close(self):
        try:
            if hasattr(self.env, "close"):
                self.env.close()
            logger.info("KungFuMasterEnv closed successfully")
        except Exception as e:
            logger.error(f"Error closing KungFuMasterEnv: {e}")
        RetroEnvManager.get_instance().unregister_env(self)


class DFPWrapper(gym.Wrapper):
    """
    Wrapper to convert standard observation space to the Dict format required for DFP.
    The original environment returns frame-stacked images, but DFP needs:
    - image: The frame-stacked game screen
    - measurements: Current metrics like score, health, etc.
    - goals: Target weights for the different measurement types
    """

    def __init__(self, env, measurement_history_length=5, fixed_goals=None):
        super().__init__(env)

        # Original observation space is a Box of stacked frames
        self.image_shape = env.observation_space.shape

        # Number of measurements to track (score, damage, progress, etc.)
        self.n_measurements = 3

        # Define the measurement history length
        self.measurement_history_length = measurement_history_length
        self.measurement_history = deque(maxlen=measurement_history_length)

        # Initialize with zeros
        for _ in range(measurement_history_length):
            self.measurement_history.append(np.zeros(self.n_measurements))

        # Set fixed goals if provided, otherwise use default progression-focused goals
        if fixed_goals is not None:
            self.fixed_goals = np.array(fixed_goals, dtype=np.float32)
        else:
            # Default goals: [score_weight, damage_weight, progress_weight]
            # Emphasize progression and score, minimize damage
            self.fixed_goals = np.array([0.3, -0.2, 0.5], dtype=np.float32)

        # Define new observation space as Dict
        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(
                    low=0, high=255, shape=self.image_shape, dtype=np.uint8
                ),
                "measurements": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_measurements * self.measurement_history_length,),
                    dtype=np.float32,
                ),
                "goals": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_measurements,),
                    dtype=np.float32,
                ),
            }
        )

        # Keep the same action space
        self.action_space = env.action_space

        # Metrics for history tracking
        self.prev_score = 0
        self.prev_damage = 0
        self.prev_progress = 0

    def reset(self, **kwargs):
        """Reset environment and measurement history."""
        result = self.env.reset(**kwargs)

        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs = result
            info = {}

        # Reset history
        for i in range(self.measurement_history_length):
            self.measurement_history.append(np.zeros(self.n_measurements))

        # Reset metrics
        self.prev_score = 0
        self.prev_damage = 0
        self.prev_progress = 0

        # Create Dict observation
        dict_obs = self._create_dict_observation(obs, info)

        if isinstance(result, tuple) and len(result) == 2:
            return dict_obs, info
        else:
            return dict_obs

    def step(self, action):
        """Take a step and convert the observation to Dict format."""
        result = self.env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        # Extract measurements from info
        current_score = info.get("score_increase", 0)
        current_damage = info.get("damage_taken", 0)
        current_progress = info.get("progress_made", 0)

        # Create measurement array for this step
        # [score_delta, damage_delta, progress_delta]
        measurements = np.array(
            [current_score, current_damage, current_progress], dtype=np.float32
        )

        # Add to history
        self.measurement_history.append(measurements)

        # Update previous metrics
        self.prev_score = current_score
        self.prev_damage = current_damage
        self.prev_progress = current_progress

        # Create Dict observation
        dict_obs = self._create_dict_observation(obs, info)

        if len(result) == 5:
            return dict_obs, reward, terminated, truncated, info
        else:
            return dict_obs, reward, done, info

    def _create_dict_observation(self, image_obs, info=None):
        """Convert standard observation to Dict format."""
        # Flatten measurement history into a single vector
        flat_measurements = np.concatenate(list(self.measurement_history)).astype(
            np.float32
        )

        return {
            "image": image_obs,
            "measurements": flat_measurements,
            "goals": self.fixed_goals,
        }

    def set_goals(self, new_goals):
        """Set new goal weights during training or evaluation."""
        self.fixed_goals = np.array(new_goals, dtype=np.float32)


def make_kungfu_env(is_play_mode=False, frame_stack=4, use_dfp=True):
    """Create a Kung Fu Master environment with appropriate wrappers"""
    try:
        # Import here to avoid circular imports
        from stable_baselines3.common.atari_wrappers import (
            MaxAndSkipEnv,
            ClipRewardEnv,
            WarpFrame,
            EpisodicLifeEnv,
        )

        # Import FrameStack from gymnasium instead
        from gymnasium.wrappers import FrameStack
        from stable_baselines3.common.monitor import Monitor

        # Create base environment
        logger.info("Creating Kung Fu Master Retro environment")
        env = retro.make(
            game="KungFu-Nes",
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode="human" if is_play_mode else None,
        )

        # Register with the manager
        RetroEnvManager.get_instance().register_env(env)

        # Apply standard wrappers
        logger.info("Applying environment wrappers")
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)

        # Apply our custom wrapper
        env = KungFuMasterEnv(env)

        # Add remaining standard wrappers
        env = WarpFrame(env)
        env = ClipRewardEnv(env)
        env = FrameStack(env, frame_stack)
        env = Monitor(env)

        # If using DFP, add the DFP wrapper
        if use_dfp:
            logger.info("Adding DFP wrapper to environment")
            # Define goals based on env config
            goals = [
                0.3,  # Score weight
                -0.2,  # Damage weight (negative because we want to minimize)
                ENV_CONFIG["progression_weight"] / 3.0,  # Progress weight (normalized)
            ]
            env = DFPWrapper(env, measurement_history_length=5, fixed_goals=goals)

        logger.info("Environment setup complete")
        return env
    except Exception as e:
        logger.error(f"Error creating Kung Fu environment: {e}")
        raise
