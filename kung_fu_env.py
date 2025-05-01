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

logger = logging.getLogger("kungfu_env")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for detailed tracing
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
    "boss_hp": 0x04A5,
    "boss_x": 0x0093,
    "boss_action": 0x004E,
    "player_action": 0x0069,
    "player_action_timer": 0x0021,
    "player_air_mode": 0x036A,
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],
    "frame_counter": 0x0049,
    "screen_scroll1": 0x00E5,
    "screen_scroll2": 0x00D4,
    "game_submode": 0x0008,
    "enemy1_x": 0x008E,
    "enemy2_x": 0x008F,
    "enemy3_x": 0x0090,
    "enemy4_x": 0x0091,
    "enemy1_action": 0x0080,
    "enemy2_action": 0x0081,
    "enemy3_action": 0x0082,
    "enemy4_action": 0x0083,
    "enemy1_timer": 0x002B,
    "enemy2_timer": 0x002C,
    "enemy3_timer": 0x002D,
    "enemy4_timer": 0x002E,
    "grab_counter": 0x0374,
    "shrug_counter": 0x0378,
}

# Maximum episode duration
MAX_EPISODE_STEPS = 3600  # 2 minutes

# Set default model path
MODEL_PATH = "model/kungfu_dfp.zip"

# Global config for environment behavior
ENV_CONFIG = {
    "progression_weight": 1.5,
    "combat_engagement_weight": 1.2,
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
        # Option 1: If using index-based actions with the PPO agent
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))

        # Option 2: Alternative if directly using button combinations
        # self.action_space = gym.spaces.MultiBinary(9)  # For 9-button actions

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

        # Enemy tracking
        self.prev_enemy_positions = {}
        self.time_since_last_attack = 0
        self.enemies_in_range_count = 0
        self.last_engagement_reward = 0

        # Track for DFP measurements - initialize all as proper types
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
            return int(score)
        except Exception as e:
            logger.error(f"Error getting score: {e}")
            return 0

    def get_boss_hp(self):
        return self.get_ram_value(MEMORY["boss_hp"])

    def get_enemy_positions(self):
        enemies = []
        try:
            player_x = self.get_ram_value(MEMORY["player_x"])
            enemy_y_positions = [80, 120, 160, 200]
            for i in range(1, 5):
                x = self.get_ram_value(MEMORY[f"enemy{i}_x"])
                action = self.get_ram_value(MEMORY[f"enemy{i}_action"])
                timer = self.get_ram_value(MEMORY[f"enemy{i}_timer"])
                if x > 0:
                    y = enemy_y_positions[min(i - 1, len(enemy_y_positions) - 1)]
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
        try:
            boss_x = self.get_ram_value(MEMORY["boss_x"])
            boss_hp = self.get_ram_value(MEMORY["boss_hp"])
            boss_action = self.get_ram_value(MEMORY["boss_action"])
            if boss_hp > 0 and boss_x > 0:
                return {
                    "x": boss_x,
                    "hp": boss_hp,
                    "action": boss_action,
                    "y": 160,
                }
            return None
        except Exception as e:
            logger.error(f"Error getting boss info: {e}")
            return None

    def calculate_nearest_enemy_distance(self, enemies, player_x, player_y):
        if not enemies:
            return 999, None
        distances = []
        for enemy in enemies:
            dx = abs(enemy["x"] - player_x)
            dy = abs(enemy["y"] - player_y)
            distance = (dx**2 + dy**2) ** 0.5
            distances.append((distance, enemy))
        distances.sort()
        return distances[0] if distances else (999, None)

    def calculate_engagement_status(self, enemies, player_x, player_y):
        close_enemies = 0
        distance_enemies = 0
        for enemy in enemies:
            dx = abs(enemy["x"] - player_x)
            dy = abs(enemy["y"] - player_y)
            if dx < 30 and dy < 20:
                close_enemies += 1
            elif dx < 100:
                distance_enemies += 1
        return {
            "close_enemies": close_enemies,
            "distance_enemies": distance_enemies,
        }

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
        self.scores_by_step = []
        self.damage_by_step = []
        self.progress_by_step = []
        self.combat_engagement_by_step = []
        self.standing_still_count = 0
        self.no_progress_count = 0
        self.time_since_last_attack = 0

        # Check if enemy_distance_history is a deque, recreate if not
        if not isinstance(self.enemy_distance_history, deque):
            logger.warning("enemy_distance_history was not a deque. Recreating.")
            self.enemy_distance_history = deque(maxlen=10)
        else:
            self.enemy_distance_history.clear()

        self.prev_enemy_positions = {}

        self.prev_hp = self.get_hp()
        self.prev_x_pos, self.prev_y_pos = self.get_player_position()
        self.prev_stage = self.get_stage()
        self.prev_score = self.get_score()

        enemies = self.get_enemy_positions()
        for enemy in enemies:
            self.prev_enemy_positions[enemy["id"]] = (enemy["x"], enemy["y"])

        # Debug log for episode_steps
        logger.debug(
            f"After reset, self.episode_steps type={type(self.episode_steps)}, value={self.episode_steps}"
        )

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

        # Debug logging at step start
        logger.debug(
            f"Step begin, self.episode_steps type={type(self.episode_steps)}, value={self.episode_steps}"
        )

        # Ensure episode_steps is an integer
        if not isinstance(self.episode_steps, int):
            logger.error(
                f"self.episode_steps is not an integer at step start: type={type(self.episode_steps)}, value={self.episode_steps}. Resetting to 0."
            )
            self.episode_steps = 0

        # Ensure other tracking variables are proper types
        if not isinstance(self.total_steps, int):
            logger.warning(
                f"total_steps was not an integer: {type(self.total_steps)}. Fixing."
            )
            self.total_steps = 0

        if not isinstance(self.time_since_last_attack, int):
            logger.warning(
                f"time_since_last_attack was not an integer: {type(self.time_since_last_attack)}. Fixing."
            )
            self.time_since_last_attack = 0

        # Ensure step tracking variables are lists
        if not isinstance(self.scores_by_step, list):
            logger.warning(f"scores_by_step was not a list. Fixing.")
            self.scores_by_step = []

        if not isinstance(self.damage_by_step, list):
            logger.warning(f"damage_by_step was not a list. Fixing.")
            self.damage_by_step = []

        if not isinstance(self.progress_by_step, list):
            logger.warning(f"progress_by_step was not a list. Fixing.")
            self.progress_by_step = []

        if not isinstance(self.combat_engagement_by_step, list):
            logger.warning(f"combat_engagement_by_step was not a list. Fixing.")
            self.combat_engagement_by_step = []

        # Increment steps - ensure they're integers first
        self.total_steps = int(self.total_steps) + 1
        self.episode_steps = int(self.episode_steps) + 1
        self.time_since_last_attack = int(self.time_since_last_attack) + 1

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
                # Store the button combination for reference, but don't pass it to env.step
                button_combination = self.KUNGFU_ACTIONS[safe_action]
                # The key fix: use the integer action directly
                actual_action = safe_action
            else:
                logger.error(f"Received non-integer action: {action}, using No-op")
                button_combination = self.KUNGFU_ACTIONS[0]  # No-op
                actual_action = 0  # Use No-op action index
        except Exception as e:
            logger.error(f"Error converting action {action}: {e}")
            button_combination = self.KUNGFU_ACTIONS[0]  # Default to No-op
            actual_action = 0

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
            current_enemies = self.get_enemy_positions()
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

            self.enemy_distance_history.append(nearest_enemy_distance)

            try:
                score_diff = int(current_score - self.prev_score)
            except Exception as e:
                logger.error(f"Error calculating score_diff: {e}. Using 0.")
                score_diff = 0

            # Ensure scores_by_step is a list before appending
            if not isinstance(self.scores_by_step, list):
                self.scores_by_step = []
            self.scores_by_step.append(score_diff)

            damage_taken = self.prev_hp - current_hp
            if damage_taken < 0:
                damage_taken = 0

            # Ensure damage_by_step is a list before appending
            if not isinstance(self.damage_by_step, list):
                self.damage_by_step = []
            self.damage_by_step.append(damage_taken)

            if current_stage in [1, 3, 5]:
                progress = current_x_pos - self.prev_x_pos
                if current_stage > self.prev_stage:
                    progress += 100
            else:
                progress = self.prev_x_pos - current_x_pos
                if current_stage > self.prev_stage:
                    progress += 100

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

            # Ensure progress_by_step is a list before appending
            if not isinstance(self.progress_by_step, list):
                self.progress_by_step = []
            self.progress_by_step.append(recorded_progress)

            attack_attempt = action in [1, 6, 7, 9, 10, 11]
            if attack_attempt:
                self.time_since_last_attack = 0

            combat_engagement_reward = 0
            if attack_attempt and engagement["close_enemies"] > 0:
                combat_engagement_reward += (
                    0.5
                    * engagement["close_enemies"]
                    * ENV_CONFIG["combat_engagement_weight"]
                )

            if nearest_enemy and len(self.enemy_distance_history) > 3:
                avg_prev_distance = sum(list(self.enemy_distance_history)[:-1]) / (
                    len(self.enemy_distance_history) - 1
                )
                if nearest_enemy_distance < avg_prev_distance:
                    combat_engagement_reward += (
                        0.3 * ENV_CONFIG["combat_engagement_weight"]
                    )

            # Ensure combat_engagement_by_step is a list before appending
            if not isinstance(self.combat_engagement_by_step, list):
                self.combat_engagement_by_step = []
            self.combat_engagement_by_step.append(combat_engagement_reward)

            # Check that episode_steps is still an integer before the modulo operation
            if not isinstance(self.episode_steps, int):
                logger.error(
                    f"self.episode_steps became non-integer before modulo: {type(self.episode_steps)}. Fixing."
                )
                # Convert or reset to integer to avoid modulo error
                try:
                    self.episode_steps = int(self.episode_steps)
                except (ValueError, TypeError):
                    self.episode_steps = 0

            # Now safely perform the modulo operation
            if self.episode_steps % 100 == 0:
                try:
                    time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30
                    logger.info(
                        f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                        f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s, "
                        f"Enemies: {len(current_enemies)}"
                    )
                except Exception as e:
                    logger.error(f"Error in episode step logging: {e}")

            reward = score_diff * 0.3
            if current_stage > self.prev_stage:
                stage_bonus = 15.0 * ENV_CONFIG["progression_weight"]
                reward += stage_bonus
            if progress > 0:
                progress_reward = progress * 0.05 * ENV_CONFIG["progression_weight"]
                reward += progress_reward
            if damage_taken > 0:
                reward -= damage_taken * 0.05
            optimal_y = 160
            y_distance = abs(current_y_pos - optimal_y)
            if y_distance < 30:
                reward += 0.02
            if self.standing_still_count > 20:
                reward -= 0.05 * (self.standing_still_count - 20)
            if self.no_progress_count > 30:
                reward -= 0.1 * (self.no_progress_count - 30)
            reward += combat_engagement_reward
            reward += 0.01

            enemies_defeated = 0
            try:
                if isinstance(score_diff, (int, float)) and score_diff >= 100:
                    enemies_defeated = int(score_diff) // 100
            except Exception as e:
                logger.error(f"Error calculating enemies_defeated: {e}")
                enemies_defeated = 0

            # Ensure episode_steps is still an integer before calculating time_remaining
            if not isinstance(self.episode_steps, int):
                logger.error(
                    f"self.episode_steps became non-integer before time calculation: {type(self.episode_steps)}. Fixing."
                )
                try:
                    self.episode_steps = int(self.episode_steps)
                except (ValueError, TypeError):
                    self.episode_steps = 0

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
                        "enemies_close": int(engagement["close_enemies"]),
                        "enemies_distance": int(engagement["distance_enemies"]),
                        "combat_engagement_reward": float(combat_engagement_reward),
                        "nearest_enemy_distance": float(
                            nearest_enemy_distance
                            if nearest_enemy_distance != 999
                            else 0
                        ),
                    }
                )
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
            self.prev_enemy_positions = {
                enemy["id"]: (enemy["x"], enemy["y"]) for enemy in current_enemies
            }

        except Exception as e:
            logger.error(f"Error in measurement calculation: {e}")
            logger.error(traceback.format_exc())

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
    def __init__(self, env, measurement_history_length=5, fixed_goals=None):
        super().__init__(env)
        self.image_shape = env.observation_space.shape
        self.n_measurements = 3
        self.measurement_history_length = measurement_history_length
        self.measurement_history = deque(maxlen=measurement_history_length)
        for _ in range(measurement_history_length):
            self.measurement_history.append(np.zeros(self.n_measurements))
        if fixed_goals is not None:
            self.fixed_goals = np.array(fixed_goals, dtype=np.float32)
        else:
            self.fixed_goals = np.array([0.3, -0.2, 0.5], dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=self.image_shape, dtype=np.uint8
                ),
                "measurements": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_measurements * self.measurement_history_length,),
                    dtype=np.float32,
                ),
                "goals": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_measurements,),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = env.action_space
        self.prev_score = 0
        self.prev_damage = 0
        self.prev_progress = 0

    def reset(self, **kwargs):
        obs = np.zeros(self.image_shape, dtype=np.uint8)
        info = {}
        try:
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple) and len(result) == 2:
                obs, info = result
            else:
                obs = result
        except Exception as e:
            logger.error(f"Error in DFPWrapper reset: {e}")
            info["error"] = str(e)

        # Clear and reinitialize measurement history
        self.measurement_history = deque(maxlen=self.measurement_history_length)
        for i in range(self.measurement_history_length):
            self.measurement_history.append(np.zeros(self.n_measurements))

        self.prev_score = 0
        self.prev_damage = 0
        self.prev_progress = 0

        dict_obs = self._create_dict_observation(obs, info)
        return dict_obs, info

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        current_score = info.get("score_increase", 0)
        current_damage = info.get("damage_taken", 0)
        current_progress = info.get("progress_made", 0)
        measurements = np.array(
            [current_score, current_damage, current_progress], dtype=np.float32
        )
        self.measurement_history.append(measurements)
        self.prev_score = current_score
        self.prev_damage = current_damage
        self.prev_progress = current_progress
        dict_obs = self._create_dict_observation(obs, info)
        if len(result) == 5:
            return dict_obs, reward, terminated, truncated, info
        else:
            return dict_obs, reward, done, info

    def _create_dict_observation(self, image_obs, info=None):
        # Ensure image_obs is properly formatted
        if image_obs is None:
            logger.warning(
                "Received None for image_obs in _create_dict_observation. Creating blank image."
            )
            image_obs = np.zeros(self.image_shape, dtype=np.uint8)

        # Handle different image shapes
        if len(image_obs.shape) == 4 and image_obs.shape[-1] == 1:
            image_obs = np.array(image_obs, dtype=np.uint8)
        elif len(image_obs.shape) == 3 and image_obs.shape[-1] != 1:
            image_obs = np.expand_dims(image_obs, axis=-1)

        # Ensure measurement_history is properly initialized
        if (
            not isinstance(self.measurement_history, deque)
            or len(self.measurement_history) == 0
        ):
            logger.warning(
                "measurement_history is not properly initialized. Recreating it."
            )
            self.measurement_history = deque(maxlen=self.measurement_history_length)
            for _ in range(self.measurement_history_length):
                self.measurement_history.append(np.zeros(self.n_measurements))

        try:
            # Convert measurement history to a flat array
            flat_measurements = np.concatenate(list(self.measurement_history)).astype(
                np.float32
            )
        except Exception as e:
            logger.error(
                f"Error flattening measurements: {e}. Creating zero measurements."
            )
            flat_measurements = np.zeros(
                self.n_measurements * self.measurement_history_length, dtype=np.float32
            )

        # Ensure goals is properly initialized
        if not isinstance(self.fixed_goals, np.ndarray) or self.fixed_goals.shape != (
            self.n_measurements,
        ):
            logger.warning("fixed_goals is not properly initialized. Recreating it.")
            self.fixed_goals = np.array([0.3, -0.2, 0.5], dtype=np.float32)

        return {
            "image": image_obs,
            "measurements": flat_measurements,
            "goals": self.fixed_goals,
        }

    def set_goals(self, new_goals):
        self.fixed_goals = np.array(new_goals, dtype=np.float32)


def make_kungfu_env(is_play_mode=False, frame_stack=4, use_dfp=True):
    try:
        from stable_baselines3.common.atari_wrappers import (
            ClipRewardEnv,
            WarpFrame,
        )
        from gymnasium.wrappers import FrameStack
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.preprocessing import check_for_nested_spaces

        logger.info("Creating Kung Fu Master Retro environment")

        # Keep using DISCRETE mode but ensure we handle out-of-bounds actions in our wrapper
        env = retro.make(
            game="KungFu-Nes",
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode="human" if is_play_mode else None,
        )

        RetroEnvManager.get_instance().register_env(env)
        env = KungFuMasterEnv(env)  # Our wrapper will handle action mapping safely

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

        class CustomFrameStack(FrameStack):
            def __init__(self, env, num_stack):
                super().__init__(env, num_stack)

            def observation(self, observation):
                obs = super().observation(observation)
                return np.array(obs)

        env = CustomFrameStack(env, frame_stack)
        env = Monitor(env)
        logger.info(f"Observation space after wrapping: {env.observation_space}")
        logger.info(f"Observation space shape: {env.observation_space.shape}")
        if use_dfp:
            logger.info("Adding DFP wrapper to environment")
            goals = [
                0.3,
                -0.2,
                ENV_CONFIG["progression_weight"] / 3.0,
            ]
            env = DFPWrapper(env, measurement_history_length=5, fixed_goals=goals)
        logger.info("Environment setup complete")
        return env
    except Exception as e:
        logger.error(f"Error creating Kung Fu environment: {e}")
        logger.error(traceback.format_exc())
        raise
