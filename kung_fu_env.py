import os
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium import spaces
import logging
import time
import gc
import atexit
from collections import deque
import retro

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
MODEL_PATH = "model/kungfu_dfp.zip"


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

        # Track for DFP measurements
        self.total_steps = 0
        self.scores_by_step = []
        self.damage_by_step = []
        self.progress_by_step = []

        # Flag to track whether reset has been called
        self.reset_called = False

        logger.info("KungFuMasterEnv initialized with simplified action space")

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
        self.prev_hp = self.get_hp()
        self.prev_x_pos, self.prev_y_pos = self.get_player_position()
        self.prev_stage = self.get_stage()
        self.prev_score = self.get_score()

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
            current_hp = self.get_hp()
            current_x_pos, current_y_pos = self.get_player_position()
            current_stage = self.get_stage()
            current_score = self.get_score()

            score_diff = current_score - self.prev_score
            self.scores_by_step.append(score_diff)

            damage_taken = self.prev_hp - current_hp
            if damage_taken < 0:
                damage_taken = 0
            self.damage_by_step.append(damage_taken)

            if current_stage in [1, 3]:
                progress = current_x_pos - self.prev_x_pos
                if current_stage > self.prev_stage:
                    progress += 100
            else:
                progress = self.prev_x_pos - current_x_pos
                if current_stage > self.prev_stage:
                    progress += 100

            if progress < 0:
                progress = 0
            self.progress_by_step.append(progress)

            if self.episode_steps % 100 == 0:
                time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30
                logger.info(
                    f"Stage: {current_stage}, Position: ({current_x_pos}, {current_y_pos}), "
                    f"HP: {current_hp}, Score: {current_score}, Time left: {time_left:.1f}s"
                )

            reward = score_diff * 0.1
            if current_stage > self.prev_stage:
                reward += 10.0
            if damage_taken > 0:
                reward -= damage_taken * 0.1

            info.update(
                {
                    "current_stage": current_stage,
                    "current_score": current_score,
                    "current_hp": current_hp,
                    "player_x": current_x_pos,
                    "player_y": current_y_pos,
                    "score_increase": score_diff,
                    "damage_taken": damage_taken,
                    "progress_made": progress,
                    "time_remaining": (MAX_EPISODE_STEPS - self.episode_steps) / 30,
                }
            )

            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_y_pos = current_y_pos
            self.prev_stage = current_stage
            self.prev_score = current_score

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


# Environment wrapper for Direct Future Prediction (DFP)
class DFPKungFuWrapper(gym.Wrapper):
    def __init__(
        # self
        self,
        # env
        env,
        # 4 stack
        frame_stack=4,
        # 3D: score, hp, progress
        measurement_dims=3,
        # predict 1 step, 3 step, etc
        prediction_horizons=[1, 3, 5, 10, 20],
    ):
        # basically, we need to extend the kung fu env
        super().__init__(env)
        self.frame_stack = frame_stack
        self.measurement_dims = measurement_dims
        self.prediction_horizons = prediction_horizons
        self.max_horizon = max(prediction_horizons)

        # Image buffer for frame stacking
        self.image_buffer = deque(maxlen=frame_stack)

        # Get base image shape
        if isinstance(env.observation_space, gym.spaces.Box):
            self.image_shape = env.observation_space.shape
        else:
            raise ValueError("Expected Box observation space for base env")

        # Define stacked image space
        stacked_shape = self.image_shape[:-1] + (self.image_shape[-1] * frame_stack,)
        self.stacked_image_space = gym.spaces.Box(
            low=0, high=255, shape=stacked_shape, dtype=np.uint8
        )

        # Goals
        self.goals = np.array([1.0, -1.0, 1.0], dtype=np.float32)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "image": self.stacked_image_space,
                "measurements": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(measurement_dims,),
                    dtype=np.float32,
                ),
                "goals": spaces.Box(
                    low=-1.0, high=1.0, shape=(measurement_dims,), dtype=np.float32
                ),
            }
        )

        self.measurement_buffer = []
        RetroEnvManager.get_instance().register_env(self)
        logger.info(f"DFPKungFuWrapper initialized with frame_stack={frame_stack}")

    def _get_measurements(self, info):
        measurements = np.zeros(self.measurement_dims, dtype=np.float32)
        measurements[0] = min(info.get("score_increase", 0), 1000) / 1000.0
        measurements[1] = min(info.get("damage_taken", 0), 100) / 100.0
        measurements[2] = min(info.get("progress_made", 0), 100) / 100.0
        return measurements

    def _get_stacked_image(self):
        while len(self.image_buffer) < self.frame_stack:
            self.image_buffer.append(
                self.image_buffer[-1]
                if self.image_buffer
                else np.zeros(self.image_shape, dtype=np.uint8)
            )
        return np.concatenate(self.image_buffer, axis=-1)

    def reset(self, **kwargs):
        obs_result = self.env.reset(**kwargs)
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs, info = obs_result
        else:
            obs = obs_result
            info = {}

        self.image_buffer.clear()
        for _ in range(self.frame_stack):
            self.image_buffer.append(obs)

        self.measurement_buffer = [
            np.zeros(self.measurement_dims, dtype=np.float32)
            for _ in range(self.max_horizon + 1)
        ]
        measurements = self._get_measurements(info)
        dfp_obs = {
            "image": self._get_stacked_image(),
            "measurements": measurements,
            "goals": self.goals,
        }

        return dfp_obs, info if isinstance(obs_result, tuple) else dfp_obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.image_buffer.append(obs)
        stacked_image = self._get_stacked_image()

        measurements = self._get_measurements(info)
        self.measurement_buffer.append(measurements)
        if len(self.measurement_buffer) > self.max_horizon + 1:
            self.measurement_buffer.pop(0)

        dfp_obs = {
            "image": stacked_image,
            "measurements": measurements,
            "goals": self.goals,
        }

        if len(self.measurement_buffer) > 1:
            future_measurements = {}
            for i, horizon in enumerate(self.prediction_horizons):
                if horizon < len(self.measurement_buffer):
                    future_measurements[f"future_{horizon}"] = self.measurement_buffer[
                        -horizon
                    ]
            info["future_measurements"] = future_measurements

        return dfp_obs, reward, terminated, truncated, info

    def close(self):
        super().close()
        RetroEnvManager.get_instance().unregister_env(self)


def make_kungfu_env(is_play_mode=False, frame_stack=4, use_dfp=True):
    env_manager = RetroEnvManager.get_instance()
    max_attempts = 3

    env = None
    for attempt in range(max_attempts):
        try:
            gc.collect()
            logger.info(
                f"Attempt {attempt+1}/{max_attempts}: Creating Stable Retro environment"
            )
            render_mode = "human" if is_play_mode else None
            logger.info(
                f"Attempting to create Stable Retro environment with render_mode={render_mode}"
            )
            env = retro.make(
                game="KungFu-Nes",
                render_mode=render_mode,
                inttype=retro.data.Integrations.STABLE,
            )
            logger.info("Successfully created KungFu-Nes environment")
            env_manager.register_env(env)
            env.reset()
            logger.info("Initial reset successful")
            break
        except Exception as e:
            logger.error(f"Error creating environment (attempt {attempt+1}): {e}")
            if env is not None:
                try:
                    env.close()
                except:
                    pass
                env = None
            gc.collect()
            time.sleep(1)
            if attempt == max_attempts - 1:
                logger.error("All environment creation attempts failed")
                raise RuntimeError("Could not initialize Stable Retro environment")

    if env is None:
        raise RuntimeError("Failed to create base environment")

    logger.info("Applying KungFuMasterEnv wrapper")
    env = KungFuMasterEnv(env)

    if use_dfp:
        logger.info("Adding Direct Future Prediction wrapper with frame stacking")
        env = DFPKungFuWrapper(
            env,
            frame_stack=frame_stack,
            measurement_dims=3,
            prediction_horizons=[1, 3, 5, 10, 20],
        )

    try:
        os.makedirs("logs", exist_ok=True)
        monitor_path = os.path.join("logs", "kungfu")
        logger.info(f"Setting up monitoring at {monitor_path}")
        env = Monitor(env, monitor_path)
    except Exception as e:
        logger.warning(f"Could not set up monitoring: {e}")

    logger.info("Wrapping with DummyVecEnv")
    env = DummyVecEnv([lambda: env])

    logger.info(f"Final observation space: {env.observation_space}")
    return env
