import os
import numpy as np
import retro
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

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

# Set model path
MODEL_PATH = "model/kungfu.zip"

# Critical memory addresses based on provided RAM map
MEMORY = {
    "current_stage": 0x0058,  # Current Stage
    "player_hp": 0x04A6,  # Hero HP
    "player_x": 0x0094,  # Hero Screen Pos X
    "game_mode": 0x0051,  # Game Mode
    "boss_hp": 0x04A5,  # Boss HP
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
}

# Define action names for the environment
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

# Maximum episode duration in frames (30 frames per second * 120 seconds = 3600 frames)
MAX_EPISODE_STEPS = 3600  # 2 minutes


# Custom environment wrapper for Kung Fu Master
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # Filter out the START action from regular gameplay to prevent pausing
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
        self.n_steps = 0
        self.episode_steps = 0
        self.defeated_enemies = 0

        # Timing learning metrics
        self.defensive_actions = 0
        self.successful_defensive_actions = 0

        print(
            "KungFuMasterEnv initialized - Using enhanced frame stacking for projectile detection"
        )

    def get_stage(self):
        """Get current stage value directly from RAM"""
        try:
            ram = self.env.get_ram()
            return int(ram[MEMORY["current_stage"]])
        except:
            return 0

    def get_hp(self):
        """Get current HP value directly from RAM"""
        try:
            ram = self.env.get_ram()
            return int(ram[MEMORY["player_hp"]])
        except:
            return 0

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
        # Reset the environment
        obs_result = self.env.reset(**kwargs)

        # Handle different return types
        if isinstance(obs_result, tuple) and len(obs_result) == 2:
            obs, info = obs_result
        else:
            obs = obs_result
            info = {}

        # Reset step counter for timeout
        self.episode_steps = 0
        self.defeated_enemies = 0

        # Reset timing metrics for new episode
        defensive_success_rate = 0
        if self.defensive_actions > 0:
            defensive_success_rate = (
                self.successful_defensive_actions / self.defensive_actions
            ) * 100

        # Log defensive timing stats at end of episode if we had any defensive actions
        if hasattr(self, "defensive_actions") and self.defensive_actions > 0:
            print(
                f"Episode defensive stats - Actions: {self.defensive_actions}, "
                f"Successful: {self.successful_defensive_actions}, "
                f"Success rate: {defensive_success_rate:.1f}%"
            )

        # Reset defensive action counters
        self.defensive_actions = 0
        self.successful_defensive_actions = 0

        # Simple game start - just press START a few times
        for _ in range(5):
            self.env.step(KUNGFU_ACTIONS[3])  # Press START

        # Get initial state
        try:
            ram = self.env.get_ram()
            self.prev_hp = int(ram[MEMORY["player_hp"]])
            self.prev_x_pos = int(ram[MEMORY["player_x"]])
            self.prev_stage = int(ram[MEMORY["current_stage"]])
            self.prev_boss_hp = int(ram[MEMORY["boss_hp"]])
            self.prev_score = self.get_score()  # Initialize previous score

            print(
                f"Initial state - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
                f"X-pos: {self.prev_x_pos}, Score: {self.prev_score}"
            )

        except Exception as e:
            print(f"Error getting initial state: {str(e)}")

        return obs, info

    def step(self, action):
        # Convert to actual action
        converted_action = self.KUNGFU_ACTIONS[action]

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

        # Increment episode step counter
        self.episode_steps += 1

        # Check for timeout (2 minutes = 3600 frames)
        if self.episode_steps >= MAX_EPISODE_STEPS:
            print(f"Episode timeout after {self.episode_steps} steps (2 minutes)")
            truncated = True
            # Add timeout penalty
            reward -= 5.0

        # Enhanced reward shaping
        try:
            ram = self.env.get_ram()
            current_hp = int(ram[MEMORY["player_hp"]])
            current_x_pos = int(ram[MEMORY["player_x"]])
            current_stage = int(ram[MEMORY["current_stage"]])
            current_boss_hp = int(ram[MEMORY["boss_hp"]])
            current_score = self.get_score()

            # Log position and status occasionally
            if self.n_steps % 100 == 0:
                time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30  # in seconds
                print(
                    f"Stage: {current_stage}, Position: {current_x_pos}, "
                    f"HP: {current_hp}, Boss HP: {current_boss_hp}, "
                    f"Score: {current_score}, Time left: {time_left:.1f}s"
                )

            # Increment step counter
            self.n_steps += 1

            # Shape the reward
            shaped_reward = reward

            # Add score reward (most important progress indicator)
            score_diff = current_score - self.prev_score
            if score_diff > 0:
                shaped_reward += score_diff * 0.05  # Scale down large score changes
                if score_diff >= 100:
                    print(f"Score increased by {score_diff}! Current: {current_score}")

            # Add HP loss penalty
            hp_diff = current_hp - self.prev_hp
            if hp_diff < 0 and current_hp < 200:  # Normal health loss
                shaped_reward += hp_diff * 0.5

            # Add stage completion bonus
            if current_stage > self.prev_stage:
                print(f"Stage up! {self.prev_stage} -> {current_stage}")
                shaped_reward += 50.0  # Significant reward for stage progression
                # Reset timer urgency when reaching new stage
                self.episode_steps = max(
                    0, self.episode_steps - 600
                )  # Give 20 more seconds

            # Add boss damage bonus
            boss_hp_diff = (
                self.prev_boss_hp - current_boss_hp
                if hasattr(self, "prev_boss_hp")
                else 0
            )
            if boss_hp_diff > 0:
                shaped_reward += boss_hp_diff * 0.3  # Reward for damaging boss

            # Apply directional rewards based on stage
            x_diff = current_x_pos - self.prev_x_pos

            # Stages 0, 2, 4: reward LEFT movement
            # Stages 1, 3: reward RIGHT movement
            if current_stage in [1, 3]:  # Stages 1, 3 (move RIGHT)
                if x_diff > 0:  # Moving RIGHT
                    shaped_reward += x_diff * 0.1
                    if x_diff > 5 and self.n_steps % 50 == 0:
                        print(f"Good right movement in stage {current_stage}")
            else:  # Stages 0, 2, 4 (move LEFT)
                if x_diff < 0:  # Moving LEFT
                    shaped_reward += abs(x_diff) * 0.1
                    if abs(x_diff) > 5 and self.n_steps % 50 == 0:
                        print(f"Good left movement in stage {current_stage}")

            # Reward for well-timed defensive actions (jump/crouch)
            # If HP was maintained during an action that could be defensive, it's likely a good timing
            if action == 4 or action == 5:  # Jump or Crouch actions
                self.defensive_actions += 1

                # If we didn't lose health during a defensive action, consider it successful
                if hp_diff >= 0:
                    self.successful_defensive_actions += 1

                    # Small positive reward for potentially avoiding danger
                    shaped_reward += 0.2

                    # Extra reward if the agent maintained health and performed well
                    # This especially helps with projectile timing
                    if self.n_steps % 30 == 0:  # Don't reward too frequently
                        shaped_reward += 0.3

                    # Log successful defensive action occasionally
                    if self.n_steps % 200 == 0:
                        success_rate = (
                            self.successful_defensive_actions
                            / max(1, self.defensive_actions)
                        ) * 100
                        print(
                            f"Defensive action success rate: {success_rate:.1f}% ({self.successful_defensive_actions}/{self.defensive_actions})"
                        )

            # Add death penalty
            if current_hp == 0 and self.prev_hp > 0:
                shaped_reward -= 10.0
                print("Agent died! Applying penalty.")

            # Add urgency based on remaining time - increases penalty as time passes
            time_penalty = -0.001 * (self.episode_steps / MAX_EPISODE_STEPS)
            shaped_reward += time_penalty

            # Update previous values
            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_stage = current_stage
            self.prev_boss_hp = current_boss_hp
            self.prev_score = current_score

            # Update info dictionary
            info["current_stage"] = current_stage
            info["current_score"] = current_score
            info["time_remaining"] = (
                MAX_EPISODE_STEPS - self.episode_steps
            ) / 30  # in seconds

            # Add timing metrics to info
            if self.defensive_actions > 0:
                info["defensive_success_rate"] = (
                    self.successful_defensive_actions / self.defensive_actions
                ) * 100
            else:
                info["defensive_success_rate"] = 0
            info["defensive_actions"] = self.defensive_actions
            info["successful_defensive_actions"] = self.successful_defensive_actions

        except Exception as e:
            print(f"Error in reward shaping: {str(e)}")
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info


def make_kungfu_env(is_play_mode=False, frame_stack=8):
    """Create a single Kung Fu Master environment wrapped for RL training

    Args:
        is_play_mode: Whether to render the environment
        frame_stack: Number of frames to stack (4 or 8)
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

    env = KungFuMasterEnv(env)
    os.makedirs("logs", exist_ok=True)
    env = Monitor(env, os.path.join("logs", "kungfu"))

    # Wrap in DummyVecEnv for compatibility with stable-baselines3
    env = DummyVecEnv([lambda: env])

    # Configure frame stacking based on parameter
    print(f"Using frame stacking with n_stack={frame_stack}")
    env = VecFrameStack(env, n_stack=frame_stack)

    return env
