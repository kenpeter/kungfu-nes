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
    "game_submode": 0x0008,  # Game Submode
    "boss_hp": 0x04A5,  # Boss HP
    "player_y": 0x00B6,  # Hero Pos Y
    "player_action": 0x0069,  # Hero Action
    "boss_x": 0x0093,  # Boss Pos X
    "score": [0x0531, 0x0532, 0x0533, 0x0534, 0x0535],  # Score digits
    # New enemy-related addresses
    "enemy1_action_timer": 0x002B,  # Enemy 1 Action Timer
    "enemy2_action_timer": 0x002C,  # Enemy 2 Action Timer
    "enemy3_action_timer": 0x002D,  # Enemy 3 Action Timer
    "enemy4_action_timer": 0x002E,  # Enemy 4 Action Timer
    "enemy1_x": 0x008E,  # Enemy 1 Pos X
    "enemy2_x": 0x008F,  # Enemy 2 Pos X
    "enemy3_x": 0x0090,  # Enemy 3 Pos X
    "enemy4_x": 0x0091,  # Enemy 4 Pos X
    "enemy1_action": 0x0080,  # Enemy 1 Action
    "enemy2_action": 0x0081,  # Enemy 2 Action
    "enemy3_action": 0x0082,  # Enemy 3 Action
    "enemy4_action": 0x0083,  # Enemy 4 Action
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
        self.n_steps = 0
        self.episode_steps = 0

        # New tracking variables for enemies
        self.prev_enemy_positions = [0, 0, 0, 0]  # Track previous enemy X positions
        self.defeated_enemies = 0  # Count defeated enemies

        print("KungFuMasterEnv initialized - Enhanced version with enemy tracking")

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

    def get_enemy_data(self):
        """Get enemy positions, actions, and timers"""
        try:
            ram = self.env.get_ram()
            enemy_data = {}

            # Get positions
            enemy_data["positions"] = [
                int(ram[MEMORY["enemy1_x"]]),
                int(ram[MEMORY["enemy2_x"]]),
                int(ram[MEMORY["enemy3_x"]]),
                int(ram[MEMORY["enemy4_x"]]),
            ]

            # Get actions
            enemy_data["actions"] = [
                int(ram[MEMORY["enemy1_action"]]),
                int(ram[MEMORY["enemy2_action"]]),
                int(ram[MEMORY["enemy3_action"]]),
                int(ram[MEMORY["enemy4_action"]]),
            ]

            # Get timers
            enemy_data["timers"] = [
                int(ram[MEMORY["enemy1_action_timer"]]),
                int(ram[MEMORY["enemy2_action_timer"]]),
                int(ram[MEMORY["enemy3_action_timer"]]),
                int(ram[MEMORY["enemy4_action_timer"]]),
            ]

            return enemy_data
        except Exception as e:
            print(f"Error getting enemy data: {str(e)}")
            return {
                "positions": [0, 0, 0, 0],
                "actions": [0, 0, 0, 0],
                "timers": [0, 0, 0, 0],
            }

    def calculate_enemy_threat(self, player_x, enemy_data):
        """Calculate threat level from enemies based on positions and actions"""
        threat_level = 0
        active_enemies = 0
        attacking_enemies = 0

        for i in range(4):
            enemy_x = enemy_data["positions"][i]
            enemy_action = enemy_data["actions"][i]

            # Skip enemies that aren't on screen or active
            if enemy_x == 0:
                continue

            active_enemies += 1

            # Calculate distance to player
            distance = abs(player_x - enemy_x)

            # Check if enemy is in attacking action state (you may need to adjust these values)
            # Common attack actions in Kung Fu Master are often specific values
            is_attacking = enemy_action in [
                1,
                3,
                5,
                7,
            ]  # Example values, adjust based on game

            if is_attacking:
                attacking_enemies += 1

            # Closer enemies are more threatening
            if distance < 20:
                threat_level += 3 if is_attacking else 1
            elif distance < 40:
                threat_level += 2 if is_attacking else 0.5
            else:
                threat_level += 1 if is_attacking else 0

        return {
            "threat_level": threat_level,
            "active_enemies": active_enemies,
            "attacking_enemies": attacking_enemies,
        }

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

            # Initialize enemy tracking
            enemy_data = self.get_enemy_data()
            self.prev_enemy_positions = enemy_data["positions"]

            print(
                f"Initial state - Stage: {self.prev_stage}, HP: {self.prev_hp}, "
                f"X-pos: {self.prev_x_pos}, Score: {self.prev_score}, "
                f"Active enemies: {sum(1 for x in enemy_data['positions'] if x > 0)}"
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

        # Enhanced reward shaping with enemy data
        try:
            ram = self.env.get_ram()
            current_hp = int(ram[MEMORY["player_hp"]])
            current_x_pos = int(ram[MEMORY["player_x"]])
            current_stage = int(ram[MEMORY["current_stage"]])
            current_boss_hp = int(ram[MEMORY["boss_hp"]])
            current_score = self.get_score()

            # Get enemy data
            enemy_data = self.get_enemy_data()
            current_enemy_positions = enemy_data["positions"]

            # Calculate threat level
            threat_info = self.calculate_enemy_threat(current_x_pos, enemy_data)

            # Log position and status occasionally
            if self.n_steps % 100 == 0:
                time_left = (MAX_EPISODE_STEPS - self.episode_steps) / 30  # in seconds
                print(
                    f"Stage: {current_stage}, Position: {current_x_pos}, "
                    f"HP: {current_hp}, Boss HP: {current_boss_hp}, "
                    f"Score: {current_score}, Time left: {time_left:.1f}s, "
                    f"Enemies: {threat_info['active_enemies']}, Threat: {threat_info['threat_level']:.1f}"
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

            # NEW: Enemy interaction rewards
            defeated_enemies = 0
            for i in range(4):
                # If an enemy was active but now isn't, count it as defeated
                if self.prev_enemy_positions[i] > 0 and current_enemy_positions[i] == 0:
                    defeated_enemies += 1
                    shaped_reward += 2.0  # Reward for defeating enemies

            if defeated_enemies > 0:
                print(f"Defeated {defeated_enemies} enemies!")
                self.defeated_enemies += defeated_enemies

            # NEW: Threat avoidance and combat intelligence
            # Reward for being near enemies when performing attack actions
            if action in [
                1,
                8,
                9,
                11,
                12,
            ]:  # Punch, Kick, Punch+Kick, Crouch Kick, Crouch Punch
                # If there are close enemies, reward for attacking
                if threat_info["threat_level"] > 2:
                    shaped_reward += 0.2  # Small reward for appropriate attack timing

            # Small penalty for high threat situations
            shaped_reward -= threat_info["threat_level"] * 0.05

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
            self.prev_enemy_positions = current_enemy_positions.copy()

            # Update info dictionary
            info["current_stage"] = current_stage
            info["current_score"] = current_score
            info["time_remaining"] = (
                MAX_EPISODE_STEPS - self.episode_steps
            ) / 30  # in seconds
            info["active_enemies"] = threat_info["active_enemies"]
            info["threat_level"] = threat_info["threat_level"]
            info["defeated_enemies"] = self.defeated_enemies

        except Exception as e:
            print(f"Error in reward shaping: {str(e)}")
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info


def make_kungfu_env(is_play_mode=False):
    """Create a single Kung Fu Master environment wrapped for RL training"""
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

    # Frame stacking for better learning with visual inputs
    env = VecFrameStack(env, n_stack=4)

    return env
