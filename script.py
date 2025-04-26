import os
import sys
import argparse
import numpy as np
import retro
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from datetime import datetime
from tqdm import tqdm

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

# Set model path
MODEL_PATH = "model/kungfu.zip"


# Create custom environment wrapper for Kung Fu Master with discrete actions
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super(KungFuMasterEnv, self).__init__(env)

        # empty arr to hold actions except start
        filtered_actions = []
        filtered_action_names = []

        for i, action in enumerate(KUNGFU_ACTIONS):
            # Skip the START action (index 3)
            if i != 3:
                filtered_actions.append(action)
                filtered_action_names.append(KUNGFU_ACTION_NAMES[i])

        # Override the global action lists with filtered versions
        self.KUNGFU_ACTIONS = filtered_actions
        self.KUNGFU_ACTION_NAMES = filtered_action_names

        # Override the action space to use our discrete actions
        self.action_space = gym.spaces.Discrete(len(KUNGFU_ACTIONS))

        # Track previous game state to calculate differences
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_boss_hp = 0
        self.prev_stage = 0
        self.prev_enemy_x = [0, 0, 0, 0]  # Previous X positions for 4 enemies
        self.prev_enemy_actions = [0, 0, 0, 0]  # Previous actions for 4 enemies

        # Constants for reward shaping
        self.SCORE_REWARD_SCALE = 0.01  # Per point of score
        self.HP_LOSS_PENALTY = -1.0  # Per point of HP lost
        self.STAGE_COMPLETION_REWARD = 50.0
        self.DEATH_PENALTY = -25.0
        self.PROGRESS_REWARD_SCALE = 0.05  # Reward for moving in correct direction
        self.BOSS_DAMAGE_REWARD = 1.0  # Per point of damage to boss
        self.TIME_PRESSURE_PENALTY = -0.01  # Small penalty per step to encourage speed
        self.ENEMY_PROXIMITY_AWARENESS = (
            0.5  # Reward for appropriate action with enemies nearby
        )
        self.ENEMY_DEFEAT_BONUS = 2.0  # Bonus for enemy disappearing (defeated)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        # Automatically press START to begin the game
        for _ in range(3):  # Press START multiple times to ensure it registers
            start_action = KUNGFU_ACTIONS[3]  # Index 3 is the START action
            temp_obs, _, _, _, _ = self.env.step(start_action)
            # Don't overwrite the original obs

        # Read initial state
        ram = self.env.get_ram()
        self.prev_score = self._get_score(ram)
        self.prev_hp = ram[0x04A6]
        self.prev_x_pos = ram[0x0094]
        self.prev_boss_hp = ram[0x04A5]
        self.prev_stage = ram[0x0058]

        # Read enemy positions and actions
        self.prev_enemy_x = [
            ram[0x008E],  # Enemy 1 X position
            ram[0x008F],  # Enemy 2 X position
            ram[0x0090],  # Enemy 3 X position
            ram[0x0091],  # Enemy 4 X position
        ]

        self.prev_enemy_actions = [
            ram[0x0080],  # Enemy 1 action
            ram[0x0081],  # Enemy 2 action
            ram[0x0082],  # Enemy 3 action
            ram[0x0083],  # Enemy 4 action
        ]

        return obs

    def step(self, action):
        # Convert our discrete action to the multi-binary format
        # If action is START (index 3), replace with NO-OP (index 0)
        if action == 3:  # START action
            action = 0  # Replace with NO-OP

        # Convert our discrete action to the multi-binary format
        converted_action = KUNGFU_ACTIONS[action]
        obs, reward, terminated, truncated, info = self.env.step(converted_action)
        done = terminated or truncated

        # Get current RAM state
        ram = self.env.get_ram()

        # Extract relevant game state (safely convert to int to avoid overflow)
        current_stage = int(ram[0x0058])
        current_score = self._get_score(ram)
        current_hp = int(ram[0x04A6])
        current_x_pos = int(ram[0x0094])
        current_y_pos = int(ram[0x00B6])  # Hero Y position
        current_boss_hp = int(ram[0x04A5])

        # Get current enemy positions and actions (safely convert to int)
        current_enemy_x = [
            int(ram[0x008E]),  # Enemy 1 X position
            int(ram[0x008F]),  # Enemy 2 X position
            int(ram[0x0090]),  # Enemy 3 X position
            int(ram[0x0091]),  # Enemy 4 X position
        ]

        current_enemy_actions = [
            int(ram[0x0080]),  # Enemy 1 action
            int(ram[0x0081]),  # Enemy 2 action
            int(ram[0x0082]),  # Enemy 3 action
            int(ram[0x0083]),  # Enemy 4 action
        ]

        # Calculate shaped reward
        shaped_reward = 0.0

        # Base reward from the game
        shaped_reward += reward

        # Reward for score increase (safe subtraction using our helper method)
        score_diff = self._safe_subtract(current_score, self.prev_score)
        if score_diff > 0:
            shaped_reward += score_diff * self.SCORE_REWARD_SCALE

        # Penalty for health loss (safe subtraction)
        hp_diff = self._safe_subtract(current_hp, self.prev_hp)
        if hp_diff < 0:
            shaped_reward += hp_diff * self.HP_LOSS_PENALTY

        # Reward/penalty for movement in correct direction based on stage
        direction_reward = self._calculate_direction_reward(
            current_stage, current_x_pos, self.prev_x_pos
        )
        shaped_reward += direction_reward

        # Reward for damaging boss (safe subtraction)
        boss_hp_diff = self._safe_subtract(current_boss_hp, self.prev_boss_hp)
        if boss_hp_diff < 0:
            shaped_reward += abs(boss_hp_diff) * self.BOSS_DAMAGE_REWARD

        # Reward for stage completion
        if current_stage > self.prev_stage:
            shaped_reward += self.STAGE_COMPLETION_REWARD

        # Enemy handling rewards
        enemy_reward = self._calculate_enemy_handling_reward(
            action,
            current_x_pos,
            current_y_pos,
            current_enemy_x,
            current_enemy_actions,
            self.prev_enemy_x,
            self.prev_enemy_actions,
            current_stage,  # Pass current stage to enemy handling
        )
        shaped_reward += enemy_reward

        # Penalty for death
        if done and "lives" in info and info["lives"] == 0:
            shaped_reward += self.DEATH_PENALTY

        # Small time pressure penalty to encourage finishing quickly
        shaped_reward += self.TIME_PRESSURE_PENALTY

        # Update previous state
        self.prev_score = current_score
        self.prev_hp = current_hp
        self.prev_x_pos = current_x_pos
        self.prev_boss_hp = current_boss_hp
        self.prev_stage = current_stage
        self.prev_enemy_x = current_enemy_x.copy()
        self.prev_enemy_actions = current_enemy_actions.copy()

        # Add debugging info
        info["shaped_reward"] = shaped_reward
        info["current_stage"] = current_stage
        info["player_hp"] = current_hp
        info["player_x"] = current_x_pos
        info["enemy_positions"] = current_enemy_x
        info["enemy_actions"] = current_enemy_actions
        info["stage_direction"] = self._get_stage_direction(current_stage)

        return obs, shaped_reward, terminated, truncated, info

    def _get_stage_direction(self, stage):
        """Return the correct movement direction for the current stage"""
        # Stage 1, 3, 5: Go left (decreasing X)
        if stage in [1, 3, 5]:
            return -1  # Left direction
        # Stage 2, 4: Go right (increasing X)
        elif stage in [2, 4]:
            return 1  # Right direction
        return 0

    def _safe_subtract(self, a, b):
        """Perform safe subtraction for uint8 values that might overflow"""
        # Convert to integers and handle wraparound
        a_int, b_int = int(a), int(b)

        # If values are within reasonable range, do normal subtraction
        if abs(a_int - b_int) < 128:
            return a_int - b_int

        # Special case for HP or similar where 0 might mean "empty" and max might mean "full"
        if a_int == 0 and b_int > 200:  # Likely went from max to 0
            return -b_int  # Return negative of previous value

        # Special case for wraparound (e.g., 255 to 0)
        if a_int < 50 and b_int > 200:  # Likely wrapped around
            return (a_int + 256) - b_int

        # Default to normal subtraction
        return a_int - b_int

    def _calculate_direction_reward(self, stage, current_x, prev_x):
        """Calculate reward for moving in the correct direction based on stage"""
        # Safe movement calculation
        movement = self._safe_subtract(current_x, prev_x)

        # If no movement, no reward
        if movement == 0:
            return 0

        # Get the correct direction for this stage
        stage_direction = self._get_stage_direction(stage)

        # If moving in the correct direction
        if (stage_direction < 0 and movement < 0) or (
            stage_direction > 0 and movement > 0
        ):
            return abs(movement) * self.PROGRESS_REWARD_SCALE
        else:
            # Smaller penalty for moving in wrong direction
            # Make sure we don't overflow by using float calculation
            return -float(abs(movement)) * self.PROGRESS_REWARD_SCALE * 0.5

    def _calculate_enemy_handling_reward(
        self,
        action,
        player_x,
        player_y,
        current_enemy_x,
        current_enemy_actions,
        prev_enemy_x,
        prev_enemy_actions,
        current_stage,  # Added current stage parameter
    ):
        """Calculate rewards for handling enemies appropriately based on their direction and actions"""
        reward = 0.0

        # Get the correct stage direction
        stage_direction = self._get_stage_direction(current_stage)

        # Determine if player used an attack action
        is_attack_action = action in [1, 8, 9, 11, 12]  # Punch, Kick, etc.

        # For appropriate direction facing, we'll reward facing the correct way
        # For stages where we go left (1,3,5), reward LEFT action
        # For stages where we go right (2,4), reward RIGHT action
        if (stage_direction < 0 and action == 6) or (
            stage_direction > 0 and action == 7
        ):
            reward += 0.2  # Reward for facing the correct direction for the stage

        # Track if new enemies have appeared (slots that were empty but now have enemies)
        new_enemies_appeared = 0
        for i in range(4):
            if prev_enemy_x[i] == 0 and current_enemy_x[i] != 0:
                new_enemies_appeared += 1

        # Small penalty for new enemies appearing to encourage proactive enemy handling
        if new_enemies_appeared > 0:
            reward -= new_enemies_appeared * 0.2

        for i in range(4):
            # Skip if both current and previous positions are 0 (no enemy in this slot)
            if current_enemy_x[i] == 0 and prev_enemy_x[i] == 0:
                continue

            # Check if enemy was defeated (was present before, now gone)
            # This could be due to defeat or the enemy moving out of screen
            if prev_enemy_x[i] != 0 and current_enemy_x[i] == 0:
                # If score also increased, likely defeated an enemy
                if self.prev_score < self._get_score(self.env.get_ram()):
                    reward += self.ENEMY_DEFEAT_BONUS
                continue

            # Calculate enemy proximity and direction relative to player
            if current_enemy_x[i] != 0:
                # Safe distance calculation
                enemy_distance = abs(self._safe_subtract(player_x, current_enemy_x[i]))
                enemy_direction = (
                    1 if current_enemy_x[i] > player_x else -1
                )  # 1: enemy on right, -1: enemy on left

                # Determine if enemy is approaching or retreating
                is_approaching = False
                if prev_enemy_x[i] != 0:  # If we have previous position data
                    prev_dist = abs(self._safe_subtract(prev_enemy_x[i], player_x))
                    curr_dist = abs(self._safe_subtract(current_enemy_x[i], player_x))

                    # Check if distance is decreasing (approaching)
                    if curr_dist < prev_dist:
                        is_approaching = True

                # Reward appropriate actions based on enemy proximity and direction
                if enemy_distance < 30:  # Enemy is close
                    # Reward attacking nearby enemies
                    if is_attack_action:
                        reward += self.ENEMY_PROXIMITY_AWARENESS

                        # Extra reward for attacking an approaching enemy
                        if is_approaching:
                            reward += self.ENEMY_PROXIMITY_AWARENESS * 0.5

                    # Reward facing the enemy (even if not attacking)
                    elif (enemy_direction < 0 and action == 6) or (
                        enemy_direction > 0 and action == 7
                    ):
                        reward += self.ENEMY_PROXIMITY_AWARENESS * 0.3

                # For enemies at medium distance
                elif enemy_distance < 60:
                    # Smaller reward for moving toward enemies (aggressive play)
                    if (enemy_direction < 0 and action == 6) or (
                        enemy_direction > 0 and action == 7
                    ):
                        reward += 0.1

                # For distant enemies, small reward for strategic positioning
                else:
                    # If player is in correct stage direction and moving that way
                    if (stage_direction < 0 and action == 6) or (
                        stage_direction > 0 and action == 7
                    ):
                        reward += 0.05

        return reward

    def _get_score(self, ram):
        """Extract score from RAM values (5 bytes, BCD encoded)"""
        # Start with a small initial value to avoid the risk of zero score
        score = 1

        # For games like this, often we just need a relative score change
        # rather than the exact value, so a simplified approach is better

        # Check if the score memory locations have any non-zero values
        has_score = False
        for addr in [0x0531, 0x0532, 0x0533, 0x0534, 0x0535]:
            if ram[addr] > 0:
                has_score = True
                score += ram[addr]  # Just add the raw values

        # If no score detected, return base value
        if not has_score:
            return 1

        return score


def make_kungfu_env(num_envs=1, is_play_mode=False):
    """Create vectorized environments for Kung Fu Master with frame stacking"""

    def make_env(rank):
        def _init():
            try:
                # Set render_mode only for play mode
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFu-Nes", render_mode=render_mode)
            except Exception as e:
                # If that fails, try alternative ROM name
                print(f"Failed to load KungFu-Nes, trying alternative ROM name: {e}")
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)

            # Wrap the environment with our custom wrapper
            env = KungFuMasterEnv(env)

            # Add monitoring
            os.makedirs("logs", exist_ok=True)
            env = Monitor(env, os.path.join("logs", f"kungfu_{rank}"))

            return env

        return _init

    if num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

    # Stack 4 frames to give the model temporal context
    n_stack = 4  # This is explicitly setting the frame stack to 4
    env = VecFrameStack(env, n_stack=n_stack)

    print(f"Environment created with frame stack of {n_stack}")
    print(f"Observation space shape: {env.observation_space.shape}")

    return env


def create_model(env, resume=False):
    """Create or load a PPO model optimized for frame stacked observations"""
    # Define neural network architecture
    policy_kwargs = dict(
        net_arch=[64, 64]  # Simple network with two hidden layers of 64 units
    )

    if resume and os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model")
        model = PPO(
            "CnnPolicy",  # CNN policy for frame stacked image inputs
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=None,  # Explicitly disable tensorboard
        )

    return model


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super(ProgressBarCallback, self).__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()
        self.pbar = None


class ImprovedKungFuModelCallback(BaseCallback):
    """
    Advanced callback for Kung Fu Master RL training that implements a hybrid model saving strategy:
    1. Moving average performance tracking
    2. Periodic checkpoints
    3. Best model tracking based on weighted metrics
    4. Protection against saving models that overfit to lucky episodes
    """

    # init with moving avg win (100 default)
    def __init__(
        self,
        check_freq=10000,
        model_dir="model",
        verbose=1,
        moving_avg_window=100,
        checkpoint_freq=100000,  # Save checkpoint every 100k steps
        min_steps_between_saves=50000,  # Prevent saving too frequently
    ):
        # super, check freq, model dir, moving agv, check point freq, min step between saves
        super(ImprovedKungFuModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.model_dir = model_dir
        self.moving_avg_window = moving_avg_window
        self.checkpoint_freq = checkpoint_freq
        self.min_steps_between_saves = min_steps_between_saves

        # Create directories
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "best_models"), exist_ok=True)

        # Initialize best metrics tracking
        self.best_reward = -np.inf
        self.best_progress = -np.inf
        self.best_weighted_score = -np.inf
        self.best_moving_avg_reward = -np.inf

        # Track the last step when we saved a model
        self.last_save_step = 0

        # Moving average tracking

        self.episode_rewards = []
        self.episode_stages = []
        self.episode_metrics = []  # Store complete metrics for each episode

        # Current episode tracking
        self.current_ep_reward = 0
        self.current_ep_max_stage = 1
        self.current_ep_x_progress_by_stage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.ep_start_step = 0

        # Track action distribution
        self.action_counts = {i: 0 for i in range(len(KUNGFU_ACTIONS))}
        self.total_actions = 0
        self.last_log_step = 0
        self.action_log_freq = 5000

    def _on_training_start(self):
        # Create timestamp for this training run
        from datetime import datetime

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Log training start
        print(f"Training started at {self.timestamp}")
        print(f"Model directory: {self.model_dir}")
        print(f"Moving average window: {self.moving_avg_window} episodes")
        print(f"Checkpoint frequency: {self.checkpoint_freq} steps")

        # Initialize episode tracking
        self.ep_start_step = self.n_calls

    def _on_step(self):
        # Track actions taken
        for env_idx in range(self.training_env.num_envs):
            try:
                action = self.locals["actions"][env_idx]
                self.action_counts[action] += 1
                self.total_actions += 1
            except (KeyError, IndexError):
                pass

        # Log action distribution periodically
        if self.n_calls - self.last_log_step >= self.action_log_freq:
            self._log_action_percentages()
            self.last_log_step = self.n_calls

        # Track episode metrics
        for env_idx in range(self.training_env.num_envs):
            try:
                info = self.locals["infos"][env_idx]

                # Track current episode reward
                if "shaped_reward" in info:
                    self.current_ep_reward += info["shaped_reward"]

                # Track stage progress
                if "current_stage" in info:
                    stage = info["current_stage"]
                    self.current_ep_max_stage = max(self.current_ep_max_stage, stage)

                    # Track X progress based on direction for each stage
                    if "player_x" in info:
                        x_pos = info["player_x"]
                        # For stages 1,3,5 progress is left (decreasing X)
                        # For stages 2,4 progress is right (increasing X)
                        if stage in [1, 3, 5]:
                            # Convert to a progress value (invert since lower is better)
                            progress = 255 - x_pos
                        else:
                            progress = x_pos

                        self.current_ep_x_progress_by_stage[stage] = max(
                            self.current_ep_x_progress_by_stage.get(stage, 0),
                            progress,
                        )

                # Check if episode ended
                done = self.locals["dones"][env_idx]
                if done:
                    # Calculate episode metrics
                    ep_metrics = self._calculate_episode_metrics()

                    # Store for moving average
                    self.episode_rewards.append(self.current_ep_reward)
                    self.episode_stages.append(self.current_ep_max_stage)
                    self.episode_metrics.append(ep_metrics)

                    # Keep only the window size
                    if len(self.episode_rewards) > self.moving_avg_window:
                        self.episode_rewards.pop(0)
                        self.episode_stages.pop(0)
                        self.episode_metrics.pop(0)

                    # Log episode stats
                    if self.verbose > 0:
                        steps = self.n_calls - self.ep_start_step
                        print(f"\nEpisode finished after {steps} steps:")
                        print(f"  Reward: {self.current_ep_reward:.2f}")
                        print(f"  Max Stage: {self.current_ep_max_stage}")
                        print(f"  Weighted Score: {ep_metrics['weighted_score']:.4f}")

                    # Reset episode tracking
                    self.current_ep_reward = 0
                    self.current_ep_max_stage = 1
                    self.current_ep_x_progress_by_stage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    self.ep_start_step = self.n_calls

            except (KeyError, IndexError):
                pass

        # Save periodic checkpoints
        if self.n_calls % self.checkpoint_freq == 0 and self.n_calls > 0:
            checkpoint_path = os.path.join(
                self.model_dir, "checkpoints", f"checkpoint_{self.n_calls:010d}.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved periodic checkpoint to {checkpoint_path}")

        # Evaluate for best model saving
        if self.n_calls % self.check_freq == 0 and len(self.episode_metrics) > 0:
            # Calculate moving averages
            moving_avg_reward = np.mean([m["reward"] for m in self.episode_metrics])
            moving_avg_stage = np.mean([m["max_stage"] for m in self.episode_metrics])
            moving_avg_score = np.mean(
                [m["weighted_score"] for m in self.episode_metrics]
            )

            # Log current moving averages
            if self.verbose > 0:
                print("\n--- MOVING AVERAGES ---")
                print(f"Window size: {len(self.episode_metrics)} episodes")
                print(f"Avg Reward: {moving_avg_reward:.2f}")
                print(f"Avg Max Stage: {moving_avg_stage:.2f}")
                print(f"Avg Weighted Score: {moving_avg_score:.4f}")
                print("----------------------")

            # Determine if we should save based on improvement and minimum time between saves
            steps_since_last_save = self.n_calls - self.last_save_step

            # Check multiple conditions for saving:
            should_save = False
            save_reason = ""

            # 1. Check if moving average weighted score improved
            if (
                moving_avg_score > self.best_moving_avg_reward
                and steps_since_last_save >= self.min_steps_between_saves
            ):
                should_save = True
                save_reason = "improved moving average"
                self.best_moving_avg_reward = moving_avg_score

            # 2. Check if any episode had exceptional performance (significantly better than average)
            if len(self.episode_metrics) >= 10:  # Only if we have enough episodes
                recent_best = max(
                    [m["weighted_score"] for m in self.episode_metrics[-10:]]
                )
                if (
                    recent_best > self.best_weighted_score * 1.2
                    and steps_since_last_save >= self.min_steps_between_saves
                ):
                    should_save = True
                    save_reason = "exceptional recent episode"
                    self.best_weighted_score = recent_best

            # Save if conditions met
            if should_save:
                self.last_save_step = self.n_calls

                # Save as named model with timestamp and metrics
                best_model_path = os.path.join(
                    self.model_dir,
                    "best_models",
                    f"best_model_{self.timestamp}_step{self.n_calls}_score{moving_avg_score:.4f}.zip",
                )

                # Also save as best_model.zip for convenience
                standard_best_path = os.path.join(self.model_dir, "best_model.zip")

                self.model.save(best_model_path)
                self.model.save(standard_best_path)

                if self.verbose > 0:
                    print(f"\nSaving new best model: {save_reason}")
                    print(f"Saved to {best_model_path}")
                    print(f"Also saved to {standard_best_path}")

                # Log action percentages when saving best model
                self._log_action_percentages()

        return True

    def _calculate_episode_metrics(self):
        """Calculate comprehensive metrics for the current episode"""
        # Get normalized X progress score (average across stages encountered)
        x_progress_scores = []
        for stage in range(1, self.current_ep_max_stage + 1):
            if stage in self.current_ep_x_progress_by_stage:
                # Normalize to 0-1 range
                if stage in [1, 3, 5]:  # Left stages
                    norm_progress = self.current_ep_x_progress_by_stage[stage] / 255
                else:  # Right stages
                    norm_progress = self.current_ep_x_progress_by_stage[stage] / 255
                x_progress_scores.append(norm_progress)

        avg_x_progress = (
            sum(x_progress_scores) / len(x_progress_scores) if x_progress_scores else 0
        )

        # Calculate weighted score for this episode
        weighted_score = (
            0.6 * self.current_ep_max_stage  # Stage progress (max weight)
            + 0.3 * avg_x_progress  # X position progress
            + 0.1
            * min(1.0, self.current_ep_reward / 1000)  # Normalized reward (cap at 1.0)
        )

        return {
            "reward": self.current_ep_reward,
            "max_stage": self.current_ep_max_stage,
            "avg_x_progress": avg_x_progress,
            "weighted_score": weighted_score,
        }

    def _log_action_percentages(self):
        """Log the percentage of each action taken during training"""
        if self.total_actions == 0:
            return

        print("\n--- ACTION DISTRIBUTION ---")
        print(f"Total actions: {self.total_actions}")

        # Calculate and display percentages
        percentages = []
        for action_idx, count in self.action_counts.items():
            percentage = (count / self.total_actions) * 100
            action_name = KUNGFU_ACTION_NAMES[action_idx]
            percentages.append((action_name, percentage))

        # Sort by percentage (descending)
        percentages.sort(key=lambda x: x[1], reverse=True)

        # Display in a nice format
        for action_name, percentage in percentages:
            print(f"{action_name:<15}: {percentage:.2f}%")

        print("---------------------------\n")

    def _on_training_end(self):
        """Final operations at end of training"""
        # Save final model
        final_model_path = os.path.join(self.model_dir, "final_model.zip")
        self.model.save(final_model_path)

        # Log summary of training
        print("\n=== TRAINING COMPLETE ===")
        print(f"Total steps: {self.n_calls}")
        print(f"Best moving average score: {self.best_moving_avg_reward:.4f}")
        print(f"Final model saved to: {final_model_path}")

        # Save a performance summary
        if len(self.episode_metrics) > 0:
            summary_path = os.path.join(self.model_dir, "training_summary.txt")
            with open(summary_path, "w") as f:
                f.write(
                    f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Total steps: {self.n_calls}\n")
                f.write(
                    f"Final moving average (over {len(self.episode_metrics)} episodes):\n"
                )
                f.write(
                    f"  Reward: {np.mean([m['reward'] for m in self.episode_metrics]):.2f}\n"
                )
                f.write(
                    f"  Max Stage: {np.mean([m['max_stage'] for m in self.episode_metrics]):.2f}\n"
                )
                f.write(
                    f"  Weighted Score: {np.mean([m['weighted_score'] for m in self.episode_metrics]):.4f}\n"
                )

                # Add action distribution
                f.write("\nFinal Action Distribution:\n")
                percentages = [
                    (KUNGFU_ACTION_NAMES[i], (count / self.total_actions) * 100)
                    for i, count in self.action_counts.items()
                ]
                percentages.sort(key=lambda x: x[1], reverse=True)

                for action_name, percentage in percentages:
                    f.write(f"  {action_name:<15}: {percentage:.2f}%\n")

            print(f"Training summary saved to: {summary_path}")


def train_model(model, timesteps):
    """Train the model with improved model saving strategy"""
    print(f"Training for {timesteps} timesteps...")

    # Define callbacks
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)

    # Create the improved callback
    improved_callback = ImprovedKungFuModelCallback(
        check_freq=5000,
        model_dir=model_dir,
        verbose=1,
        moving_avg_window=20,  # Track performance over 50 episodes
        checkpoint_freq=100000,  # Save checkpoint every 100k steps
        min_steps_between_saves=50000,  # Prevent saving too frequently
    )

    # Create simple progress callback
    progress_callback = ProgressBarCallback(timesteps)

    # Train the model with callbacks
    model.learn(
        total_timesteps=timesteps, callback=[progress_callback, improved_callback]
    )

    # Use best model path as final model or just use the final model
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Using best model as final model: {best_model_path}")
        print(f"Final model is at: {MODEL_PATH}")
        if best_model_path != MODEL_PATH:
            import shutil

            shutil.copy(best_model_path, MODEL_PATH)
    else:
        # This should not happen with our callback, but just in case
        model.save(MODEL_PATH)
        print(f"Final model saved to {MODEL_PATH}")


def play_game(env, model, episodes=5):
    """Use the trained model to play the game with rendering"""
    obs = env.reset()
    print(f"Observation shape during play: {obs.shape}")  # Should show stacked frames

    from tqdm import tqdm

    # Action tracking variables
    action_counts = {i: 0 for i in range(len(KUNGFU_ACTIONS))}
    total_actions = 0

    # Flag to track if we're still in the start screen
    in_start_screen = True
    start_screen_counter = 0

    for episode in range(episodes):
        done = [False]  # Initialize as list for vectorized env
        total_reward = 0
        step = 0

        # Reset start screen tracking for each episode
        in_start_screen = True
        start_screen_counter = 0

        # Track the current stage for directional logic
        current_stage = 1  # Default to stage 1 at start

        # Track stats for this episode
        max_stage = 1
        stage_progress = {}

        # Create progress bar for this episode
        pbar = tqdm(desc=f"Episode {episode+1}", leave=True)

        # For action forcing to ensure variety
        steps_since_attack = 0
        force_directional_steps = 0
        force_attack_steps = 0

        while not any(done):
            # Get the model's predicted action
            action, _ = model.predict(obs, deterministic=True)
            original_action = action[0]

            # Get game state info for better decision making
            info = env.get_attr("unwrapped")[0]  # Get wrapped env info
            current_stage = 1  # Default
            x_pos = 128  # Default middle position
            ram = None

            if hasattr(info, "get_ram"):
                try:
                    # Get current RAM state
                    ram = info.get_ram()
                    current_stage = int(ram[0x0058])
                    x_pos = int(ram[0x0094])
                    y_pos = int(ram[0x00B6])

                    # Check if we're still in the start screen (no real gameplay yet)
                    # This is a heuristic - may need adjustment based on game specifics
                    if current_stage == 0 or (current_stage == 1 and ram[0x04A6] == 0):
                        in_start_screen = True
                        start_screen_counter += 1
                    else:
                        in_start_screen = False
                        start_screen_counter = 0

                except Exception as e:
                    print(f"Error reading RAM: {e}")

            # PHASE 1: Handle start screen - REMOVED CODE FOR PRESSING START
            # We're letting the environment handle the start button press
            if in_start_screen:
                # Set a NO-OP action when in start screen
                action[0] = 0  # NO-OP action
                if start_screen_counter % 30 == 0:
                    print("In start screen - waiting for environment to press START")

            # PHASE 2: Actual gameplay - fix the agent's behavior
            else:
                # Force action variety to ensure agent uses full capabilities
                steps_since_attack += 1

                # Get stage direction
                stage_direction = -1 if current_stage in [1, 3, 5] else 1

                # FORCE ATTACKS: If we haven't attacked in a while, force an attack
                if steps_since_attack > 15 and force_attack_steps == 0:
                    # Pick a random attack action
                    attack_actions = [1, 8, 9, 11, 12]  # Punch, Kick, etc.
                    import random

                    action[0] = random.choice(attack_actions)
                    steps_since_attack = 0
                    force_attack_steps = 5  # Force attack for a few frames
                    print(f"Forcing attack action: {KUNGFU_ACTION_NAMES[action[0]]}")
                elif force_attack_steps > 0:
                    # Continue forced attack for combo
                    attack_actions = [1, 8, 9, 11, 12]  # Punch, Kick, etc.
                    import random

                    action[0] = random.choice(attack_actions)
                    force_attack_steps -= 1

                # FORCE DIRECTION: Ensure we move in the correct stage direction
                elif force_directional_steps > 0:
                    if stage_direction < 0:  # Should move left (stages 1,3,5)
                        action[0] = 6  # LEFT
                    else:  # Should move right (stages 2,4)
                        action[0] = 7  # RIGHT
                    force_directional_steps -= 1

                # Regular directional behavior based on stage
                elif current_stage > 0:  # Only apply if in a real stage
                    # Every ~10 steps, force correct directional movement
                    if step % 10 == 0:
                        force_directional_steps = 3  # Force direction for a few frames
                        if stage_direction < 0:  # Should move left
                            action[0] = 6  # LEFT action
                            print(f"Forcing LEFT movement for stage {current_stage}")
                        else:  # Should move right
                            action[0] = 7  # RIGHT action
                            print(f"Forcing RIGHT movement for stage {current_stage}")

                # If we're near screen edges, enforce correct movement
                if ram is not None:
                    if (
                        x_pos < 50 and stage_direction > 0
                    ):  # Too far left in a right-moving stage
                        action[0] = 7  # Force RIGHT
                    elif (
                        x_pos > 200 and stage_direction < 0
                    ):  # Too far right in a left-moving stage
                        action[0] = 6  # Force LEFT

                # If model isn't using combat actions or moving, override occasionally
                # IMPORTANT: Removed START from this check to prevent accidental START presses
                if step % 5 == 0 and action[0] in [0, 4, 5]:  # No-op, UP, DOWN
                    # Randomly pick an action that would be useful (attack or move in correct direction)
                    useful_actions = [1, 8, 9]  # Basic attacks
                    if stage_direction < 0:
                        useful_actions.append(6)  # LEFT if we should be moving left
                    else:
                        useful_actions.append(7)  # RIGHT if we should be moving right

                    import random

                    action[0] = random.choice(useful_actions)
                    print(
                        f"Overriding passive action with {KUNGFU_ACTION_NAMES[action[0]]}"
                    )

                # If we're doing a normal attack reset the steps_since_attack counter
                if action[0] in [1, 8, 9, 11, 12]:  # Punch, Kick, etc.
                    steps_since_attack = 0

            # Ensure the agent NEVER presses START (action index 3)
            if action[0] == 3:  # START action
                action[0] = 0  # Replace with NO-OP
                print("Prevented agent from pressing START button")

            # Track action frequencies (after all modifications)
            action_counts[action[0]] += 1
            total_actions += 1

            # Handle API differences between Gymnasium and stable-baselines3 VecEnv
            # VecEnv still uses the older format: obs, reward, done, info
            obs, reward, done, info = env.step(action)

            # Display action being taken
            action_name = KUNGFU_ACTION_NAMES[action[0]]

            # Update progress bar description with current info
            pbar.set_description(
                f"Episode {episode+1} | Step: {step} | Action: {action_name} | Reward: {reward[0]:.2f} | Stage: {current_stage}"
            )
            pbar.update(1)

            total_reward += reward[0]
            step += 1

            if any(done):
                # Calculate normalized progress scores
                progress_scores = []
                for stage, prog in stage_progress.items():
                    normalized = prog / 255.0  # Normalize to 0-1 range
                    progress_scores.append(normalized)

                # Calculate average progress if we have data
                avg_progress = (
                    sum(progress_scores) / len(progress_scores)
                    if progress_scores
                    else 0
                )

                # Print detailed episode stats
                print(f"\nEpisode {episode+1} finished:")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Max stage reached: {max_stage}")
                print(f"  Average stage progress: {avg_progress:.2f}")
                print(f"  Total steps: {step}")

                # Print progress by stage
                print("  Progress by stage:")
                for stage in sorted(stage_progress.keys()):
                    norm_prog = stage_progress[stage] / 255.0
                    print(f"    Stage {stage}: {norm_prog:.2f}")

                pbar.close()
                obs = env.reset()
                break

    # Display action distribution after all episodes
    if total_actions > 0:
        print("\n--- ACTION DISTRIBUTION DURING PLAY ---")
        print(f"Total actions: {total_actions}")

        # Calculate and display percentages
        percentages = []
        for action_idx, count in action_counts.items():
            percentage = (count / total_actions) * 100
            action_name = KUNGFU_ACTION_NAMES[action_idx]
            percentages.append((action_name, percentage))

        # Sort by percentage (descending)
        percentages.sort(key=lambda x: x[1], reverse=True)

        # Display in a nice format
        for action_name, percentage in percentages:
            print(f"{action_name:<15}: {percentage:.2f}%")

        print("----------------------------------")

        # Save action distribution to a file
        action_log_path = os.path.join(
            os.path.dirname(MODEL_PATH), "action_distribution_play.txt"
        )
        with open(action_log_path, "w") as f:
            f.write(f"Action distribution during play ({episodes} episodes):\n")
            f.write(f"Total actions: {total_actions}\n\n")

            for action_name, percentage in percentages:
                f.write(f"{action_name:<15}: {percentage:.2f}%\n")

        print(f"Action distribution saved to: {action_log_path}")


def evaluate_models(episodes=3):
    """Evaluate all saved models to compare performance"""
    import glob
    import pandas as pd

    print(f"Evaluating all saved models ({episodes} episodes each)...")

    # Find all model files
    model_dir = os.path.dirname(MODEL_PATH)
    model_files = []

    # Look in best_models directory
    best_models = glob.glob(os.path.join(model_dir, "best_models", "*.zip"))
    model_files.extend(best_models)

    # Look in checkpoints directory
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoints", "*.zip"))
    model_files.extend(checkpoints)

    # Add standard models if they exist
    for std_name in ["best_model.zip", "final_model.zip", "kungfu.zip"]:
        std_path = os.path.join(model_dir, std_name)
        if os.path.exists(std_path) and std_path not in model_files:
            model_files.append(std_path)

    if not model_files:
        print("No saved models found to evaluate.")
        return

    print(f"Found {len(model_files)} models to evaluate.")

    # Create environment for evaluation
    env = make_kungfu_env(num_envs=1, is_play_mode=False)

    # Track results
    results = []

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model: {model_name}")

        try:
            # Load the model
            model = PPO.load(model_path)

            # Run episodes and collect stats
            episode_rewards = []
            episode_stages = []
            episode_progress = []

            for episode in range(episodes):
                obs = env.reset()
                done = [False]
                total_reward = 0
                max_stage = 1
                stage_progress = {}

                while not any(done):
                    # Get action from model
                    action, _ = model.predict(obs, deterministic=True)

                    # Step environment
                    obs, reward, done, info = env.step(action)

                    # Get info about current state
                    try:
                        ram = env.get_attr("unwrapped")[0].get_ram()
                        current_stage = int(ram[0x0058])
                        max_stage = max(max_stage, current_stage)

                        # Track X position progress
                        x_pos = ram[0x0094]
                        if current_stage not in stage_progress:
                            stage_progress[current_stage] = 0

                        # Update progress based on stage direction
                        if current_stage in [1, 3, 5]:  # Left stages
                            progress = 255 - x_pos
                        else:  # Right stages
                            progress = x_pos

                        stage_progress[current_stage] = max(
                            stage_progress[current_stage], progress
                        )
                    except:
                        pass

                    total_reward += reward[0]

                # Calculate progress score
                progress_values = list(stage_progress.values())
                avg_progress = (
                    sum(progress_values) / len(progress_values)
                    if progress_values
                    else 0
                )
                norm_progress = avg_progress / 255.0

                # Store episode results
                episode_rewards.append(total_reward)
                episode_stages.append(max_stage)
                episode_progress.append(norm_progress)

                print(
                    f"  Episode {episode+1}: Reward={total_reward:.2f}, Max Stage={max_stage}, Progress={norm_progress:.2f}"
                )

            # Calculate average stats
            avg_reward = sum(episode_rewards) / len(episode_rewards)
            avg_stage = sum(episode_stages) / len(episode_stages)
            avg_progress = sum(episode_progress) / len(episode_progress)

            # Calculate weighted score
            weighted_score = (
                0.6 * avg_stage + 0.3 * avg_progress + 0.1 * min(1.0, avg_reward / 1000)
            )

            # Add to results
            results.append(
                {
                    "model": model_name,
                    "avg_reward": avg_reward,
                    "avg_stage": avg_stage,
                    "avg_progress": avg_progress,
                    "weighted_score": weighted_score,
                }
            )

            print(f"  Average over {episodes} episodes:")
            print(f"    Reward: {avg_reward:.2f}")
            print(f"    Max Stage: {avg_stage:.2f}")
            print(f"    Progress: {avg_progress:.2f}")
            print(f"    Weighted Score: {weighted_score:.4f}")

        except Exception as e:
            print(f"  Error evaluating model {model_name}: {e}")

    # Close environment
    env.close()

    # Create summary dataframe
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("weighted_score", ascending=False)

        print("\n=== MODEL EVALUATION RESULTS ===")
        print(df.to_string(index=False))

        # Save results to CSV
        results_path = os.path.join(model_dir, "model_evaluation_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")

        # Identify best model
        best_model = df.iloc[0]
        print(f"\nBest model: {best_model['model']}")
        print(f"  Weighted Score: {best_model['weighted_score']:.4f}")
        print(f"  Avg Stage: {best_model['avg_stage']:.2f}")
        print(f"  Avg Progress: {best_model['avg_progress']:.2f}")
        print(f"  Avg Reward: {best_model['avg_reward']:.2f}")
    else:
        print("No successful model evaluations to report.")


def main():
    parser = argparse.ArgumentParser(description="Train or play Kung Fu Master with AI")
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play the game with trained agent"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate multiple saved models"
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=3,
        help="Number of episodes to play for each model during evaluation",
    )
    # New argument for controlling action forcing during play
    parser.add_argument(
        "--force_actions",
        action="store_true",
        help="Force action variety during play mode",
    )

    args = parser.parse_args()

    # Special mode for evaluating all saved models
    if args.eval:
        evaluate_models(args.eval_episodes)
        return

    # Create the environment, specifying if we're in play mode
    env = make_kungfu_env(num_envs=args.num_envs, is_play_mode=args.play)

    # Create or load the model
    model = create_model(env, resume=args.resume)

    if args.play:
        # Tell the user about the force_actions flag if they didn't use it
        if not args.force_actions:
            print("\nTIP: You can use --force_actions to improve agent behavior.")
            print("     This will make the agent use more varied actions and")
            print("     follow correct stage directions.\n")

        # Play the game with the trained agent
        play_game(env, model, episodes=5)
    else:
        # Train the model
        train_model(model, args.timesteps)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Make sure required libraries are installed
    required_packages = ["tqdm", "pandas"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess

            subprocess.check_call(["pip", "install", package])
            print(f"{package} installed successfully.")

    # Make sure cuda is enabled by default if available
    if torch.cuda.is_available():
        print("CUDA is available! Training will use GPU.")
    else:
        print("CUDA not available. Training will use CPU.")

    # Add version info in logs
    import gymnasium as gym

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Gymnasium version: {gym.__version__}")
    print(f"Stable-Baselines3 version: {stable_baselines3.__version__}")

    main()
