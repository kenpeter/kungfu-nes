import os
import argparse
import numpy as np
import retro
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

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


def train_model(model, timesteps):
    """Train the model with best model saving based on progress and performance"""
    print(f"Training for {timesteps} timesteps...")

    # Define callbacks for progress bar and best model saving
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    from stable_baselines3.common.vec_env import VecMonitor
    from tqdm import tqdm
    import numpy as np

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

    class KungFuBestModelCallback(BaseCallback):
        """Custom callback to save best model based on game progress and reward"""

        def __init__(self, check_freq=10000, best_model_save_path=None, verbose=1):
            super(KungFuBestModelCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.best_model_save_path = best_model_save_path

            # Initialize best metrics tracking
            self.best_reward = -np.inf
            self.best_progress = -np.inf
            self.best_weighted_score = -np.inf

            # Track metrics from environment
            self.cum_reward = 0
            self.max_stage = 1
            self.best_x_progress_by_stage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

            # Track action distribution
            self.action_counts = {i: 0 for i in range(len(KUNGFU_ACTIONS))}
            self.total_actions = 0
            self.last_log_step = 0
            self.action_log_freq = 5000  # Log action percentages every 5000 steps

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

            # Log rewards and progress
            if (
                len(self.model.ep_info_buffer) > 0
                and len(self.model.ep_info_buffer[-1]) > 0
            ):
                # Get info from latest step
                for env_idx in range(self.training_env.num_envs):
                    try:
                        info = self.locals["infos"][env_idx]

                        # Track cumulative reward
                        if "shaped_reward" in info:
                            self.cum_reward += info["shaped_reward"]

                        # Track stage progress
                        if "current_stage" in info:
                            stage = info["current_stage"]
                            self.max_stage = max(self.max_stage, stage)

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

                                self.best_x_progress_by_stage[stage] = max(
                                    self.best_x_progress_by_stage.get(stage, 0),
                                    progress,
                                )
                    except (KeyError, IndexError):
                        pass

            # Check if it's time to evaluate model and save if better
            if self.n_calls % self.check_freq == 0:
                # Calculate current weighted score
                # Stage progress (60%), X position (30%), reward (10%)

                # Get normalized X progress score (average across stages encountered)
                x_progress_scores = []
                for stage in range(1, self.max_stage + 1):
                    if stage in self.best_x_progress_by_stage:
                        # Normalize to 0-1 range
                        if stage in [1, 3, 5]:  # Left stages
                            norm_progress = self.best_x_progress_by_stage[stage] / 255
                        else:  # Right stages
                            norm_progress = self.best_x_progress_by_stage[stage] / 255
                        x_progress_scores.append(norm_progress)

                avg_x_progress = (
                    sum(x_progress_scores) / len(x_progress_scores)
                    if x_progress_scores
                    else 0
                )

                # Calculate weighted score
                weighted_score = (
                    0.6 * self.max_stage  # Stage progress (max weight)
                    + 0.3 * avg_x_progress  # X position progress
                    + 0.1
                    * min(1.0, self.cum_reward / 1000)  # Normalized reward (cap at 1.0)
                )

                # Check if this is the best model
                if weighted_score > self.best_weighted_score:
                    self.best_weighted_score = weighted_score
                    self.best_progress = self.max_stage
                    self.best_reward = self.cum_reward

                    # Save the best model
                    if self.best_model_save_path is not None:
                        path = os.path.join(self.best_model_save_path, "best_model")
                        self.model.save(path)
                        if self.verbose > 0:
                            print(f"Saving new best model to {path}.zip")
                            print(
                                f"New best metrics: Stage={self.max_stage}, "
                                f"X-Progress={avg_x_progress:.2f}, "
                                f"Reward={self.cum_reward:.2f}"
                            )

                    # Log action percentages when saving best model
                    self._log_action_percentages()

                # Reset cumulative metrics for next evaluation window
                self.cum_reward = 0

            return True

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

    # Create directory for best model
    best_model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(best_model_dir, exist_ok=True)

    # Create callbacks
    best_model_callback = KungFuBestModelCallback(
        check_freq=10000, best_model_save_path=best_model_dir, verbose=1
    )

    progress_callback = ProgressBarCallback(timesteps)

    # Train the model with callbacks
    model.learn(
        total_timesteps=timesteps, callback=[progress_callback, best_model_callback]
    )

    # Copy best model to final model path or save final model
    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        import shutil

        shutil.copy(best_model_path, MODEL_PATH)
        print(f"Best model saved to {MODEL_PATH}")
    else:
        # If no best model was saved, save the final model
        model.save(MODEL_PATH)
        print(f"Final model saved to {MODEL_PATH}")


def play_game(env, model, episodes=5):
    """Use the trained model to play the game with rendering"""
    obs = env.reset()
    print(f"Observation shape during play: {obs.shape}")  # Should show stacked frames

    from tqdm import tqdm

    for episode in range(episodes):
        done = [False]  # Initialize as list for vectorized env
        total_reward = 0
        step = 0

        # Track the current stage for directional logic
        current_stage = 1  # Default to stage 1 at start

        # Create progress bar for this episode
        pbar = tqdm(desc=f"Episode {episode+1}", leave=True)

        while not any(done):
            # Get the model's predicted action
            action, _ = model.predict(obs, deterministic=True)

            # Apply stage-specific logic to prioritize correct directional movement
            info = env.get_attr("unwrapped")[0]  # Get wrapped env info
            if hasattr(info, "get_ram"):
                try:
                    # Try to get current stage from RAM
                    ram = info.get_ram()
                    current_stage = int(ram[0x0058])

                    # Get stage direction
                    stage_direction = -1 if current_stage in [1, 3, 5] else 1

                    # Override with directional movement if no enemy is close
                    # This prioritizes correct stage movement
                    if stage_direction < 0:  # Should move left
                        # Check if the model wants to go right, if so, override to left
                        if action[0] == 7:  # RIGHT action
                            print("Overriding RIGHT to LEFT for stage", current_stage)
                            action[0] = 6  # LEFT action
                    elif stage_direction > 0:  # Should move right
                        # Check if the model wants to go left, if so, override to right
                        if action[0] == 6:  # LEFT action
                            print("Overriding LEFT to RIGHT for stage", current_stage)
                            action[0] = 7  # RIGHT action
                except Exception as e:
                    print(f"Error reading stage info: {e}")

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
                print(
                    f"\nEpisode {episode+1} finished with total reward: {total_reward:.2f}"
                )
                pbar.close()
                obs = env.reset()
                break


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

    args = parser.parse_args()

    # Create the environment, specifying if we're in play mode
    env = make_kungfu_env(num_envs=args.num_envs, is_play_mode=args.play)

    # Create or load the model
    model = create_model(env, resume=args.resume)

    if args.play:
        # Play the game with the trained agent
        play_game(env, model)
    else:
        # Train the model
        train_model(model, args.timesteps)

    # Close the environment
    env.close()


if __name__ == "__main__":
    # Make sure required libraries are installed
    try:
        import tqdm
    except ImportError:
        print("Installing tqdm for progress bars...")
        import subprocess

        subprocess.check_call(["pip", "install", "tqdm"])
        print("tqdm installed successfully.")

    # Make sure cuda is enabled by default if available
    if torch.cuda.is_available():
        print("CUDA is available! Training will use GPU.")
    else:
        print("CUDA not available. Training will use CPU.")

    main()
