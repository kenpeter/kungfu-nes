from collections import deque
import gymnasium as gym
import numpy as np
from gymnasium import spaces, Wrapper
import cv2
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import retro
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
KUNGFU_MAX_ENEMIES = 5  # Increased by 1 to include boss as special enemy
MAX_PROJECTILES = 2
N_STACK = 4  # Number of frames to stack for input - KEEP THIS AT 4

# Define actions
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

# Define observation space with consistent dimensions
KUNGFU_OBSERVATION_SPACE = spaces.Dict(
    {
        # 160, 160, 3*n_stack - CHANNEL LAST FORMAT
        "viewport": spaces.Box(
            low=0, high=255, shape=(160, 160, 3 * N_STACK), dtype=np.uint8
        ),
        "projectile_vectors": spaces.Box(
            low=-255, high=255, shape=(MAX_PROJECTILES * 4,), dtype=np.float32
        ),
        "enemy_vectors": spaces.Box(
            low=-255, high=255, shape=(KUNGFU_MAX_ENEMIES * 4,), dtype=np.float32
        ),
    }
)


class KungFuWrapper(Wrapper):
    def __init__(self, env, n_stack=N_STACK):
        # super
        super().__init__(env)
        # reset env
        result = env.reset()
        # obs there?
        if isinstance(result, tuple):
            obs, _ = result
        else:
            obs = result

        # true height
        self.true_height, self.true_width = obs.shape[:2]
        # view port size
        self.viewport_size = (160, 160)  # Consistent viewport size

        self.actions = KUNGFU_ACTIONS
        self.action_names = KUNGFU_ACTION_NAMES
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = KUNGFU_OBSERVATION_SPACE

        self.furthest_left_position = float("inf")

        # State tracking
        self.last_hp = 0
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.total_steps = 0
        self.last_projectile_distances = [float("inf")] * MAX_PROJECTILES
        self.survival_reward_total = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.player_x = 0
        self.last_player_x = 0
        self.timer = 0
        self.last_timer = 0
        self.n_stack = n_stack

        # frame stack here
        self.frame_stack = deque(maxlen=n_stack)
        logger.info(
            f"KungFu wrapper initialized with viewport size {self.viewport_size}, n_stack={n_stack}"
        )

    def reset(self, seed=None, options=None, **kwargs):
        # Reset the environment
        result = self.env.reset(seed=seed, options=options, **kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Press start to begin the game
        for _ in range(5):  # Press start multiple times to ensure it registers
            obs, _, done, _, _ = self.env.step(self.actions[3])  # START action
            if done:  # If somehow pressing start causes a done signal
                obs, info = self.env.reset()

        # Reset state tracking variables
        self.furthest_left_position = float("inf")
        ram = self.env.get_ram()
        self.last_hp = float(ram[0x04A6])
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.total_steps = 0
        self.last_projectile_distances = [float("inf")] * MAX_PROJECTILES
        self.survival_reward_total = 0
        self.reward_mean = 0
        self.reward_std = 1
        self.player_x = float(ram[0x0094])
        self.last_player_x = self.player_x
        self.timer = float(ram[0x0391])
        self.last_timer = self.timer

        # Initialize frame-based detection tracking
        self.last_enemy_vectors = np.zeros(KUNGFU_MAX_ENEMIES * 4)

        # Resize the observation before adding to frame stack
        resized_obs = cv2.resize(obs, (160, 160), interpolation=cv2.INTER_AREA)

        # Clear and initialize frame stack
        self.frame_stack.clear()
        for _ in range(self.n_stack):
            self.frame_stack.append(resized_obs)

        # Get observation with all the detected objects
        initial_obs = self._get_obs(obs)

        return initial_obs, info

    def step(self, action):
        # action count will be used as percentage
        self.total_steps += 1
        self.action_counts[action] += 1

        # new gym 5 and old gym 4
        result = self.env.step(self.actions[action])
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
            terminated = done
            truncated = False

        # Get the observation first for frame-based detection
        current_obs = self._get_obs(obs)

        # Get RAM data (only what's still needed)
        ram = self.env.get_ram()
        hp = float(ram[0x04A6])
        self.player_x = float(ram[0x0094])
        self.timer = float(ram[0x0391])

        # --- REWARD CALCULATION ---
        reward = 0

        # HP-based reward (still from RAM for accuracy)
        hp_change_rate = (hp - self.last_hp) / 255.0
        if hp_change_rate < 0:
            reward += (hp_change_rate**2) * 50  # Squared penalty for taking damage
        else:
            reward += hp_change_rate * 5  # Small reward for healing

        # Game over penalty
        if done:
            reward -= 50

        # --- PROGRESSION REWARD ---
        x_change = (
            self.last_player_x - self.player_x
        )  # In Kung Fu, moving left is progress
        progression_reward = x_change * 0.2  # Basic reward for movement

        if action in [6]:  # left
            progression_reward += 0.5
        reward += progression_reward

        # --- TIMER PENALTY ---
        timer_change = self.last_timer - self.timer
        if timer_change > 0:
            reward -= timer_change * 0.5

        # --- PROJECTILE REWARD ---
        projectile_info = current_obs["projectile_vectors"]
        projectile_distances = [
            projectile_info[i] for i in range(0, len(projectile_info), 4)
        ]
        dodge_reward = 0
        for i, (curr_dist, last_dist) in enumerate(
            zip(projectile_distances, self.last_projectile_distances)
        ):
            if last_dist < 20 and curr_dist > last_dist:
                if action in [4, 5]:
                    dodge_reward += 3
        reward += dodge_reward
        self.last_projectile_distances = projectile_distances

        # --- ENEMY/BOSS REWARD (combined) ---
        enemy_vectors = current_obs["enemy_vectors"]
        # Track enemies between frames
        self.last_enemy_vectors = getattr(
            self, "last_enemy_vectors", np.zeros_like(enemy_vectors)
        )

        # Process rewards for all enemies including the boss
        combat_reward = 0

        # Process each enemy sequentially
        for i in range(0, len(enemy_vectors), 4):
            last_enemy_present = np.sum(np.abs(self.last_enemy_vectors[i : i + 4])) > 5
            curr_enemy_present = np.sum(np.abs(enemy_vectors[i : i + 4])) > 5

            # Get the enemy state value (4th value indicates if it's a boss)
            enemy_state = enemy_vectors[i + 3]
            is_boss = enemy_state > 0.9  # If state > 0.9, it's a boss

            # Calculate position and size
            enemy_dx = enemy_vectors[i]
            enemy_dy = enemy_vectors[i + 1]
            enemy_width = enemy_vectors[i + 2]

            # Check for enemy disappearance (defeated)
            if last_enemy_present and not curr_enemy_present:
                # Enemy defeated
                if action in [1, 8, 9, 11, 12]:  # Attack actions
                    if is_boss:
                        combat_reward += 30.0  # Big reward for hitting boss
                    else:
                        combat_reward += 10.0  # Regular reward for enemies

            # Check for position change (enemy hit but not defeated)
            elif last_enemy_present and curr_enemy_present:
                last_pos = np.array(
                    [self.last_enemy_vectors[i], self.last_enemy_vectors[i + 1]]
                )
                curr_pos = np.array([enemy_vectors[i], enemy_vectors[i + 1]])
                position_change = np.linalg.norm(curr_pos - last_pos)

                if position_change > 10 and action in [1, 8, 9, 11, 12]:
                    if is_boss:
                        combat_reward += (
                            15.0  # Reward for hitting but not defeating boss
                        )
                    else:
                        combat_reward += (
                            5.0  # Reward for hitting but not defeating enemy
                        )

            # Reward for good positioning - being close to enemies when attacking
            if curr_enemy_present:
                # If enemy is close
                if abs(enemy_dx) < 30 and abs(enemy_dy) < 20:
                    # Reward for attacking when enemies are in range
                    if action in [1, 8, 9, 11, 12]:  # Attack actions
                        if is_boss:
                            combat_reward += 3.0  # More reward for boss positioning
                        else:
                            combat_reward += 2.0  # Normal reward for enemy positioning

                # If boss is attacking (state > 1.0), reward defensive moves
                if is_boss and enemy_state > 1.0 and abs(enemy_dx) < 40:
                    if action in [4, 5, 6, 7]:  # Defensive moves
                        combat_reward += 4.0  # Reward for dodging boss attacks

        reward += combat_reward

        # --- ACTION-BASED REWARDS ---
        if hp_change_rate < 0 and action in [5, 6]:
            reward += 5  # Reward for moving away when damaged
        if action in [1, 8, 11, 12]:
            reward += 2.0  # Reward for attacking
        elif action in [9, 10]:
            reward -= 0.5  # Slight penalty for complex actions

        # --- ACTION ENTROPY ---
        action_entropy = -np.sum(
            (self.action_counts / (self.total_steps + 1e-6))
            * np.log(self.action_counts / (self.total_steps + 1e-6) + 1e-6)
        )
        reward += action_entropy * 0.1  # Encourage varied action use

        # --- SURVIVAL REWARD ---
        if not done and hp > 0:
            reward += 0.05
            self.survival_reward_total += 0.05

        # --- NORMALIZE REWARD ---
        self.reward_mean = 0.99 * self.reward_mean + 0.01 * reward
        self.reward_std = (
            0.99 * self.reward_std + 0.01 * (reward - self.reward_mean) ** 2
        )
        normalized_reward = (reward - self.reward_mean) / (
            np.sqrt(self.reward_std) + 1e-6
        )
        normalized_reward = np.clip(normalized_reward, -10, 10)

        # --- EXPLORATION REWARD ---
        if self.player_x < self.furthest_left_position:
            self.furthest_left_position = self.player_x
            progression_bonus = 5.0  # Big reward for reaching new territory
            reward += progression_bonus

        # Update state tracking variables
        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_player_x = self.player_x
        self.last_timer = self.timer
        self.last_enemy_vectors = enemy_vectors.copy()

        # Update info dict with frame-based detections
        info.update(
            {
                "hp": hp,
                "combat_reward": combat_reward,
                "enemy_vectors": enemy_vectors.tolist(),
                "action_percentages": self.action_counts / (self.total_steps + 1e-6),
                "action_names": self.action_names,
                "dodge_reward": dodge_reward,
                "survival_reward_total": self.survival_reward_total,
                "raw_reward": reward,
                "normalized_reward": normalized_reward,
                "progression_reward": progression_reward,
                "player_x": self.player_x,
                "timer": self.timer,
            }
        )

        return current_obs, normalized_reward, terminated, truncated, info

    def _get_obs(self, obs):
        # obs become 160x160 single frame
        frame = cv2.resize(obs, (160, 160), interpolation=cv2.INTER_AREA)

        # push single frame to frame stack
        self.frame_stack.append(frame)

        # Combine all frames along channels: (160, 160, 3*N)
        # Ensure we have enough frames
        if len(self.frame_stack) < self.n_stack:
            # If we don't have enough frames yet, duplicate the last one
            frames = list(self.frame_stack)
            while len(frames) < self.n_stack:
                frames.append(frames[-1])
        else:
            frames = list(self.frame_stack)

        stacked_frames = np.concatenate(frames, axis=2)

        # Check that stacked_frames has the correct shape
        expected_shape = (160, 160, 3 * self.n_stack)
        if stacked_frames.shape != expected_shape:
            logger.warning(
                f"Stacked frames shape {stacked_frames.shape} doesn't match expected {expected_shape}"
            )
            # Try to reshape
            try:
                stacked_frames = stacked_frames.reshape(expected_shape)
            except Exception as e:
                logger.error(f"Failed to reshape stacked frames: {e}")
                # Create empty array as fallback
                stacked_frames = np.zeros(expected_shape, dtype=np.uint8)

        # Get object detection data
        detected_objects = self._detect_objects()

        return {
            # frame stacks into viewport
            "viewport": stacked_frames.astype(np.uint8),
            # we detect projectile
            "projectile_vectors": np.array(
                detected_objects["projectiles"], dtype=np.float32
            ),
            # we detect enemy and boss
            "enemy_vectors": np.array(detected_objects["enemies"], dtype=np.float32),
        }

    def _detect_objects(self):
        """
        Detect projectiles and enemies (including boss) using the full frame stack.
        This allows for better temporal information and more robust detection.
        Uses both frame detection and RAM values for boss identification.
        Returns info about projectiles and enemies (with boss as a special enemy).
        """
        # Use frame stack to detect objects
        if len(self.frame_stack) < self.n_stack:
            return {
                "projectiles": [0] * (MAX_PROJECTILES * 4),
                "enemies": [0] * (KUNGFU_MAX_ENEMIES * 4),
            }

        # Get the frames from the stack
        frames = list(self.frame_stack)
        latest_frame = frames[-1]

        # Get player position from RAM (still need this as reference point)
        ram = self.env.get_ram()
        hero_x = int(ram[0x0094])
        hero_y = int(ram[0x0097])

        # Get boss information from RAM
        boss_pos_x = int(ram[0x0093])  # Boss X position
        boss_hp = int(ram[0x04A5])  # Boss HP
        boss_action = int(ram[0x004E])  # Boss action state
        boss_present = boss_hp > 0

        # We'll use frame differences between consecutive frames to detect motion
        moving_object_candidates = []

        # Process each pair of consecutive frames in the stack
        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            # Calculate frame difference to detect motion
            frame_diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(frame_diff, axis=2).astype(np.uint8)
            _, motion_mask = cv2.threshold(diff_sum, 20, 255, cv2.THRESH_BINARY)

            # Find contours of all moving objects
            contours, _ = cv2.findContours(
                motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process each contour
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 3:  # Filter out very small noise
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                obj_x = x + w // 2
                obj_y = y + h // 2

                # Add to candidates with frame index for temporal information
                moving_object_candidates.append((obj_x, obj_y, w, h, area, i))

        # If no candidates found, return zeros
        if not moving_object_candidates:
            return {
                "projectiles": [0] * (MAX_PROJECTILES * 4),
                "enemies": [0] * (KUNGFU_MAX_ENEMIES * 4),
            }

        # Sort candidates by area (descending)
        moving_object_candidates.sort(key=lambda x: -x[4])

        # --- DETECT PROJECTILES ---
        # Filter for small, fast-moving white objects
        projectile_candidates = []

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            # Calculate frame difference
            frame_diff = cv2.absdiff(curr_frame, prev_frame)
            diff_sum = np.sum(frame_diff, axis=2).astype(np.uint8)
            _, motion_mask = cv2.threshold(diff_sum, 20, 255, cv2.THRESH_BINARY)

            # Filter for white/bright objects (typical for projectiles)
            lower_white = np.array([180, 180, 180])
            upper_white = np.array([255, 255, 255])
            white_mask = cv2.inRange(curr_frame, lower_white, upper_white)

            # Combine masks to find moving white objects
            combined_mask = cv2.bitwise_and(motion_mask, white_mask)

            contours, _ = cv2.findContours(
                combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                # Filter by area for projectiles
                if 5 < area < 50:
                    x, y, w, h = cv2.boundingRect(contour)
                    proj_x = x + w // 2
                    proj_y = y + h // 2
                    projectile_candidates.append((proj_x, proj_y, i, area))

        # Process projectiles
        projectile_info = []
        processed_projectile_positions = set()

        for proj_x, proj_y, frame_idx, _ in projectile_candidates:
            # Create a position key to avoid duplicates
            pos_key = (proj_x // 5, proj_y // 5)

            if (
                pos_key not in processed_projectile_positions
                and len(projectile_info) < MAX_PROJECTILES * 4
            ):
                # Calculate distance relative to player
                distance = proj_x - hero_x

                # Estimate velocity
                velocity = 0
                for prev_x, prev_y, prev_frame_idx, _ in projectile_candidates:
                    prev_pos_key = (prev_x // 5, prev_y // 5)
                    if prev_pos_key == pos_key and prev_frame_idx < frame_idx:
                        velocity = (proj_x - prev_x) / (frame_idx - prev_frame_idx)
                        break

                projectile_info.extend([distance, proj_y, velocity, 0])
                processed_projectile_positions.add(pos_key)

        # Ensure exact size
        projectile_info = projectile_info[: MAX_PROJECTILES * 4]
        while len(projectile_info) < MAX_PROJECTILES * 4:
            projectile_info.append(0)

        # --- DETECT ENEMIES & BOSS COMBINED ---
        # Convert to HSV for better color filtering
        hsv_frame = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2HSV)

        # Detect all enemy-like objects (including regular enemies and boss)
        enemy_candidates = []

        # Filter for enemy colors (typically red/brown in Kung Fu)
        # Red color range in HSV (two ranges because red wraps around in HSV)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])

        # Blue color range for boss
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])

        # Combine color masks for all enemy types
        red_mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
        red_mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
        blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)

        enemy_color_mask = cv2.bitwise_or(
            cv2.bitwise_or(red_mask1, red_mask2), blue_mask
        )

        # Find contours of potential enemies
        enemy_contours, _ = cv2.findContours(
            enemy_color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Frame-based boss detection criteria
        boss_detected_by_frame = False
        boss_x, boss_y, boss_w, boss_h = 0, 0, 0, 0

        # First, prioritize RAM-based boss detection if available
        if boss_present:
            # Search for objects near the known boss position
            boss_screen_x = boss_pos_x  # Might need scaling depending on game

            # Look for large objects near RAM boss position
            best_match_distance = float("inf")
            best_boss_candidate = None

            for contour in enemy_contours:
                area = cv2.contourArea(contour)
                if area > 200:  # Boss is typically large
                    x, y, w, h = cv2.boundingRect(contour)
                    obj_center_x = x + w // 2

                    # Check if this object is close to the RAM boss position
                    distance_to_ram_boss = abs(obj_center_x - boss_screen_x)
                    if (
                        distance_to_ram_boss < best_match_distance
                        and distance_to_ram_boss < 30
                    ):
                        best_match_distance = distance_to_ram_boss
                        best_boss_candidate = (x, y, w, h, area)

            # If we found a good match for the boss
            if best_boss_candidate:
                x, y, w, h, area = best_boss_candidate
                # Skip if this is likely the player
                player_distance = abs((x + w // 2) - hero_x) + abs(
                    (y + h // 2) - hero_y
                )
                if player_distance > 20:  # Not the player
                    boss_detected_by_frame = True
                    boss_x, boss_y, boss_w, boss_h = x, y, w, h
                    # Add to candidates with special flag
                    enemy_candidates.append((x, y, w, h, area, True))

        # If RAM-based detection didn't work, try purely visual detection
        if not boss_detected_by_frame and boss_present:
            # Look for large objects at screen edges (typical for bosses)
            for contour in enemy_contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Boss is larger than regular enemies
                    x, y, w, h = cv2.boundingRect(contour)
                    # Skip if this is likely the player
                    player_distance = abs((x + w // 2) - hero_x) + abs(
                        (y + h // 2) - hero_y
                    )
                    if player_distance < 20:
                        continue

                    # Check if it's at the edge of screen - boss often is
                    is_at_edge = x < 20 or x + w > 140

                    if is_at_edge:
                        boss_detected_by_frame = True
                        boss_x, boss_y, boss_w, boss_h = x, y, w, h
                        enemy_candidates.append((x, y, w, h, area, True))
                        break

        # Process regular enemies
        for contour in enemy_contours:
            area = cv2.contourArea(contour)
            # Accept enemies of medium size
            if 50 < area < 300:
                x, y, w, h = cv2.boundingRect(contour)
                # Skip if this is likely the player
                player_distance = abs((x + w // 2) - hero_x) + abs(
                    (y + h // 2) - hero_y
                )
                if player_distance < 20:
                    continue

                # Skip if this is too close to the detected boss (likely part of boss animation)
                if boss_detected_by_frame:
                    boss_overlap = (
                        x < boss_x + boss_w
                        and x + w > boss_x
                        and y < boss_y + boss_h
                        and y + h > boss_y
                    )
                    if boss_overlap:
                        continue

                # Regular enemy
                enemy_candidates.append((x, y, w, h, area, False))

        # Also consider moving objects for enemy detection
        for obj_x, obj_y, w, h, area, _ in moving_object_candidates:
            if 50 < area < 300:  # Regular enemy size
                # Skip if this is likely the player
                player_distance = abs(obj_x - hero_x) + abs(obj_y - hero_y)
                if player_distance < 20:
                    continue

                # Check if this overlaps with anything we've already detected
                is_duplicate = False
                for ex, ey, ew, eh, _, _ in enemy_candidates:
                    if (abs(obj_x - (ex + ew // 2)) < ew // 2) and (
                        abs(obj_y - (ey + eh // 2)) < eh // 2
                    ):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    enemy_candidates.append(
                        (obj_x - w // 2, obj_y - h // 2, w, h, area, False)
                    )

        # Sort enemies by type and then by distance to player
        # Put boss first if present, then sort others by distance to player
        enemy_candidates.sort(
            key=lambda e: (
                -int(e[5]),
                abs((e[0] + e[2] // 2) - hero_x) + abs((e[1] + e[3] // 2) - hero_y),
            )
        )

        # Process enemy info - ensure boss is first if present
        enemy_info = []

        # Take up to KUNGFU_MAX_ENEMIES candidates
        for i, (ex, ey, ew, eh, area, is_boss) in enumerate(
            enemy_candidates[:KUNGFU_MAX_ENEMIES]
        ):
            # Calculate enemy center
            enemy_center_x = ex + ew // 2
            enemy_center_y = ey + eh // 2

            # Calculate distance and direction vectors relative to player
            dx = enemy_center_x - hero_x
            dy = enemy_center_y - hero_y

            # Special fourth value for boss flag (1.0) or regular enemy size/state
            if is_boss:
                # For boss, use a marker value to indicate boss status
                # Use RAM-based boss action if available
                if boss_present:
                    # Incorporate RAM boss action with scaling factor
                    enemy_state = 1.0 + (float(boss_action) * 0.1)
                else:
                    # Fallback to frame-based boss action detection
                    boss_action_detected = 0
                    # Look for motion within boss area
                    for obj_x, obj_y, _, _, obj_area, _ in moving_object_candidates:
                        if (
                            (ex <= obj_x <= ex + ew)
                            and (ey <= obj_y <= ey + eh)
                            and obj_area < ew * eh * 0.5
                        ):
                            boss_action_detected = 1
                            break
                    enemy_state = 1.0 + boss_action_detected * 0.1
            else:
                # For regular enemies, use normalized size as proxy for action/state
                enemy_state = min(
                    0.9, ew * eh / 400
                )  # Cap at 0.9 to distinguish from boss

            enemy_info.extend([dx, dy, int(ew), enemy_state])

        # Ensure exact size
        enemy_info = enemy_info[: KUNGFU_MAX_ENEMIES * 4]
        while len(enemy_info) < KUNGFU_MAX_ENEMIES * 4:
            enemy_info.append(0)

        # Return all detected information
        return {"projectiles": projectile_info, "enemies": enemy_info}

    def _detect_projectiles(self):
        """
        Backward compatibility method that calls _detect_objects and returns only projectile data
        """
        return self._detect_objects()["projectiles"]


class SimpleCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256):
        # super with obs and feature dim
        super().__init__(observation_space, features_dim)
        # Get viewport shape from observation space
        viewport_shape = observation_space["viewport"].shape

        # CRITICAL: Hard-code the correct values based on N_STACK=4
        channels = 3 * N_STACK  # 12 channels for 4 RGB frames
        height = 160
        width = 160

        logger.info(
            f"SimpleCNN initialized with n_stack={N_STACK}, viewport shape={viewport_shape}"
        )

        # CNN layers - explicitly use the correct number of input channels
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output size of the CNN using a dummy input
        with torch.no_grad():
            # Always use consistent shape for sample calculation
            sample_input = torch.zeros((1, channels, height, width))
            n_flatten = self.cnn(sample_input).shape[1]

        # Process all vector data along with CNN features
        proj_vectors_size = observation_space["projectile_vectors"].shape[0]
        enemy_vectors_size = observation_space["enemy_vectors"].shape[0]

        total_vector_size = proj_vectors_size + enemy_vectors_size

        # Encoder for vector data
        self.vector_encoder = nn.Sequential(
            nn.Linear(total_vector_size, 128),
            nn.ReLU(),
        )

        # Combined network
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 128, 512),
            nn.ReLU(),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

        logger.info(
            f"SimpleCNN architecture: n_flatten={n_flatten}, total_vector_size={total_vector_size}, features_dim={features_dim}"
        )

    def forward(self, observations):
        viewport = observations["viewport"]  # Initial shape varies

        # Handle VecTransposeImage inconsistency
        # The goal is to get to [batch, 12, 160, 160] format

        # Debug shape
        if torch.rand(1).item() < 0.01:  # Print occasionally to avoid spam
            logger.info(f"Original viewport shape: {viewport.shape}")

        # Calculate input size to determine correct reshaping
        batch_size = viewport.shape[0]
        input_size = viewport.numel()  # Total number of elements
        expected_elements = batch_size * 12 * 160 * 160

        # Check if we have an unexpected amount of data
        if input_size != expected_elements:
            logger.warning(
                f"Input size {input_size} doesn't match expected size {expected_elements}"
            )
            # If total size is 4915200, which is 4 * 48 * 160 * 160
            if input_size == batch_size * 48 * 160 * 160:
                logger.info("Detected 48 channels, extracting first 12 channels")
                # Extract only the first 12 channels (4 RGB frames)
                if viewport.shape[1] == 48:
                    viewport = viewport[:, :12, :, :]
                elif viewport.shape[2] == 48:
                    viewport = viewport[:, :, :12, :]
                else:
                    # Try to reshape and then extract
                    try:
                        viewport = viewport.reshape(batch_size, 48, 160, 160)[
                            :, :12, :, :
                        ]
                        logger.info(f"Reshaped viewport to {viewport.shape}")
                    except Exception as e:
                        logger.error(f"Failed to reshape: {e}")
                        # Last resort - create a new tensor with zeros
                        viewport = torch.zeros(
                            (batch_size, 12, 160, 160),
                            device=viewport.device,
                            dtype=viewport.dtype,
                        )
        else:
            # Handle normal reshaping for expected sized tensors
            if len(viewport.shape) == 4:
                if viewport.shape[1] == 160 and viewport.shape[2] == 12:
                    # Shape is [batch, H=160, C=12, W=160] - this is wrong format
                    viewport = viewport.permute(0, 2, 1, 3)
                elif (
                    viewport.shape[1] == 160
                    and viewport.shape[2] == 160
                    and viewport.shape[3] == 12
                ):
                    # Shape is [batch, H=160, W=160, C=12] - channel last format
                    viewport = viewport.permute(0, 3, 1, 2)
                elif viewport.shape[1] == 12:
                    # Shape is already [batch, C=12, H, W]
                    pass
                else:
                    # Known shape but wrong format - try to reshape
                    logger.warning(
                        f"Reshaping from {viewport.shape} to [batch, 12, 160, 160]"
                    )
                    try:
                        viewport = viewport.reshape(batch_size, 12, 160, 160)
                    except Exception as e:
                        logger.error(f"Reshape failed: {e}")
                        # Emergency fallback
                        viewport = torch.zeros(
                            (batch_size, 12, 160, 160),
                            device=viewport.device,
                            dtype=viewport.dtype,
                        )

        # Final check to ensure we have the right shape
        if (
            viewport.shape[1] != 12
            or viewport.shape[2] != 160
            or viewport.shape[3] != 160
        ):
            logger.error(
                f"Failed to reshape viewport to expected format: {viewport.shape}"
            )
            # Last resort - create a new tensor with zeros
            viewport = torch.zeros(
                (batch_size, 12, 160, 160), device=viewport.device, dtype=viewport.dtype
            )

        # Normalize and process
        viewport = viewport.float() / 255.0
        cnn_features = self.cnn(viewport)  # Pass through Conv layers

        # Combine all vector data
        proj_vectors = observations["projectile_vectors"].float()
        enemy_vectors = observations["enemy_vectors"].float()

        # Concatenate all vector data
        all_vectors = torch.cat([proj_vectors, enemy_vectors], dim=1)

        # Encode vector data
        encoded_vectors = self.vector_encoder(all_vectors)

        # Combine CNN features with encoded vector data
        combined = torch.cat([cnn_features, encoded_vectors], dim=1)

        return self.linear(combined)


def make_env():
    """
    Create and wrap the KungFu environment
    """
    env = retro.make(
        "KungFu-Nes", use_restricted_actions=retro.Actions.ALL, render_mode="rgb_array"
    )
    env = Monitor(KungFuWrapper(env))
    return env
