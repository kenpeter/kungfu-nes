import numpy as np


class ProjectileDetector:
    """
    A class to detect projectiles in frame stacks by analyzing pixel changes
    between consecutive frames. Optimized for Kung Fu Master's projectiles.
    """

    def __init__(
        self,
        projectile_color_ranges=None,
        movement_threshold=10,
        min_projectile_size=4,
        max_projectile_size=20,
    ):
        """
        Initialize the projectile detector with color ranges and thresholds.

        Args:
            projectile_color_ranges: RGB ranges for projectile colors
            movement_threshold: Minimum pixel difference to consider as movement
            min_projectile_size: Minimum size of a projectile in pixels
            max_projectile_size: Maximum size of a projectile in pixels
        """
        # Default color ranges for common projectiles in Kung Fu Master
        # (knives, bottles, etc. - mostly white/light colored)
        self.projectile_color_ranges = projectile_color_ranges or [
            # Light colors (knives, etc.) - RGB ranges
            {"r": (180, 255), "g": (180, 255), "b": (180, 255)},
            # Yellow/orange fireballs
            {"r": (200, 255), "g": (140, 220), "b": (20, 100)},
        ]

        self.movement_threshold = movement_threshold
        self.min_projectile_size = min_projectile_size
        self.max_projectile_size = max_projectile_size

        # Track frame history for smoother detection
        self.previous_projectiles = []

    def detect_projectiles(self, frame_stack):
        """
        Detect projectiles in a stack of frames.

        Args:
            frame_stack: Numpy array of stacked frames [stack_size, height, width, channels]
                         or [stack_size, height, width] for grayscale

        Returns:
            List of detected projectiles with their positions and velocities
            Format: [{'x': x, 'y': y, 'vx': vx, 'vy': vy, 'size': size}, ...]
        """
        if frame_stack.ndim == 4:  # RGB frames
            # Use only the last two frames for movement detection
            current_frame = frame_stack[-1]
            previous_frame = frame_stack[-2]
        else:
            # Handle grayscale frames if needed
            current_frame = frame_stack[-1]
            previous_frame = frame_stack[-2]

        # Calculate absolute difference between frames
        frame_diff = np.abs(
            current_frame.astype(np.int16) - previous_frame.astype(np.int16)
        )

        # For RGB, convert to grayscale for movement detection
        if frame_diff.ndim == 3:
            diff_gray = np.mean(frame_diff, axis=2).astype(np.uint8)
        else:
            diff_gray = frame_diff.astype(np.uint8)

        # Threshold to find significant movement
        movement_mask = diff_gray > self.movement_threshold

        # Find connected components (potential projectiles)
        labeled_mask, num_objects = self._connected_components(movement_mask)

        projectiles = []
        for i in range(1, num_objects + 1):
            # Get object pixels
            obj_pixels = np.where(labeled_mask == i)

            # Skip if too few or too many pixels (not likely a projectile)
            if (
                len(obj_pixels[0]) < self.min_projectile_size
                or len(obj_pixels[0]) > self.max_projectile_size
            ):
                continue

            # Calculate centroid
            y = np.mean(obj_pixels[0])
            x = np.mean(obj_pixels[1])

            # Check if the object matches projectile color profiles
            if self._check_projectile_color(current_frame, obj_pixels):
                # Estimate velocity by comparing with previous frames
                vx, vy = self._estimate_velocity(x, y)

                # Add to projectiles list
                projectiles.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "vx": vx,
                        "vy": vy,
                        "size": len(obj_pixels[0]),
                    }
                )

        # Update previous projectiles for velocity tracking
        self.previous_projectiles = projectiles

        return projectiles

    def _connected_components(self, binary_image):
        """
        Simple implementation of connected components labeling
        Returns labeled image and number of components
        """
        h, w = binary_image.shape
        labels = np.zeros_like(binary_image, dtype=np.int32)
        label_count = 0

        # First pass - assign temporary labels and store equivalences
        for y in range(h):
            for x in range(w):
                if binary_image[y, x] == 0:
                    continue

                # Check neighbors (4-connectivity)
                neighbors = []
                if y > 0 and labels[y - 1, x] > 0:
                    neighbors.append(labels[y - 1, x])
                if x > 0 and labels[y, x - 1] > 0:
                    neighbors.append(labels[y, x - 1])

                if not neighbors:
                    # New label
                    label_count += 1
                    labels[y, x] = label_count
                else:
                    # Assign smallest neighbor label
                    labels[y, x] = min(neighbors)

        return labels, label_count

    def _check_projectile_color(self, frame, obj_pixels):
        """
        Check if the object matches the color profile of projectiles
        Returns True if the object likely represents a projectile
        """
        # Extract RGB values for the object pixels
        if frame.ndim == 3:  # RGB
            r = frame[obj_pixels[0], obj_pixels[1], 0]
            g = frame[obj_pixels[0], obj_pixels[1], 1]
            b = frame[obj_pixels[0], obj_pixels[1], 2]

            # Check against color ranges
            for color_range in self.projectile_color_ranges:
                r_in_range = (
                    np.mean((r >= color_range["r"][0]) & (r <= color_range["r"][1]))
                    > 0.6
                )
                g_in_range = (
                    np.mean((g >= color_range["g"][0]) & (g <= color_range["g"][1]))
                    > 0.6
                )
                b_in_range = (
                    np.mean((b >= color_range["b"][0]) & (b <= color_range["b"][1]))
                    > 0.6
                )

                if r_in_range and g_in_range and b_in_range:
                    return True

        # Default to movement-based detection for grayscale or if no color match
        return True

    def _estimate_velocity(self, x, y):
        """
        Estimate velocity by finding the closest previous projectile
        Returns estimated x and y velocity components
        """
        if not self.previous_projectiles:
            return 0, 0

        # Find closest previous projectile
        min_dist = float("inf")
        closest_proj = None

        for proj in self.previous_projectiles:
            dist = ((proj["x"] - x) ** 2 + (proj["y"] - y) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_proj = proj

        # If we found a close match (likely the same projectile)
        if min_dist < 15:  # Threshold for considering it the same projectile
            vx = x - closest_proj["x"]
            vy = y - closest_proj["y"]
            return vx, vy

        return 0, 0

    def predict_collision(
        self, projectiles, player_x, player_y, player_width=20, player_height=40
    ):
        """
        Predict if any projectiles will collide with the player in the near future

        Args:
            projectiles: List of detected projectiles
            player_x: Player's x position
            player_y: Player's y position
            player_width: Player width in pixels
            player_height: Player height in pixels

        Returns:
            collision_time: Estimated frames until collision (-1 if no collision)
            projectile_height: Height of the incoming projectile (for jump/crouch decision)
        """
        collision_time = -1
        projectile_height = 0

        for proj in projectiles:
            # Skip projectiles moving away from the player
            if (proj["x"] < player_x and proj["vx"] < 0) or (
                proj["x"] > player_x and proj["vx"] > 0
            ):
                continue

            # Calculate time to x-collision
            if abs(proj["vx"]) > 0.5:  # If projectile has significant x velocity
                time_to_x = abs((proj["x"] - player_x) / proj["vx"])

                # Predict y position at collision time
                future_y = proj["y"] + proj["vy"] * time_to_x

                # Check if y position will be within player bounds
                if (
                    future_y >= player_y - player_height / 2
                    and future_y <= player_y + player_height / 2
                ):

                    # If this is the closest projectile on a collision course
                    if collision_time == -1 or time_to_x < collision_time:
                        collision_time = time_to_x
                        projectile_height = future_y

        return collision_time, projectile_height


def enhance_observation_with_projectiles(obs, projectile_detector, player_pos):
    """
    Enhance the observation with projectile detection information

    Args:
        obs: Original observation (frame stack)
        projectile_detector: ProjectileDetector instance
        player_pos: (x, y) position of the player

    Returns:
        Dictionary with projectile information:
        {
            'projectiles': List of detected projectiles,
            'collision_time': Estimated frames until collision,
            'projectile_height': Height of incoming projectile,
            'recommended_action': Recommended defensive action (4=jump, 5=crouch, 0=none)
        }
    """
    # Process the observation to get frames
    # Assuming obs is a stacked frame array [stack_size, height, width, channels]

    # Detect projectiles
    projectiles = projectile_detector.detect_projectiles(obs)

    # Predict potential collisions
    collision_time, projectile_height = projectile_detector.predict_collision(
        projectiles, player_pos[0], player_pos[1]
    )

    # Determine recommended action based on projectile height
    recommended_action = 0  # Default: no defensive action

    if (
        collision_time > 0 and collision_time < 10
    ):  # If collision is imminent (within 10 frames)
        player_y = player_pos[1]

        # If projectile is high, crouch
        if projectile_height < player_y:
            recommended_action = 5  # Crouch
        # If projectile is low, jump
        else:
            recommended_action = 4  # Jump

    return {
        "projectiles": projectiles,
        "collision_time": collision_time,
        "projectile_height": projectile_height,
        "recommended_action": recommended_action,
    }
