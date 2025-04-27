import numpy as np


class ProjectileDetector:
    """
    A class to detect projectiles in frame stacks by analyzing pixel changes
    across multiple frames. Optimized for Kung Fu Master's projectiles.
    Utilizes full n-frame stack for more robust detection and tracking.
    """

    # init self, projectile color, movement threshold, min project, max project
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
        self.projectile_color_ranges = projectile_color_ranges or [
            # light color for kinves, etc
            {"r": (180, 255), "g": (180, 255), "b": (180, 255)},
            # yellow orange fireball
            {"r": (200, 255), "g": (140, 220), "b": (20, 100)},
        ]

        # how fast threshold
        self.movement_threshold = movement_threshold
        # projectile min
        self.min_projectile_size = min_projectile_size
        # projectile max
        self.max_projectile_size = max_projectile_size

        # prev projectile arr
        self.previous_projectiles = []

        # detect motion pattern
        self.projectile_trajectories = []

        # Projectile ID counter to maintain identity across frames
        self.next_projectile_id = 1

    def detect_projectiles(self, frame_stack):
        """
        Detect projectiles in a stack of frames, utilizing the full history.

        Args:
            frame_stack: Numpy array of stacked frames [stack_size, height, width, channels]
                         or [stack_size, height, width] for grayscale

        Returns:
            List of detected projectiles with their positions and velocities
            Format: [{'id': id, 'x': x, 'y': y, 'vx': vx, 'vy': vy, 'size': size, 'confidence': conf}, ...]
        """
        stack_size = frame_stack.shape[0]
        current_frame = frame_stack[-1]

        # Create a difference map using multiple frame comparisons
        diff_accumulator = np.zeros_like(frame_stack[0], dtype=np.float32)
        if diff_accumulator.ndim == 3:  # RGB
            diff_accumulator = np.mean(diff_accumulator, axis=2)

        # up to 5 recent frames, more recent and more higher weight
        for i in range(1, min(stack_size, 5)):
            # recent frame has more weight
            weight = 1.0 / (i)
            # prev index, and current index, but backward
            prev_idx = -i - 1
            curr_idx = -i

            # need to bound check
            if abs(prev_idx) >= stack_size or abs(curr_idx) >= stack_size:
                continue

            # frame diff
            frame_diff = np.abs(
                frame_stack[curr_idx].astype(np.float32)
                - frame_stack[prev_idx].astype(np.float32)
            )

            # if rgb, convert to grey scale
            if frame_diff.ndim == 3:
                frame_diff = np.mean(frame_diff, axis=2)

            # Accumulate weighted differences
            diff_accumulator += weight * frame_diff

        # Threshold to find areas of consistent movement
        movement_mask = diff_accumulator > self.movement_threshold
        movement_mask = movement_mask.astype(np.uint8)

        # Find connected components (potential projectiles)
        labeled_mask, num_objects = self._connected_components(movement_mask)

        # Extract potential projectiles from current frame
        current_projectiles = []
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
                # Add to current projectiles list
                current_projectiles.append(
                    {
                        "x": int(x),
                        "y": int(y),
                        "size": len(obj_pixels[0]),
                        "pixels": obj_pixels,
                    }
                )

        # Match current detections with trajectories from previous frames
        final_projectiles = self._track_projectiles_across_frames(
            current_projectiles, frame_stack
        )

        # Update previous projectiles for next iteration
        self.previous_projectiles = final_projectiles

        return final_projectiles

    def _track_projectiles_across_frames(self, current_projectiles, frame_stack):
        """
        Track projectiles across multiple frames to establish trajectories,
        calculate velocities, and filter false positives.
        """
        stack_size = frame_stack.shape[0]
        result_projectiles = []

        # Step 1: Match current projectiles with existing trajectories
        matched_indices = set()

        # Copy trajectories for modification
        updated_trajectories = []

        # need to predict where is next projectile will be
        for traj in self.projectile_trajectories:
            # last x, y position
            last_pos = traj["positions"][-1]  # [x, y]
            last_vel = (
                traj["velocities"][-1] if traj["velocities"] else [0, 0]
            )  # [vx, vy]

            predicted_x = last_pos[0] + last_vel[0]
            predicted_y = last_pos[1] + last_vel[1]

            # Find closest detection in current frame
            best_match = -1
            min_dist = 20  # Max distance to consider as the same projectile

            for i, proj in enumerate(current_projectiles):
                if i in matched_indices:
                    continue

                dist = (
                    (proj["x"] - predicted_x) ** 2 + (proj["y"] - predicted_y) ** 2
                ) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    best_match = i

            # If we found a match, update the trajectory
            if best_match >= 0:
                matched_proj = current_projectiles[best_match]
                matched_indices.add(best_match)

                # Calculate velocity
                new_x, new_y = matched_proj["x"], matched_proj["y"]
                new_vx = new_x - last_pos[0]
                new_vy = new_y - last_pos[1]

                # Check for velocity consistency
                velocity_consistent = True
                if len(traj["velocities"]) >= 2:
                    prev_vx, prev_vy = traj["velocities"][-1]
                    # Allow some variation but detect large changes
                    if abs(new_vx - prev_vx) > 3 or abs(new_vy - prev_vy) > 3:
                        velocity_consistent = False

                # Update trajectory
                updated_traj = {
                    "id": traj["id"],
                    "positions": traj["positions"] + [[new_x, new_y]],
                    "velocities": traj["velocities"] + [[new_vx, new_vy]],
                    "sizes": traj["sizes"] + [matched_proj["size"]],
                    "age": traj["age"] + 1,
                    "missed_frames": 0,
                }

                # Calculate confidence based on trajectory consistency and length
                confidence = min(0.5 + 0.1 * updated_traj["age"], 0.95)
                if not velocity_consistent:
                    confidence *= 0.8

                # Add to results if trajectory is mature (seen in multiple frames)
                if updated_traj["age"] >= 2:
                    # Use average velocity from last few frames for stability
                    avg_vels = np.mean(
                        updated_traj["velocities"][
                            -min(3, len(updated_traj["velocities"])) :
                        ],
                        axis=0,
                    )
                    vx, vy = avg_vels[0], avg_vels[1]

                    result_projectiles.append(
                        {
                            "id": updated_traj["id"],
                            "x": new_x,
                            "y": new_y,
                            "vx": vx,
                            "vy": vy,
                            "size": matched_proj["size"],
                            "confidence": confidence,
                            "position": (new_x, new_y),  # For compatibility
                            "velocity": (vx, vy),  # For compatibility
                        }
                    )

                updated_trajectories.append(updated_traj)
            else:
                # No match found - projectile might be temporarily occluded
                # Keep trajectory but mark it as missed for this frame
                if (
                    traj["missed_frames"] < 3
                ):  # Only keep for a limited number of missed frames
                    # Predict position using last velocity
                    predicted_pos = [predicted_x, predicted_y]

                    updated_traj = {
                        "id": traj["id"],
                        "positions": traj["positions"] + [predicted_pos],
                        "velocities": traj["velocities"],  # Keep last velocity
                        "sizes": traj["sizes"],
                        "age": traj["age"],
                        "missed_frames": traj["missed_frames"] + 1,
                    }

                    # Add to results with lower confidence
                    if updated_traj["age"] >= 3:  # Only keep mature trajectories
                        confidence = max(0.3 - 0.1 * updated_traj["missed_frames"], 0.1)
                        vx, vy = updated_traj["velocities"][-1]

                        result_projectiles.append(
                            {
                                "id": updated_traj["id"],
                                "x": predicted_pos[0],
                                "y": predicted_pos[1],
                                "vx": vx,
                                "vy": vy,
                                "size": updated_traj["sizes"][-1],
                                "confidence": confidence,
                                "position": (
                                    predicted_pos[0],
                                    predicted_pos[1],
                                ),  # For compatibility
                                "velocity": (vx, vy),  # For compatibility
                            }
                        )

                    updated_trajectories.append(updated_traj)

        # Step 2: Create new trajectories for unmatched detections
        for i, proj in enumerate(current_projectiles):
            if i not in matched_indices:
                # Start new trajectory
                new_traj = {
                    "id": self.next_projectile_id,
                    "positions": [[proj["x"], proj["y"]]],
                    "velocities": [],  # No velocity yet
                    "sizes": [proj["size"]],
                    "age": 1,
                    "missed_frames": 0,
                }

                self.next_projectile_id += 1
                updated_trajectories.append(new_traj)

                # Don't add to results yet - wait until trajectory is confirmed in multiple frames

        # Update trajectories for next iteration
        # Only keep trajectories that are recent enough (limit age or missed frames)
        self.projectile_trajectories = [
            traj
            for traj in updated_trajectories
            if traj["missed_frames"] < 3 and traj["age"] < 15
        ]

        return result_projectiles

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

    def predict_collision(
        self,
        projectiles,
        player_x,
        player_y,
        player_width=20,
        player_height=40,
        lookahead_frames=15,
    ):
        """
        Predict if any projectiles will collide with the player in the near future

        Args:
            projectiles: List of detected projectiles
            player_x: Player's x position
            player_y: Player's y position
            player_width: Player width in pixels
            player_height: Player height in pixels
            lookahead_frames: Maximum number of frames to predict ahead

        Returns:
            collision_time: Estimated frames until collision (-1 if no collision)
            projectile_height: Height of the incoming projectile (for jump/crouch decision)
            confidence: Confidence level in the collision prediction (0-1)
        """
        collision_time = -1
        projectile_height = 0
        collision_confidence = 0

        # Sort projectiles by confidence to prioritize more reliable detections
        sorted_projectiles = sorted(
            projectiles, key=lambda p: p.get("confidence", 0), reverse=True
        )

        for proj in sorted_projectiles:
            # Skip low confidence projectiles
            if proj.get("confidence", 0) < 0.3:
                continue

            # Skip projectiles moving away from the player
            if (proj["x"] < player_x and proj["vx"] < 0) or (
                proj["x"] > player_x and proj["vx"] > 0
            ):
                continue

            # Calculate time to x-collision
            if abs(proj["vx"]) > 0.5:  # If projectile has significant x velocity
                time_to_x = abs((proj["x"] - player_x) / proj["vx"])

                # Skip if too far into the future
                if time_to_x > lookahead_frames:
                    continue

                # Predict y position at collision time
                future_y = proj["y"] + proj["vy"] * time_to_x

                # Calculate vertical distance from player center at collision time
                v_distance = abs(future_y - player_y)

                # Check if y position will be within player bounds with some margin
                if v_distance <= (player_height / 2) + 5:
                    # If this is the closest projectile on a collision course or higher confidence
                    current_confidence = proj.get("confidence", 0.5) * (
                        1 - time_to_x / lookahead_frames
                    )

                    if collision_time == -1 or (
                        time_to_x < collision_time
                        and current_confidence >= collision_confidence
                    ):
                        collision_time = time_to_x
                        projectile_height = future_y
                        collision_confidence = current_confidence

        return collision_time, projectile_height, collision_confidence


def enhance_observation_with_projectiles(frame_stack, detector, player_position):
    """
    Detect projectiles and enhance observation with projectile information

    Args:
        frame_stack: Stack of recent frames
        detector: ProjectileDetector instance
        player_position: (x, y) position of the player

    Returns:
        Dictionary with projectile information and recommended action
    """
    # Existing detection code...
    projectiles = detector.detect_projectiles(frame_stack)

    # Enhanced data: Calculate velocity and trajectory info for each projectile
    player_x, player_y = player_position

    for i, projectile in enumerate(projectiles):
        x, y = (
            projectile["position"]
            if "position" in projectile
            else (projectile["x"], projectile["y"])
        )
        width, height = projectile.get("size", (0, 0))
        if not isinstance(width, tuple):
            width, height = (
                width,
                width,
            )  # Use size as both width and height if not a tuple

        # Make sure position is set
        projectile["position"] = (x, y)

        # If we have a previous position, calculate velocity
        if "prev_position" in projectile:
            prev_x, prev_y = projectile["prev_position"]
            projectile["velocity"] = (x - prev_x, y - prev_y)
        else:
            # If this is a new projectile, try to match with previous frames
            # to estimate velocity
            if i > 0 and len(frame_stack) > 1:
                # Try to find similar projectile in previous detection
                # This is simplified - could be more sophisticated
                projectile["velocity"] = (0, 0)  # Default if no match found

        # Calculate trajectory information
        dx = player_x - x
        dy = player_y - y
        projectile["distance"] = np.sqrt(dx**2 + dy**2)

        # Calculate time to impact if velocity is available
        if "velocity" in projectile:
            vel_x, vel_y = projectile["velocity"]
            vel_magnitude = max(np.sqrt(vel_x**2 + vel_y**2), 0.01)  # Avoid div by zero
            projectile["time_to_impact"] = projectile["distance"] / vel_magnitude

            # Determine if projectile is approaching player
            is_approaching = (
                (vel_x > 0 and dx < 0)
                or (vel_x < 0 and dx > 0)
                or (vel_y > 0 and dy < 0)
                or (vel_y < 0 and dy > 0)
            )
            projectile["is_approaching"] = is_approaching

    # Enhanced recommendation logic based on projectile analysis
    recommended_action = 0  # Default: no-op

    # Find the most threatening projectile
    threat_level = 0
    for projectile in projectiles:
        # Only consider approaching projectiles
        if not projectile.get("is_approaching", False):
            continue

        # Calculate threat based on distance and time to impact
        distance = projectile.get("distance", float("inf"))
        time_to_impact = projectile.get("time_to_impact", float("inf"))

        # Simple threat calculation - threat is higher for closer projectiles
        # and shorter time to impact
        current_threat = 0
        if distance < 100 and time_to_impact < 30:
            current_threat = (100 - distance) + (30 - time_to_impact)

        if current_threat > threat_level:
            threat_level = current_threat

            # Determine appropriate defensive action
            # This logic will depend on your game mechanics
            pos_x, pos_y = projectile["position"]
            vel_x, vel_y = projectile.get("velocity", (0, 0))

            # Example logic: Jump (4) if projectile is coming low, crouch (5) if coming high
            if pos_y > player_y + 10:
                recommended_action = 4  # Jump
            elif pos_y < player_y - 10:
                recommended_action = 5  # Crouch
            else:
                # If coming directly at player, choose based on velocity
                if abs(vel_y) > abs(vel_x):
                    recommended_action = 4 if vel_y > 0 else 5
                else:
                    # If mostly horizontal, try to jump
                    recommended_action = 4

    return {
        "projectiles": projectiles,
        "recommended_action": recommended_action,
    }
