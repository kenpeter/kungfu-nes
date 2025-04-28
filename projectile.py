import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class OpenCVProjectileDetector:
    """
    A class to detect projectiles in Kung Fu Master using OpenCV.
    Uses computer vision techniques to identify and track moving objects.
    """

    def __init__(
        self,
        min_size=4,
        max_size=30,
        movement_threshold=15,
        projectile_color_ranges=None,
    ):
        """
        Initialize the projectile detector.

        Args:
            min_size: Minimum size of a projectile in pixels
            max_size: Maximum size of a projectile in pixels
            movement_threshold: Threshold for motion detection
            projectile_color_ranges: Optional color ranges to filter projectiles
        """
        self.min_size = min_size
        self.max_size = max_size
        self.movement_threshold = movement_threshold

        # Default color ranges for common projectiles (light objects and fireballs)
        self.projectile_color_ranges = projectile_color_ranges or [
            # light color for knives, etc.
            {"lower": np.array([180, 180, 180]), "upper": np.array([255, 255, 255])},
            # yellow/orange fireball
            {"lower": np.array([0, 140, 180]), "upper": np.array([100, 220, 255])},
        ]

        # Keep track of previous frame and detections
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []

        # For tracking projectiles across frames
        self.next_projectile_id = 1
        self.tracked_projectiles = {}

        # Debug flags
        self.debug = False

        print("OpenCV ProjectileDetector initialized")

    def reset(self):
        """Reset the detector state between episodes"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []
        self.tracked_projectiles = {}
        self.next_projectile_id = 1

    def detect_projectiles(self, current_frame):
        """
        Detect projectiles in the current frame.

        Args:
            current_frame: Current frame as numpy array [height, width, channels]

        Returns:
            List of detected projectiles with position, velocity, and size
        """
        # Handle frame stacks by taking the most recent frame
        if isinstance(current_frame, np.ndarray) and len(current_frame.shape) == 4:
            # Frame stack format [stack_size, height, width, channels]
            current_frame = current_frame[-1]  # Take most recent frame

        # Make a copy to avoid modifying the original
        frame = current_frame.copy()

        # Convert to BGR for OpenCV if it's RGB
        if frame.shape[2] == 3:  # Has color channels
            # Assuming input is RGB, convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # On first frame, just store it and return empty list
        if self.prev_gray is None:
            self.prev_frame = frame_bgr.copy()
            self.prev_gray = gray.copy()
            return []

        # 1. Motion detection with frame differencing
        frame_diff = cv2.absdiff(self.prev_gray, gray)

        # 2. Threshold to get binary image of moving objects
        _, thresh = cv2.threshold(
            frame_diff, self.movement_threshold, 255, cv2.THRESH_BINARY
        )

        # 3. Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 4. Find contours of moving objects
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Process each contour and identify potential projectiles
        current_detections = []
        for contour in contours:
            # Filter by area/size
            area = cv2.contourArea(contour)
            if area < self.min_size or area > self.max_size:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Color filtering (optional)
            if self._check_color_match(frame_bgr, x, y, w, h):
                # Calculate centroid
                cx = x + w // 2
                cy = y + h // 2

                # Add to current detections
                current_detections.append(
                    {
                        "position": (cx, cy),
                        "velocity": (0, 0),  # Will be calculated later
                        "size": area,
                        "width": w,
                        "height": h,
                        "confidence": 0.5,  # Initial confidence
                    }
                )

                # Debug visualization
                if self.debug:
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 6. Match with previous detections to calculate velocities
        final_projectiles = self._match_and_track_projectiles(current_detections)

        # Update previous frame for next iteration
        self.prev_frame = frame_bgr.copy()
        self.prev_gray = gray.copy()
        self.prev_detections = current_detections

        return final_projectiles

    def _check_color_match(self, frame, x, y, w, h):
        """
        Check if the region matches projectile color profiles.
        Returns True if it matches any color range.
        """
        # Extract the ROI (Region of Interest)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:  # Empty ROI
            return False

        # Skip color check if no color ranges defined
        if not self.projectile_color_ranges:
            return True

        # Check against each color range
        for color_range in self.projectile_color_ranges:
            # Create mask for this color range
            mask = cv2.inRange(roi, color_range["lower"], color_range["upper"])

            # Calculate percentage of pixels that match this color range
            match_percentage = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

            # If at least 30% of pixels match, consider it a match
            if match_percentage > 0.3:
                return True

        # Default to motion-based detection
        return True

    def _match_and_track_projectiles(self, current_detections):
        """
        Match current detections with previously tracked projectiles.
        Calculate velocities and update tracking.
        """
        # List to hold final projectiles with velocities and tracking IDs
        final_projectiles = []

        # List to track which current detections have been matched
        matched_indices = set()

        # Dictionary to hold updated projectile tracks
        updated_tracks = {}

        # First, try to match with existing tracked projectiles
        for track_id, track_info in self.tracked_projectiles.items():
            last_pos = track_info["positions"][-1]
            last_vel = (
                track_info["velocities"][-1] if track_info["velocities"] else (0, 0)
            )

            # Predict where this projectile should be now
            pred_x = last_pos[0] + last_vel[0]
            pred_y = last_pos[1] + last_vel[1]

            # Find closest detection
            best_match = -1
            min_dist = 25  # Maximum matching distance threshold

            for i, detection in enumerate(current_detections):
                if i in matched_indices:
                    continue

                x, y = detection["position"]
                dist = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5

                if dist < min_dist:
                    min_dist = dist
                    best_match = i

            # If a match was found
            if best_match >= 0:
                matched_proj = current_detections[best_match]
                matched_indices.add(best_match)

                # Calculate new velocity
                new_x, new_y = matched_proj["position"]
                new_vx = new_x - last_pos[0]
                new_vy = new_y - last_pos[1]

                # Update track information
                updated_tracks[track_id] = {
                    "positions": track_info["positions"] + [(new_x, new_y)],
                    "velocities": track_info["velocities"] + [(new_vx, new_vy)],
                    "sizes": track_info["sizes"] + [matched_proj["size"]],
                    "frames_tracked": track_info["frames_tracked"] + 1,
                    "missed_frames": 0,
                    "last_seen": 0,
                }

                # Only consider reliable tracks (seen in 2+ frames)
                if updated_tracks[track_id]["frames_tracked"] >= 2:
                    # Calculate average velocity from last few frames for stability
                    velocities = updated_tracks[track_id]["velocities"]
                    recent_vels = velocities[-min(3, len(velocities)) :]
                    avg_vx = sum(v[0] for v in recent_vels) / len(recent_vels)
                    avg_vy = sum(v[1] for v in recent_vels) / len(recent_vels)

                    # Calculate confidence based on tracking history
                    confidence = min(
                        0.5 + 0.1 * updated_tracks[track_id]["frames_tracked"], 0.95
                    )

                    # Add to final projectiles list
                    final_projectiles.append(
                        {
                            "id": track_id,
                            "position": (new_x, new_y),
                            "velocity": (avg_vx, avg_vy),
                            "size": matched_proj["size"],
                            "width": matched_proj.get("width", 5),
                            "height": matched_proj.get("height", 5),
                            "confidence": confidence,
                        }
                    )

            else:
                # No match found - projectile might be temporarily occluded
                # Keep track but mark as missed for this frame
                if track_info["missed_frames"] < 3:  # Only keep for a few missed frames
                    # Predict new position using last velocity
                    pred_pos = (pred_x, pred_y)

                    updated_tracks[track_id] = {
                        "positions": track_info["positions"] + [pred_pos],
                        "velocities": track_info["velocities"],  # Keep last velocity
                        "sizes": track_info["sizes"],
                        "frames_tracked": track_info["frames_tracked"],
                        "missed_frames": track_info["missed_frames"] + 1,
                        "last_seen": track_info["last_seen"] + 1,
                    }

                    # Add to final projectiles with lower confidence
                    if updated_tracks[track_id]["frames_tracked"] >= 3:
                        confidence = max(
                            0.3 - 0.1 * updated_tracks[track_id]["missed_frames"], 0.1
                        )

                        # Get last velocity
                        last_vx, last_vy = track_info["velocities"][-1]

                        final_projectiles.append(
                            {
                                "id": track_id,
                                "position": pred_pos,
                                "velocity": (last_vx, last_vy),
                                "size": track_info["sizes"][-1],
                                "width": 5,  # Default size
                                "height": 5,  # Default size
                                "confidence": confidence,
                            }
                        )

        # Start new tracks for unmatched detections
        for i, detection in enumerate(current_detections):
            if i not in matched_indices:
                track_id = self.next_projectile_id
                self.next_projectile_id += 1

                # Initialize new track
                updated_tracks[track_id] = {
                    "positions": [detection["position"]],
                    "velocities": [(0, 0)],  # No velocity yet
                    "sizes": [detection["size"]],
                    "frames_tracked": 1,
                    "missed_frames": 0,
                    "last_seen": 0,
                }

        # Update tracked projectiles for next iteration
        # Only keep active tracks (recently seen)
        self.tracked_projectiles = {
            k: v
            for k, v in updated_tracks.items()
            if v["missed_frames"] < 3 and v["frames_tracked"] < 20
        }

        # Sort projectiles by confidence (highest first)
        final_projectiles.sort(key=lambda x: x["confidence"], reverse=True)

        return final_projectiles

    def recommend_action(self, projectiles, player_position):
        """
        Determine the best defensive action based on projectile trajectories.

        Args:
            projectiles: List of detected projectiles
            player_position: (x, y) tuple of player position

        Returns:
            Recommended action index (0=no-op, 4=jump, 5=crouch)
        """
        if not projectiles:
            return 0  # No action if no projectiles

        player_x, player_y = player_position

        # Only consider projectiles with reasonable confidence
        credible_projectiles = [p for p in projectiles if p.get("confidence", 0) > 0.4]

        for proj in credible_projectiles:
            x, y = proj["position"]
            vx, vy = proj["velocity"]

            # Skip if not moving toward player
            if (x < player_x and vx <= 0) or (x > player_x and vx >= 0):
                continue

            # Check if projectile has significant horizontal velocity
            if abs(vx) > 0.5:
                # Calculate time to potential x-collision
                time_to_x = abs((x - player_x) / vx) if vx != 0 else float("inf")

                # Only consider imminent threats (within next ~10 frames)
                if 0 < time_to_x <= 10:
                    # Predict y position at collision
                    future_y = y + vy * time_to_x

                    # Calculate vertical distance from player
                    player_height = 40  # Approximate player height
                    vertical_margin = 10  # Buffer zone

                    # Player hit zone ranges from feet to head
                    lower_bound = player_y - player_height
                    upper_bound = player_y + vertical_margin

                    # If projectile will be in player's vertical range at collision time
                    if (
                        lower_bound - vertical_margin
                        <= future_y
                        <= upper_bound + vertical_margin
                    ):
                        # Determine optimal defensive action
                        if future_y < player_y - player_height / 2:
                            return 5  # Crouch if projectile will be in upper half
                        else:
                            return 4  # Jump if projectile will be in lower half

                        # Priority to first imminent threat
                        break

        # No imminent threat detected
        return 0  # No-op


class OpenCVProjectileDetector:
    """
    A class to detect projectiles in Kung Fu Master using OpenCV.
    Uses computer vision techniques to identify and track moving objects.
    """

    def __init__(
        self,
        min_size=4,
        max_size=30,
        movement_threshold=15,
        projectile_color_ranges=None,
    ):
        """
        Initialize the projectile detector.

        Args:
            min_size: Minimum size of a projectile in pixels
            max_size: Maximum size of a projectile in pixels
            movement_threshold: Threshold for motion detection
            projectile_color_ranges: Optional color ranges to filter projectiles
        """
        self.min_size = min_size
        self.max_size = max_size
        self.movement_threshold = movement_threshold

        # Default color ranges for common projectiles (light objects and fireballs)
        self.projectile_color_ranges = projectile_color_ranges or [
            # light color for knives, etc.
            {"lower": np.array([180, 180, 180]), "upper": np.array([255, 255, 255])},
            # yellow/orange fireball
            {"lower": np.array([0, 140, 180]), "upper": np.array([100, 220, 255])},
        ]

        # Keep track of previous frame and detections
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []

        # For tracking projectiles across frames
        self.next_projectile_id = 1
        self.tracked_projectiles = {}

        # Debug flags
        self.debug = False

        print("OpenCV ProjectileDetector initialized")

    def reset(self):
        """Reset the detector state between episodes"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []
        self.tracked_projectiles = {}
        self.next_projectile_id = 1

    def detect_projectiles(self, current_frame):
        """
        Detect projectiles in the current frame.

        Args:
            current_frame: Current frame as numpy array [height, width, channels]

        Returns:
            List of detected projectiles with position, velocity, and size
        """
        # Handle frame stacks by taking the most recent frame
        if isinstance(current_frame, np.ndarray) and len(current_frame.shape) == 4:
            # Frame stack format [stack_size, height, width, channels]
            current_frame = current_frame[-1]  # Take most recent frame

        # Make a copy to avoid modifying the original
        frame = current_frame.copy()

        # Convert to BGR for OpenCV if it's RGB
        if frame.shape[2] == 3:  # Has color channels
            # Assuming input is RGB, convert to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # On first frame, just store it and return empty list
        if self.prev_gray is None:
            self.prev_frame = frame_bgr.copy()
            self.prev_gray = gray.copy()
            return []

        # 1. Motion detection with frame differencing
        frame_diff = cv2.absdiff(self.prev_gray, gray)

        # 2. Threshold to get binary image of moving objects
        _, thresh = cv2.threshold(
            frame_diff, self.movement_threshold, 255, cv2.THRESH_BINARY
        )

        # 3. Clean up with morphological operations
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 4. Find contours of moving objects
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 5. Process each contour and identify potential projectiles
        current_detections = []
        for contour in contours:
            # Filter by area/size
            area = cv2.contourArea(contour)
            if area < self.min_size or area > self.max_size:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Color filtering (optional)
            if self._check_color_match(frame_bgr, x, y, w, h):
                # Calculate centroid
                cx = x + w // 2
                cy = y + h // 2

                # Add to current detections
                current_detections.append(
                    {
                        "position": (cx, cy),
                        "velocity": (0, 0),  # Will be calculated later
                        "size": area,
                        "width": w,
                        "height": h,
                        "confidence": 0.5,  # Initial confidence
                    }
                )

                # Debug visualization
                if self.debug:
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 6. Match with previous detections to calculate velocities
        final_projectiles = self._match_and_track_projectiles(current_detections)

        # Update previous frame for next iteration
        self.prev_frame = frame_bgr.copy()
        self.prev_gray = gray.copy()
        self.prev_detections = current_detections

        return final_projectiles

    def _check_color_match(self, frame, x, y, w, h):
        """
        Check if the region matches projectile color profiles.
        Returns True if it matches any color range.
        """
        # Extract the ROI (Region of Interest)
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:  # Empty ROI
            return False

        # Skip color check if no color ranges defined
        if not self.projectile_color_ranges:
            return True

        # Check against each color range
        for color_range in self.projectile_color_ranges:
            # Create mask for this color range
            mask = cv2.inRange(roi, color_range["lower"], color_range["upper"])

            # Calculate percentage of pixels that match this color range
            match_percentage = cv2.countNonZero(mask) / (roi.shape[0] * roi.shape[1])

            # If at least 30% of pixels match, consider it a match
            if match_percentage > 0.3:
                return True

        # Default to motion-based detection
        return True

    def _match_and_track_projectiles(self, current_detections):
        """
        Match current detections with previously tracked projectiles.
        Calculate velocities and update tracking.
        """
        # List to hold final projectiles with velocities and tracking IDs
        final_projectiles = []

        # List to track which current detections have been matched
        matched_indices = set()

        # Dictionary to hold updated projectile tracks
        updated_tracks = {}

        # First, try to match with existing tracked projectiles
        for track_id, track_info in self.tracked_projectiles.items():
            last_pos = track_info["positions"][-1]
            last_vel = (
                track_info["velocities"][-1] if track_info["velocities"] else (0, 0)
            )

            # Predict where this projectile should be now
            pred_x = last_pos[0] + last_vel[0]
            pred_y = last_pos[1] + last_vel[1]

            # Find closest detection
            best_match = -1
            min_dist = 25  # Maximum matching distance threshold

            for i, detection in enumerate(current_detections):
                if i in matched_indices:
                    continue

                x, y = detection["position"]
                dist = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5

                if dist < min_dist:
                    min_dist = dist
                    best_match = i

            # If a match was found
            if best_match >= 0:
                matched_proj = current_detections[best_match]
                matched_indices.add(best_match)

                # Calculate new velocity
                new_x, new_y = matched_proj["position"]
                new_vx = new_x - last_pos[0]
                new_vy = new_y - last_pos[1]

                # Update track information
                updated_tracks[track_id] = {
                    "positions": track_info["positions"] + [(new_x, new_y)],
                    "velocities": track_info["velocities"] + [(new_vx, new_vy)],
                    "sizes": track_info["sizes"] + [matched_proj["size"]],
                    "frames_tracked": track_info["frames_tracked"] + 1,
                    "missed_frames": 0,
                    "last_seen": 0,
                }

                # Only consider reliable tracks (seen in 2+ frames)
                if updated_tracks[track_id]["frames_tracked"] >= 2:
                    # Calculate average velocity from last few frames for stability
                    velocities = updated_tracks[track_id]["velocities"]
                    recent_vels = velocities[-min(3, len(velocities)) :]
                    avg_vx = sum(v[0] for v in recent_vels) / len(recent_vels)
                    avg_vy = sum(v[1] for v in recent_vels) / len(recent_vels)

                    # Calculate confidence based on tracking history
                    confidence = min(
                        0.5 + 0.1 * updated_tracks[track_id]["frames_tracked"], 0.95
                    )

                    # Add to final projectiles list
                    final_projectiles.append(
                        {
                            "id": track_id,
                            "position": (new_x, new_y),
                            "velocity": (avg_vx, avg_vy),
                            "size": matched_proj["size"],
                            "width": matched_proj.get("width", 5),
                            "height": matched_proj.get("height", 5),
                            "confidence": confidence,
                        }
                    )

            else:
                # No match found - projectile might be temporarily occluded
                # Keep track but mark as missed for this frame
                if track_info["missed_frames"] < 3:  # Only keep for a few missed frames
                    # Predict new position using last velocity
                    pred_pos = (pred_x, pred_y)

                    updated_tracks[track_id] = {
                        "positions": track_info["positions"] + [pred_pos],
                        "velocities": track_info["velocities"],  # Keep last velocity
                        "sizes": track_info["sizes"],
                        "frames_tracked": track_info["frames_tracked"],
                        "missed_frames": track_info["missed_frames"] + 1,
                        "last_seen": track_info["last_seen"] + 1,
                    }

                    # Add to final projectiles with lower confidence
                    if updated_tracks[track_id]["frames_tracked"] >= 3:
                        confidence = max(
                            0.3 - 0.1 * updated_tracks[track_id]["missed_frames"], 0.1
                        )

                        # Get last velocity
                        last_vx, last_vy = track_info["velocities"][-1]

                        final_projectiles.append(
                            {
                                "id": track_id,
                                "position": pred_pos,
                                "velocity": (last_vx, last_vy),
                                "size": track_info["sizes"][-1],
                                "width": 5,  # Default size
                                "height": 5,  # Default size
                                "confidence": confidence,
                            }
                        )

        # Start new tracks for unmatched detections
        for i, detection in enumerate(current_detections):
            if i not in matched_indices:
                track_id = self.next_projectile_id
                self.next_projectile_id += 1

                # Initialize new track
                updated_tracks[track_id] = {
                    "positions": [detection["position"]],
                    "velocities": [(0, 0)],  # No velocity yet
                    "sizes": [detection["size"]],
                    "frames_tracked": 1,
                    "missed_frames": 0,
                    "last_seen": 0,
                }

        # Update tracked projectiles for next iteration
        # Only keep active tracks (recently seen)
        self.tracked_projectiles = {
            k: v
            for k, v in updated_tracks.items()
            if v["missed_frames"] < 3 and v["frames_tracked"] < 20
        }

        # Sort projectiles by confidence (highest first)
        final_projectiles.sort(key=lambda x: x["confidence"], reverse=True)

        return final_projectiles

    def recommend_action(self, projectiles, player_position):
        """
        Determine the best defensive action based on projectile trajectories.

        Args:
            projectiles: List of detected projectiles
            player_position: (x, y) tuple of player position

        Returns:
            Recommended action index (0=no-op, 4=jump, 5=crouch)
        """
        if not projectiles:
            return 0  # No action if no projectiles

        player_x, player_y = player_position

        # Debug output
        if len(projectiles) > 0:
            print(
                f"Analyzing {len(projectiles)} projectiles for threat assessment at player pos {player_position}"
            )

        # Much more aggressive detection - ANY projectile is potentially a threat
        # Include even low confidence projectiles
        credible_projectiles = projectiles

        # Closest threatening projectile and its details
        closest_threat_time = float("inf")
        recommended_action = 0

        for proj in credible_projectiles:
            x, y = proj["position"]
            vx, vy = proj["velocity"]

            # Debug output for all projectiles
            print(f"  Projectile at ({x},{y}) with velocity ({vx},{vy})")

            # Consider ALL projectiles as potential threats
            # Instead of filtering by direction, just check if they're within range

            # Distance-based threat assessment
            horizontal_distance = abs(x - player_x)
            vertical_distance = abs(y - player_y)

            # Consider any projectile within this radius as a threat
            threat_radius = 100

            # Simple distance calculation
            distance = ((x - player_x) ** 2 + (y - player_y) ** 2) ** 0.5

            # If projectile is close enough to be a threat
            if distance < threat_radius:
                print(
                    f"  Projectile is within threat radius ({distance:.1f} < {threat_radius})"
                )

                # Estimate time to impact based on distance and velocity
                if abs(vx) > 0.1:  # If moving horizontally
                    time_to_x = horizontal_distance / max(abs(vx), 0.1)
                else:  # Slow or stationary
                    time_to_x = horizontal_distance / 5  # Assume some speed

                # Extremely generous time window (up to 20 frames)
                if time_to_x <= 20:
                    print(
                        f"  Potential threat - estimated time to player: {time_to_x:.1f} frames"
                    )

                    # Predict future y position
                    future_y = y + (vy * time_to_x if abs(vx) > 0.1 else 0)

                    # Very generous vertical threat zone
                    player_height = 50  # Increased player height
                    vertical_margin = 30  # Much larger buffer

                    # Player hit zone ranges from feet to head
                    lower_bound = player_y - player_height
                    upper_bound = player_y

                    print(
                        f"  Projectile will be at y={future_y} when it reaches player x"
                    )
                    print(f"  Player vertical range: {lower_bound} to {upper_bound}")

                    # If anywhere near the player's vertical range
                    in_vertical_range = (
                        lower_bound - vertical_margin
                        <= future_y
                        <= upper_bound + vertical_margin
                    )

                    # Or if it's just generally close
                    is_very_close = distance < 50

                    if in_vertical_range or is_very_close:
                        print(
                            f"  THREAT DETECTED - time to impact: {time_to_x:.1f} frames"
                        )

                        # If this is the closest threatening projectile so far
                        if time_to_x < closest_threat_time:
                            closest_threat_time = time_to_x

                            # Determine defensive action based on projectile position
                            # Simple rule: if projectile is above player's center, crouch; otherwise jump
                            if y < player_y - 10:
                                recommended_action = 5  # Crouch if projectile is above
                                print(
                                    f"  Recommending CROUCH - projectile is above player"
                                )
                            else:
                                recommended_action = (
                                    4  # Jump if projectile is below or at level
                                )
                                print(
                                    f"  Recommending JUMP - projectile is at or below player"
                                )

        # If no specific threat detected but projectiles exist, be defensive anyway
        if recommended_action == 0 and len(projectiles) > 0:
            # If any projectile is within extreme close range, take defensive action
            for proj in projectiles:
                x, y = proj["position"]
                distance = ((x - player_x) ** 2 + (y - player_y) ** 2) ** 0.5

                if distance < 40:  # Extreme close range
                    print(f"  PROXIMITY ALERT - projectile at distance {distance:.1f}")
                    # Default to jump as a general defensive move
                    recommended_action = 4
                    print(f"  Recommending JUMP as general defensive action")
                    break

        # Return the action for the closest threat
        return recommended_action


def enhance_observation_with_projectiles(current_frame, detector, player_position):
    """
    Process the current frame to detect projectiles and recommend actions.

    Args:
        current_frame: The current game frame
        detector: OpenCVProjectileDetector instance
        player_position: (x, y) tuple of player position

    Returns:
        Dictionary with projectile information and recommended action
    """
    try:
        # Validate player position format
        if not isinstance(player_position, tuple) or len(player_position) != 2:
            print(f"Warning: Invalid player_position format: {player_position}")
            player_position = (0, 0)

        # Detect projectiles using OpenCV
        projectiles = detector.detect_projectiles(current_frame)

        # Get recommended defensive action
        recommended_action = detector.recommend_action(projectiles, player_position)

        # Action name mapping for clear logging
        action_names = [
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

        # Only log meaningful detections to reduce spam
        if len(projectiles) > 0:
            action_name = (
                action_names[recommended_action]
                if 0 <= recommended_action < len(action_names)
                else f"Unknown({recommended_action})"
            )
            print(
                f"Detected {len(projectiles)} projectiles, recommended action: {action_name}"
            )

            # Extra logging for defensive actions
            if recommended_action in [4, 5]:  # Jump or Crouch
                print(f"DEFENSIVE ACTION RECOMMENDED: {action_name}")

        # Return enhanced observation
        return {"projectiles": projectiles, "recommended_action": recommended_action}

    except Exception as e:
        print(f"Error in enhance_observation_with_projectiles: {e}")
        # Return empty results as fallback
        return {"projectiles": [], "recommended_action": 0}
