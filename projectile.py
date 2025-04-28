import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time


import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time


class ImprovedProjectileDetector:
    """
    An improved class to detect projectiles in Kung Fu Master using OpenCV.
    Addresses timing issues with defensive actions and improves threat assessment.
    """

    def __init__(
        self,
        min_size=4,
        max_size=30,
        movement_threshold=8,  # Further reduced for higher sensitivity
        projectile_color_ranges=None,
        debug=False,
    ):
        """
        Initialize the projectile detector with improved parameters.

        Args:
            min_size: Minimum size of a projectile in pixels
            max_size: Maximum size of a projectile in pixels
            movement_threshold: Threshold for motion detection
            projectile_color_ranges: Optional color ranges to filter projectiles
            debug: Enable debug mode for verbose logging
        """
        self.min_size = min_size
        self.max_size = max_size
        self.movement_threshold = movement_threshold

        # Enhanced color ranges for better detection
        self.projectile_color_ranges = projectile_color_ranges or [
            # light color for knives, etc.
            {"lower": np.array([170, 170, 170]), "upper": np.array([255, 255, 255])},
            # yellow/orange fireball
            {"lower": np.array([0, 140, 180]), "upper": np.array([100, 220, 255])},
            # darker projectiles
            {"lower": np.array([50, 50, 50]), "upper": np.array([150, 150, 150])},
        ]

        # Keep track of previous frame and detections
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []

        # For tracking projectiles across frames
        self.next_projectile_id = 1
        self.tracked_projectiles = {}

        # Debug flag
        self.debug = debug

        # Action cooldown system to prevent rapid action switching
        self.last_action_time = 0
        self.action_cooldown = 0.1  # seconds
        self.last_recommended_action = 0

        # Frame counter for timing
        self.frame_counter = 0

        # Threat memory to track past threats and avoid repeated warnings
        self.threat_memory = {}

        # Player state tracking
        self.player_state = {
            "position": (0, 0),
            "in_defensive_action": False,
            "defensive_action_start": 0,
            "defensive_action_type": None,
            "last_health": 0,
            "current_health": 0,
        }

        # Enhanced success tracking
        self.successful_avoidance_count = 0
        self.failed_avoidance_count = 0

        # Critical threats for immediate response
        self.critical_threats = []

        # Add counters for better debugging
        self.total_threats_detected = 0
        self.total_defensive_recommendations = 0

        # Early detection parameters
        self.early_detection_factor = 1.5  # Detect threats earlier
        self.last_player_actions = []  # Track recent player actions

        print("Improved ProjectileDetector initialized with enhanced threat detection")

    def reset(self):
        """Reset the detector state between episodes"""
        self.prev_frame = None
        self.prev_gray = None
        self.prev_detections = []
        self.tracked_projectiles = {}
        self.next_projectile_id = 1
        self.last_action_time = 0
        self.last_recommended_action = 0
        self.frame_counter = 0
        self.threat_memory = {}
        self.player_state = {
            "position": (0, 0),
            "in_defensive_action": False,
            "defensive_action_start": 0,
            "defensive_action_type": None,
        }

    def detect_projectiles(self, current_frame):
        """
        Detect projectiles in the current frame with improved filtering.

        Args:
            current_frame: Current frame as numpy array [height, width, channels]

        Returns:
            List of detected projectiles with position, velocity, and size
        """
        self.frame_counter += 1

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

            # Reject objects that are too tall or too wide (likely not projectiles)
            if h > 3 * w or w > 3 * h:
                continue

            # Color filtering
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
                        "frame_detected": self.frame_counter,
                    }
                )

        # 6. Match with previous detections to calculate velocities
        final_projectiles = self._match_and_track_projectiles(current_detections)

        # 7. Filter out stationary objects that haven't moved after several frames
        filtered_projectiles = self._filter_stationary_objects(final_projectiles)

        # Update previous frame for next iteration
        self.prev_frame = frame_bgr.copy()
        self.prev_gray = gray.copy()
        self.prev_detections = current_detections

        return filtered_projectiles

    def _check_color_match(self, frame, x, y, w, h):
        """
        Check if the region matches projectile color profiles with improved matching.
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

            # If at least 25% of pixels match (reduced threshold), consider it a match
            if match_percentage > 0.25:
                return True

        # Default to motion-based detection
        return True

    def _filter_stationary_objects(self, projectiles):
        """
        Filter out objects that appear to be stationary background elements.

        Args:
            projectiles: List of detected projectiles

        Returns:
            Filtered list with likely stationary objects removed
        """
        filtered = []

        for proj in projectiles:
            vx, vy = proj["velocity"]

            # Skip objects with very low velocity in both directions after being tracked for a while
            frames_tracked = self.tracked_projectiles.get(proj.get("id"), {}).get(
                "frames_tracked", 0
            )

            # If it's been tracked for several frames and has almost no movement, likely background
            if frames_tracked > 5 and abs(vx) < 0.3 and abs(vy) < 0.3:
                continue

            # Check if object is outside visible area
            x, y = proj["position"]
            if x < 0 or x > 255 or y < 0 or y > 240:
                continue

            filtered.append(proj)

        return filtered

    def _match_and_track_projectiles(self, current_detections):
        """
        Match current detections with previously tracked projectiles with improved matching.
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
            if not track_info["positions"]:
                continue

            last_pos = track_info["positions"][-1]
            last_vel = (
                track_info["velocities"][-1] if track_info["velocities"] else (0, 0)
            )

            # Predict where this projectile should be now
            pred_x = last_pos[0] + last_vel[0]
            pred_y = last_pos[1] + last_vel[1]

            # Find closest detection
            best_match = -1
            min_dist = 20  # Reduced maximum matching distance for better precision

            # For fast-moving objects, increase matching distance
            speed = (last_vel[0] ** 2 + last_vel[1] ** 2) ** 0.5
            if speed > 5:
                min_dist = 30  # Allow larger matching window for fast objects

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
                    # Calculate average velocity with more weight on recent measurements
                    velocities = updated_tracks[track_id]["velocities"]
                    # Use up to last 5 frames, with more weight on recent ones
                    recent_vels = velocities[-min(5, len(velocities)) :]

                    # Apply exponential weighting - more recent velocities count more
                    weights = [1.5**i for i in range(len(recent_vels))]
                    total_weight = sum(weights)

                    avg_vx = (
                        sum(v[0] * w for v, w in zip(recent_vels, weights))
                        / total_weight
                    )
                    avg_vy = (
                        sum(v[1] * w for v, w in zip(recent_vels, weights))
                        / total_weight
                    )

                    # Calculate confidence based on tracking history and movement
                    # Higher velocity = higher confidence it's a projectile
                    velocity_magnitude = (avg_vx**2 + avg_vy**2) ** 0.5
                    velocity_confidence = min(
                        velocity_magnitude * 0.1, 0.4
                    )  # Up to 0.4 confidence from velocity
                    history_confidence = min(
                        0.3 + 0.05 * updated_tracks[track_id]["frames_tracked"], 0.5
                    )  # Up to 0.5 from history

                    confidence = min(velocity_confidence + history_confidence, 0.95)

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
                            "frames_tracked": updated_tracks[track_id][
                                "frames_tracked"
                            ],
                        }
                    )

            else:
                # No match found - projectile might be temporarily occluded
                # Keep track but mark as missed for this frame
                if track_info["missed_frames"] < 2:  # Reduced missed frames tolerance
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
                        # Rapidly decrease confidence for missing objects
                        confidence = max(
                            0.3 - 0.2 * updated_tracks[track_id]["missed_frames"], 0.1
                        )

                        # Get last velocity
                        if track_info["velocities"]:
                            last_vx, last_vy = track_info["velocities"][-1]

                            final_projectiles.append(
                                {
                                    "id": track_id,
                                    "position": pred_pos,
                                    "velocity": (last_vx, last_vy),
                                    "size": (
                                        track_info["sizes"][-1]
                                        if track_info["sizes"]
                                        else 5
                                    ),
                                    "width": 5,  # Default size
                                    "height": 5,  # Default size
                                    "confidence": confidence,
                                    "frames_tracked": updated_tracks[track_id][
                                        "frames_tracked"
                                    ],
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
            if v["missed_frames"] < 2
            and v["frames_tracked"] < 15  # Shorter tracking history
        }

        # Sort projectiles by confidence (highest first)
        final_projectiles.sort(key=lambda x: x["confidence"], reverse=True)

        return final_projectiles

    def recommend_action(self, projectiles, player_position):
        """
        Determine the best defensive action based on projectile trajectories
        with improved timing and decision making.

        Args:
            projectiles: List of detected projectiles
            player_position: (x, y) tuple of player position

        Returns:
            Recommended action index (0=no-op, 4=jump, 5=crouch)
        """
        if not projectiles:
            # Reset defensive state if no projectiles
            if self.player_state["in_defensive_action"]:
                frames_in_defense = (
                    self.frame_counter - self.player_state["defensive_action_start"]
                )
                # End defensive action after sufficient time has passed
                if frames_in_defense > 10:  # Allow ~10 frames for defensive action
                    self.player_state["in_defensive_action"] = False
            return 0  # No action if no projectiles

        player_x, player_y = player_position
        # Update player state
        self.player_state["position"] = player_position

        # Fix for invalid player y-coordinate
        if player_y <= 10:
            player_y = 100  # Use default

        # Check cooldown for action changes
        current_time = time.time()
        if current_time - self.last_action_time < self.action_cooldown:
            # If in cooldown, maintain last action for stability
            return self.last_recommended_action

        # Debug output
        if len(projectiles) > 0 and self.debug:
            print(
                f"Analyzing {len(projectiles)} projectiles for threat assessment at player pos {player_position}"
            )

        # Filter projectiles by confidence - require higher confidence for action
        credible_projectiles = [p for p in projectiles if p.get("confidence", 0) > 0.5]

        # If already in a defensive action, maintain it for a few frames to complete the move
        if self.player_state["in_defensive_action"]:
            frames_in_defense = (
                self.frame_counter - self.player_state["defensive_action_start"]
            )
            if frames_in_defense < 8:  # Keep defensive action for ~8 frames
                return self.player_state["defensive_action_type"]
            else:
                # End the defensive action
                self.player_state["in_defensive_action"] = False

        # Track threats with collision course and time to impact
        threats = []

        for proj in credible_projectiles:
            x, y = proj["position"]
            vx, vy = proj["velocity"]

            if self.debug:
                print(f"  Projectile at ({x},{y}) with velocity ({vx},{vy})")

            # Skip if velocity is too low - stationary objects
            if abs(vx) < 0.5 and abs(vy) < 0.5:
                continue

            # Skip if not moving toward player horizontally
            if (x < player_x and vx <= 0) or (x > player_x and vx >= 0):
                continue

            # Calculate estimated time to reach player x-coordinate
            time_to_x = abs((player_x - x) / vx) if vx != 0 else float("inf")

            # Only consider projectiles that will reach the player soon
            if 0 < time_to_x <= 20:  # Within next 20 frames
                # Predict y position at collision
                future_y = y + (vy * time_to_x)

                # More accurate player hit box
                player_height = 40  # Player height in pixels
                player_top = player_y - player_height
                player_bottom = player_y

                # Check if projectile will be in player's vertical range
                vertical_distance = min(
                    abs(future_y - player_top), abs(future_y - player_bottom)
                )
                vertical_buffer = 15  # Buffer zone around player

                if vertical_distance <= player_height / 2 + vertical_buffer:
                    # This is a real threat on collision course
                    threats.append(
                        {
                            "projectile": proj,
                            "time_to_impact": time_to_x,
                            "impact_y": future_y,
                            "vertical_distance": vertical_distance,
                        }
                    )

                    if self.debug:
                        print(
                            f"  THREAT DETECTED - will hit in {time_to_x:.1f} frames at y={future_y:.1f}"
                        )

        # Sort threats by time to impact (soonest first)
        threats.sort(key=lambda t: t["time_to_impact"])

        # Determine action based on most immediate threat
        if threats:
            imminent_threat = threats[0]
            threat_projectile = imminent_threat["projectile"]
            time_to_impact = imminent_threat["time_to_impact"]
            impact_y = imminent_threat["impact_y"]

            # Get threat ID for memory
            threat_id = threat_projectile.get("id", -1)

            # Check if we've already reacted to this threat
            if threat_id in self.threat_memory:
                # Already reacting to this threat, maintain action
                return self.last_recommended_action

            # Initiate defensive action earlier for faster projectiles
            vx, vy = threat_projectile["velocity"]
            speed = (vx**2 + vy**2) ** 0.5

            # Action timing windows based on projectile speed
            # Faster projectiles need earlier reaction
            timing_window = 8 if speed < 5 else 15 if speed < 10 else 20

            # Determine if it's time to take action
            if time_to_impact <= timing_window:
                # Determine optimal defensive action
                player_top = player_y - 40  # Top of player
                player_middle = player_y - 20  # Middle of player

                # Record this threat as being handled
                self.threat_memory[threat_id] = self.frame_counter

                # Start defensive action
                self.player_state["in_defensive_action"] = True
                self.player_state["defensive_action_start"] = self.frame_counter

                if impact_y < player_middle:
                    # Projectile will hit upper part of player - crouch
                    action = 5  # Crouch
                    self.player_state["defensive_action_type"] = action
                    if self.debug:
                        print(
                            f"  DEFENSIVE ACTION: CROUCH - projectile will hit upper body"
                        )
                else:
                    # Projectile will hit lower part - jump
                    action = 4  # Jump
                    self.player_state["defensive_action_type"] = action
                    if self.debug:
                        print(
                            f"  DEFENSIVE ACTION: JUMP - projectile will hit lower body"
                        )

                # Update action timestamp
                self.last_action_time = current_time
                self.last_recommended_action = action
                return action

        # Clean up old threats from memory
        current_frame = self.frame_counter
        self.threat_memory = {
            k: v for k, v in self.threat_memory.items() if current_frame - v < 30
        }  # Keep for 30 frames

        # No imminent threats
        self.last_recommended_action = 0
        return 0


def enhance_observation_with_projectiles(current_frame, detector, player_position):
    """
    Process the current frame to detect projectiles and recommend actions.

    Args:
        current_frame: The current game frame
        detector: ProjectileDetector instance
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
