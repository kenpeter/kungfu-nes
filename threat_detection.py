import numpy as np
import cv2
from enum import Enum
from collections import deque
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filename="threat_detection.log",
)
logger = logging.getLogger("threat_detection")

# Setup console handler for important messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)


class ThreatType(Enum):
    TINY = 1  # Small projectiles, knife throws, etc.
    REGULAR = 2  # Normal enemies (including bosses)


class ThreatDirection(Enum):
    LEFT = 1
    RIGHT = 2


class AgentActionType(Enum):
    NO_OP = 0
    PUNCH = 1
    JUMP = 2
    CROUCH = 3
    MOVE_LEFT = 4
    MOVE_RIGHT = 5
    KICK = 6
    PUNCH_KICK = 7
    JUMP_RIGHT = 8
    CROUCH_KICK = 9
    PUNCH_RIGHT = 10
    LEFT_KICK = 11


class Threat:
    def __init__(
        self, threat_id, threat_type, position, frame_time, frame_size=(84, 84)
    ):
        """
        Initialize a threat object

        Args:
            threat_id: Unique identifier for this threat
            threat_type: Type of threat (TINY, REGULAR)
            position: Initial position (x, y)
            frame_time: Time of first detection
            frame_size: Size of the game frame
        """
        self.threat_id = threat_id
        self.threat_type = threat_type
        self.first_seen_time = frame_time
        self.last_seen_time = frame_time

        # Store positions in a queue to calculate velocity
        self.position_history = deque(maxlen=10)
        self.position_history.append(position)

        # Current position
        self.x, self.y = position

        # Derived attributes
        self.velocity = (0, 0)  # (vx, vy)
        self.direction = self._determine_direction(position, frame_size)
        self.priority = 0
        self.distance_to_player = self._calculate_distance_to_player(
            position, frame_size
        )
        self.time_to_collide = float("inf")
        self.recommended_action = AgentActionType.NO_OP
        self.can_attack = self._can_attack(self.threat_type, self.distance_to_player)

        # Visual confidence (0-1) - how certain we are that this is a real threat
        self.visual_confidence = 0.8

        # Depth estimation (z-coordinate) - approximation of how far in the background the threat is
        self.depth = 0

        # Last time this threat was updated
        self.last_update_time = frame_time

        # Time without updates before forgetting this threat
        self.forget_after = 1.0  # seconds

        logger.debug(
            f"New threat detected: ID={threat_id}, Type={threat_type}, Position=({self.x}, {self.y})"
        )

    def update(self, new_position, frame_time, player_position=None):
        """Update threat position and recalculate derived attributes"""
        # Update times
        self.last_seen_time = frame_time
        time_delta = frame_time - self.last_update_time
        self.last_update_time = frame_time

        # Update position history
        self.position_history.append(new_position)
        self.x, self.y = new_position

        # Calculate velocity
        if len(self.position_history) >= 2 and time_delta > 0:
            old_x, old_y = self.position_history[-2]
            vx = (self.x - old_x) / time_delta
            vy = (self.y - old_y) / time_delta
            self.velocity = (vx, vy)

        # Update other attributes
        frame_size = (84, 84)  # Default frame size
        self.direction = self._determine_direction(new_position, frame_size)

        if player_position:
            self.distance_to_player = self._calculate_distance_to_player(
                new_position, frame_size, player_position
            )
        else:
            self.distance_to_player = self._calculate_distance_to_player(
                new_position, frame_size
            )

        self.time_to_collide = self._calculate_time_to_collide()
        self.can_attack = self._can_attack(self.threat_type, self.distance_to_player)
        self.priority = self._determine_priority()
        self.recommended_action = self._determine_recommended_action()

        logger.debug(
            f"Updated threat: ID={self.threat_id}, Position=({self.x}, {self.y}), Priority={self.priority}"
        )

    def _determine_direction(self, position, frame_size):
        """Determine which direction the threat is coming from"""
        x, y = position
        center_x = frame_size[0] // 2

        if x < center_x:
            return ThreatDirection.LEFT
        else:
            return ThreatDirection.RIGHT

    def _calculate_distance_to_player(self, position, frame_size, player_position=None):
        """Calculate the distance to the player character"""
        x, y = position

        # If player position is provided, use it
        if player_position:
            player_x, player_y = player_position
        else:
            # Otherwise assume player is at the center of the screen
            player_x, player_y = frame_size[0] // 2, frame_size[1] // 2

        # Euclidean distance
        return np.sqrt((x - player_x) ** 2 + (y - player_y) ** 2)

    def _calculate_time_to_collide(self):
        """Calculate estimated time until collision with player"""
        if abs(self.velocity[0]) < 0.1:
            return float("inf")  # Not moving horizontally

        # Estimate distance to collision
        if self.direction == ThreatDirection.LEFT:
            # Threat is on the left, moving right
            if self.velocity[0] > 0:
                distance = 42 - self.x  # Assuming player is at x=42 (center)
                return max(0, distance / self.velocity[0])
        elif self.direction == ThreatDirection.RIGHT:
            # Threat is on the right, moving left
            if self.velocity[0] < 0:
                distance = self.x - 42  # Assuming player is at x=42 (center)
                return max(0, distance / -self.velocity[0])

        return float("inf")  # No collision imminent

    def _can_attack(self, threat_type, distance):
        """Determine if the threat is in range to attack"""
        if threat_type == ThreatType.TINY:
            return False  # Tiny threats (projectiles) can't be attacked

        # Regular enemies can be attacked if they're close enough
        return distance < 30

    def _determine_priority(self):
        """Calculate threat priority based on distance, collision time, and type"""
        priority = 0

        # Base priority by threat type
        if self.threat_type == ThreatType.REGULAR:
            # Check if it's a larger enemy (like a boss) based on size
            if self.y > 70:  # Bosses are usually taller/larger
                priority = 100
            else:
                priority = 50
        elif self.threat_type == ThreatType.TINY:
            priority = 30

        # Adjust by distance - closer is higher priority
        if self.distance_to_player < 20:
            priority += 50
        elif self.distance_to_player < 40:
            priority += 30
        elif self.distance_to_player < 60:
            priority += 10

        # Adjust by collision time - sooner is higher priority
        if self.time_to_collide < 0.5:
            priority += 40
        elif self.time_to_collide < 1.0:
            priority += 20

        # Adjust by visual confidence
        priority *= self.visual_confidence

        return priority

    def _determine_recommended_action(self):
        """Determine the best action to take against this threat"""
        # Default to NO_OP
        action = AgentActionType.NO_OP

        # Handle based on threat type and direction
        if self.threat_type == ThreatType.TINY:
            # For projectiles, dodge based on height
            if self.y < 50:  # Projectile is high, crouch
                action = AgentActionType.CROUCH
            else:  # Projectile is low, jump
                action = AgentActionType.JUMP

        elif self.threat_type == ThreatType.REGULAR:
            # For enemies, attack if in range
            if self.can_attack:
                if self.direction == ThreatDirection.LEFT:
                    action = AgentActionType.LEFT_KICK
                elif self.direction == ThreatDirection.RIGHT:
                    action = AgentActionType.PUNCH_RIGHT
            else:
                # Not in range, move toward enemy
                if self.direction == ThreatDirection.LEFT:
                    action = AgentActionType.MOVE_LEFT
                elif self.direction == ThreatDirection.RIGHT:
                    action = AgentActionType.MOVE_RIGHT

        return action

    def is_stale(self, current_time):
        """Check if the threat hasn't been updated recently and should be forgotten"""
        return current_time - self.last_seen_time > self.forget_after


class ThreatDetection:
    def __init__(self, frame_size=(84, 84)):
        """
        Initialize threat detection system

        Args:
            frame_size: Size of the game frame (height, width)
        """
        self.frame_size = frame_size
        self.threats = {}  # Dictionary of active threats
        self.next_threat_id = 0

        # Timing parameters
        self.reaction_time = 0.15  # seconds
        self.execution_time = 0.1  # seconds

        # Visual processing parameters
        self.motion_history = deque(maxlen=4)  # Keep a few frames to detect motion

        # Initialize background model with proper dimensions and type
        self.background_model = np.zeros(frame_size, dtype=np.float32)

        # Prepare for CV operations
        self.foreground_mask = None
        self.last_frame = None

        # Matching distance threshold
        self.matching_threshold = 15

        logger.info("Threat detection system initialized")

    def process_frame(self, frame, player_position=None, current_time=None):
        """
        Process a new frame to detect and track threats

        Args:
            frame: The current game frame (numpy array)
            player_position: (x, y) position of the player character
            current_time: Current game time

        Returns:
            highest_priority_threat: The most important threat to address
            all_threats: Dictionary of all active threats
        """
        if current_time is None:
            current_time = time.time()

        # Store frame in motion history
        self.motion_history.append(frame)

        # Process visual threats
        detected_positions = self._process_visual_threats(frame)

        # Match detections with existing threats or create new ones
        self._update_threats(detected_positions, current_time, player_position)

        # Remove stale threats
        self._remove_stale_threats(current_time)

        # Determine the highest priority threat
        highest_priority_threat = self._get_highest_priority_threat()

        return highest_priority_threat, self.threats

    def _process_visual_threats(self, frame):
        """Process frame to detect visual threats using computer vision techniques"""
        try:
            # Convert to grayscale if needed
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            elif len(frame.shape) == 3 and frame.shape[2] == 1:
                gray = frame.squeeze()
            else:
                gray = frame

            # Ensure gray is 2D
            if len(gray.shape) > 2:
                gray = gray.squeeze()

            # Resize if needed
            if gray.shape != self.frame_size:
                gray = cv2.resize(gray, (self.frame_size[1], self.frame_size[0]))

            # Ensure background model matches gray's shape and type
            if (
                self.background_model is None
                or self.background_model.shape != gray.shape
            ):
                logger.info(f"Reinitializing background model to shape {gray.shape}")
                self.background_model = np.zeros_like(gray, dtype=np.float32)

            # Update background model with slow adaptation
            # Convert gray to float32 before accumulation
            gray_float = gray.astype(np.float32)
            cv2.accumulateWeighted(gray_float, self.background_model, 0.05)

            # Convert back to uint8 for further processing
            bg_model_uint8 = self.background_model.astype(np.uint8)

            # Calculate absolute difference
            foreground = cv2.absdiff(gray, bg_model_uint8)

            # Threshold to get binary mask
            _, mask = cv2.threshold(foreground, 15, 255, cv2.THRESH_BINARY)

            # Apply morphological operations to remove noise
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Process contours to get threat positions
            detected_positions = []
            for contour in contours:
                # Filter small contours (noise)
                if cv2.contourArea(contour) < 10:
                    continue

                # Get the bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate centroid
                center_x = x + w // 2
                center_y = y + h // 2

                # Determine threat type based on size
                threat_type = ThreatType.TINY if (w * h) < 50 else ThreatType.REGULAR

                detected_positions.append(
                    {
                        "position": (center_x, center_y),
                        "type": threat_type,
                        "size": (w, h),
                    }
                )

            return detected_positions

        except Exception as e:
            logger.error(f"Error in _process_visual_threats: {e}")
            logger.error(
                f"Frame shape: {frame.shape if hasattr(frame, 'shape') else 'Unknown'}"
            )
            if hasattr(gray, "shape"):
                logger.error(f"Gray shape: {gray.shape}")
            if hasattr(self.background_model, "shape"):
                logger.error(f"Background model shape: {self.background_model.shape}")

            # Return empty list on error
            return []

    def _update_threats(self, detected_positions, current_time, player_position):
        """Match detected positions with existing threats or create new ones"""
        # Track which threats were updated this frame
        updated_threats = set()

        for detection in detected_positions:
            position = detection["position"]
            threat_type = detection["type"]

            # Try to match with existing threats
            matched = False
            for threat_id, threat in self.threats.items():
                # Calculate distance to existing threat
                distance = np.sqrt(
                    (threat.x - position[0]) ** 2 + (threat.y - position[1]) ** 2
                )

                # If close enough, update the existing threat
                if distance < self.matching_threshold:
                    threat.update(position, current_time, player_position)
                    updated_threats.add(threat_id)
                    matched = True
                    break

            # If no match found, create a new threat
            if not matched:
                new_threat = Threat(
                    threat_id=self.next_threat_id,
                    threat_type=threat_type,
                    position=position,
                    frame_time=current_time,
                )
                self.threats[self.next_threat_id] = new_threat
                updated_threats.add(self.next_threat_id)
                self.next_threat_id += 1

        # Mark threats that weren't updated as having decreased confidence
        for threat_id, threat in self.threats.items():
            if threat_id not in updated_threats:
                # Decrease visual confidence when not detected
                threat.visual_confidence *= 0.8

    def _remove_stale_threats(self, current_time):
        """Remove threats that haven't been seen for a while"""
        to_remove = []
        for threat_id, threat in self.threats.items():
            if threat.is_stale(current_time) or threat.visual_confidence < 0.2:
                to_remove.append(threat_id)

        for threat_id in to_remove:
            logger.debug(f"Removing stale threat: ID={threat_id}")
            del self.threats[threat_id]

    def _get_highest_priority_threat(self):
        """Find the threat with the highest priority"""
        if not self.threats:
            return None

        highest_priority = -1
        highest_threat = None

        for threat_id, threat in self.threats.items():
            if threat.priority > highest_priority:
                highest_priority = threat.priority
                highest_threat = threat

        return highest_threat

    def get_recommended_action(self):
        """Get the recommended action based on the highest priority threat"""
        highest_threat = self._get_highest_priority_threat()
        if highest_threat:
            return highest_threat.recommended_action
        else:
            return AgentActionType.NO_OP

    def visualize_threats(self, frame):
        """Draw threats on the frame for visualization"""
        if len(frame.shape) == 2:  # Grayscale
            vis_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            vis_frame = frame.copy()

        # Draw each threat
        for threat_id, threat in self.threats.items():
            # Color based on threat type
            if threat.threat_type == ThreatType.REGULAR:
                color = (0, 255, 0)  # Green for regular enemies
            else:
                color = (255, 255, 0)  # Yellow for tiny threats

            # Draw circle at threat position
            cv2.circle(
                vis_frame,
                (int(threat.x), int(threat.y)),
                radius=int(5 * threat.visual_confidence),
                color=color,
                thickness=2,
            )

            # Draw priority text
            cv2.putText(
                vis_frame,
                f"{threat.priority:.1f}",
                (int(threat.x - 10), int(threat.y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
            )

        # Draw highest priority threat recommendation
        highest_threat = self._get_highest_priority_threat()
        if highest_threat:
            action_text = highest_threat.recommended_action.name
            cv2.putText(
                vis_frame,
                action_text,
                (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
            )

        return vis_frame
