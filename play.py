import pyglet
import numpy as np
import retro
import time
import cv2
import threading
import queue
from pyglet.gl import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from kungfu_env import KungFuWrapper, SimpleCNN, N_STACK
import logging

# Configure logging - reduce verbosity
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Create a filter to avoid repeated log messages
class DuplicateFilter(logging.Filter):
    def __init__(self):
        super().__init__()
        self.last_log = None
        self.repeat_count = 0

    def filter(self, record):
        current = record.getMessage()
        if current == self.last_log:
            self.repeat_count += 1
            if self.repeat_count >= 5:  # Only log every 5th duplicate
                if self.repeat_count % 5 == 0:
                    return True
                return False
        else:
            self.last_log = current
            self.repeat_count = 0
            return True


# Apply filter to logger
logger.addFilter(DuplicateFilter())

# Constants
MODEL_PATH = "models/kungfu_ppo/kungfu_ppo.zip"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
FRAME_RATE_TARGET = 20.0
NES_DEFAULT_WIDTH = 256
NES_DEFAULT_HEIGHT = 240


def analyze_observation(obs):
    """Analyze an observation and print detailed information about its structure"""
    print("\n===== OBSERVATION ANALYSIS =====")

    if isinstance(obs, dict):
        print(f"Observation is a dictionary with {len(obs)} keys: {list(obs.keys())}")

        for key, value in obs.items():
            print(f"\nKey: '{key}'")
            if isinstance(value, np.ndarray):
                print(f"  Shape: {value.shape}")
                print(f"  Data type: {value.dtype}")
                print(f"  Min/Max values: {value.min()}/{value.max()}")

                # For viewport, analyze more deeply
                if key == "viewport" and len(value.shape) >= 3:
                    batch_size = value.shape[0]
                    channels = (
                        value.shape[1] if len(value.shape) > 3 else value.shape[0]
                    )

                    print(f"  Viewport details:")
                    print(f"    Batch size: {batch_size}")
                    print(f"    Channels: {channels}")

                    # If it's likely stacked frames (divisible by 3 for RGB)
                    if channels % 3 == 0:
                        frames = channels // 3
                        print(f"    Possible RGB frames: {frames}")

                        # Sample different parts of the viewport to better understand structure
                        if channels >= 9:  # At least 3 RGB frames
                            print(
                                "\n    Channel stats (sampling first/middle/last RGB frame):"
                            )
                            # First RGB frame (channels 0-2)
                            first_frame = (
                                value[0, :3] if len(value.shape) > 3 else value[:3]
                            )
                            print(
                                f"      First frame (ch 0-2): min={first_frame.min()}, max={first_frame.max()}"
                            )

                            # Middle RGB frame
                            mid_idx = (frames // 2) * 3
                            mid_frame = (
                                value[0, mid_idx : mid_idx + 3]
                                if len(value.shape) > 3
                                else value[mid_idx : mid_idx + 3]
                            )
                            print(
                                f"      Middle frame (ch {mid_idx}-{mid_idx+2}): min={mid_frame.min()}, max={mid_frame.max()}"
                            )

                            # Last RGB frame
                            last_idx = (frames - 1) * 3
                            last_frame = (
                                value[0, last_idx : last_idx + 3]
                                if len(value.shape) > 3
                                else value[last_idx : last_idx + 3]
                            )
                            print(
                                f"      Last frame (ch {last_idx}-{last_idx+2}): min={last_frame.min()}, max={last_frame.max()}"
                            )
            else:
                print(f"  Type: {type(value)}")
                try:
                    print(f"  Length: {len(value)}")
                except:
                    print(f"  Value: {value}")
    else:
        print(f"Observation is a {type(obs)}")
        if isinstance(obs, np.ndarray):
            print(f"Shape: {obs.shape}, dtype: {obs.dtype}")
            print(f"Min/Max values: {obs.min()}/{obs.max()}")

    print("================================\n")


class GameThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.action = 0
        self.obs = None
        self.rewards = None
        self.dones = None
        self.infos = None
        self.screen = None
        self.env = None
        self.model = None
        self.initialized = False
        self.error = None

    def run(self):
        try:
            # Initialize environment
            logger.info("Initializing environment in background thread")

            # Create environment
            def make_kungfu_env_for_vec():
                base_env = retro.make(
                    "KungFu-Nes",
                    use_restricted_actions=retro.Actions.ALL,
                    render_mode="rgb_array",
                )
                return KungFuWrapper(base_env, n_stack=N_STACK)

            # Create the environment stack
            self.env = DummyVecEnv([make_kungfu_env_for_vec])
            self.env = VecFrameStack(self.env, n_stack=N_STACK)
            self.env = VecTransposeImage(self.env)

            # Reset environment
            self.obs = self.env.reset()

            # Log observation shapes for debugging
            if isinstance(self.obs, dict):
                for key, value in self.obs.items():
                    logger.info(f"Observation key: {key}, shape: {value.shape}")
            else:
                logger.info(f"Observation shape: {self.obs.shape}")

            # Load model
            logger.info(f"Loading model from {MODEL_PATH}")
            custom_objects = {
                "features_extractor_class": SimpleCNN,
                "features_extractor_kwargs": {"features_dim": 256},
            }
            self.model = PPO.load(MODEL_PATH, custom_objects=custom_objects)
            logger.info("Model loaded successfully")

            # Debug the first observation
            print("\nAnalyzing initial observation structure:")
            analyze_observation(self.obs)

            # Get initial screen - try multiple methods
            self.get_screen()

            self.initialized = True
            logger.info("Game thread initialization complete")

            # Main game loop
            frame_count = 0
            while self.running:
                try:
                    # Get action from model
                    action, _ = self.model.predict(self.obs, deterministic=True)
                    self.action = action[0]

                    # Step environment
                    self.obs, self.rewards, self.dones, self.infos = self.env.step(
                        [self.action]
                    )

                    # Get updated screen
                    self.get_screen()

                    # Log progress occasionally
                    frame_count += 1
                    if frame_count % 100 == 0:
                        if isinstance(self.infos, list) and len(self.infos) > 0:
                            info = self.infos[0]
                            # Extract interesting info if available
                            hp = info.get("hp", 0)
                            reward = info.get("normalized_reward", 0)
                            logger.info(
                                f"Frame {frame_count}: HP={hp}, Reward={reward:.2f}, Action={self.action}"
                            )

                    # Reset if episode is done
                    if self.dones[0]:
                        logger.info("Episode finished - resetting environment")
                        self.obs = self.env.reset()

                    # Sleep to maintain reasonable frame rate
                    time.sleep(1.0 / FRAME_RATE_TARGET)

                except Exception as e:
                    logger.error(f"Error in game loop: {e}")
                    time.sleep(0.1)  # Avoid tight error loop

            # Clean up when thread ends
            if self.env:
                self.env.close()

        except Exception as e:
            logger.error(f"Fatal error in game thread: {e}")
            self.error = str(e)
            self.running = False

    def get_screen(self):
        """Get the game screen from the observation viewport"""
        try:
            if isinstance(self.obs, dict) and "viewport" in self.obs:
                viewport = self.obs["viewport"][0]  # (160, 160, 12)（NHWC）
                if viewport.shape[-1] == 12:
                    # 最后一帧的通道索引：3*(n_stack-1) 到 3*n_stack，即 9, 10, 11（0-based）
                    r = viewport[..., 9]
                    g = viewport[..., 10]
                    b = viewport[..., 11]
                    rgb_image = np.stack([r, g, b], axis=-1)  # (160, 160, 3)
                    self.screen = rgb_image.astype(np.uint8)
                    return

                # If the viewport doesn't match expected format, try more general approach
                logger.warning(f"Viewport has unexpected shape: {viewport.shape}")

                # If channels are first dimension and there are at least 3 channels
                if len(viewport.shape) == 3 and viewport.shape[0] >= 3:
                    # Take the first 3 channels as RGB
                    rgb_channels = viewport[:3]
                    # Convert from (C,H,W) to (H,W,C)
                    rgb_image = np.transpose(rgb_channels, (1, 2, 0))

                    # Normalize to 0-255 if needed
                    if rgb_image.max() <= 1.0 and rgb_image.max() > 0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    elif rgb_image.dtype != np.uint8:
                        rgb_image = rgb_image.astype(np.uint8)

                    self.screen = rgb_image
                    return

            # If we couldn't extract viewport data, fall back to the render method
            try:
                raw_screen = self.env.render(mode="rgb_array")[0]

                # If we get a proper image format, use it
                if len(raw_screen.shape) == 3 and raw_screen.shape[2] == 3:
                    self.screen = raw_screen
                    return

                # If we have the problematic (240, 3) format
                if raw_screen.shape == (240, 3):
                    logger.info("Got (240, 3) format - creating blank screen instead")
                    # Instead of trying to reshape this problematic format,
                    # we'll create a blank screen since we've already tried the viewport approach
                    self.screen = np.zeros(
                        (NES_DEFAULT_HEIGHT, NES_DEFAULT_WIDTH, 3), dtype=np.uint8
                    )
                    return

            except Exception as render_err:
                logger.error(f"Render failed: {render_err}")

            # Last resort: create a blank screen
            self.screen = np.zeros(
                (NES_DEFAULT_HEIGHT, NES_DEFAULT_WIDTH, 3), dtype=np.uint8
            )

        except Exception as e:
            logger.error(f"Screen extraction failed: {e}")
            # Create a blank screen as fallback
            self.screen = np.zeros(
                (NES_DEFAULT_HEIGHT, NES_DEFAULT_WIDTH, 3), dtype=np.uint8
            )


def main():
    print("Initializing Kung Fu environment with AI agent...")

    # Create game thread
    game_thread = GameThread()
    game_thread.start()

    # Wait for initialization
    while not game_thread.initialized and game_thread.running:
        if game_thread.error:
            print(f"Initialization error: {game_thread.error}")
            return
        print("Waiting for game initialization...")
        time.sleep(0.5)

    if not game_thread.running:
        print("Game thread stopped unexpectedly")
        return

    # Get initial screen shape
    while game_thread.screen is None and game_thread.running:
        time.sleep(0.1)

    if not game_thread.running:
        print("Game thread stopped before rendering first frame")
        return

    # Get screen dimensions
    screen = game_thread.screen
    print(f"Initial screen shape: {screen.shape}")

    # Debug initial state
    def debug_initial_state():
        """Debug initial game state"""
        # Debug screen information
        if game_thread.screen is not None:
            screen = game_thread.screen
            print(f"\nInitial screen info:")
            print(f"  Shape: {screen.shape}")
            print(f"  dtype: {screen.dtype}")
            print(f"  min/max: {screen.min()}/{screen.max()}")

            if screen.max() == 0:
                print("  WARNING: Screen appears to be all black!")

            # Try viewport visualization if screen is black
            if (
                screen.max() == 0
                and isinstance(game_thread.obs, dict)
                and "viewport" in game_thread.obs
            ):
                print("\nAttempting to visualize viewport directly...")
                viewport = game_thread.obs["viewport"][0]
                print(f"  Viewport shape: {viewport.shape}")

                # For 48-channel viewport (expected format)
                if viewport.shape[0] == 48:
                    print(
                        "  Standard viewport format detected, extracting RGB frame..."
                    )

                    # Create a test image from channels 9-11 (assuming this is an RGB frame)
                    test_img = np.stack(
                        [viewport[9], viewport[10], viewport[11]], axis=2
                    )
                    if test_img.max() <= 1.0 and test_img.max() > 0:
                        test_img = (test_img * 255).astype(np.uint8)

                    print(
                        f"  Test image shape: {test_img.shape}, min/max: {test_img.min()}/{test_img.max()}"
                    )

                    # Save test image to file for inspection
                    try:
                        import cv2

                        cv2.imwrite(
                            "viewport_test.png",
                            cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR),
                        )
                        print("  Saved test image to 'viewport_test.png'")
                    except Exception as e:
                        print(f"  Could not save test image: {e}")

    # Call debug function
    debug_initial_state()

    # Ensure we have a valid screen with proper dimensions
    if len(screen.shape) != 3 or screen.shape[2] != 3:
        print(f"Invalid screen format: {screen.shape}, using NES defaults")
        screen_height, screen_width = NES_DEFAULT_HEIGHT, NES_DEFAULT_WIDTH
    else:
        screen_height, screen_width = screen.shape[:2]

    print(f"Using dimensions: {screen_width}x{screen_height}")

    # Calculate window dimensions to maintain aspect ratio
    win_height = int(WINDOW_WIDTH * screen_height / screen_width)

    # Create window
    win = pyglet.window.Window(
        width=WINDOW_WIDTH,
        height=win_height,
        caption="KungFu-Nes AI Player",
    )

    # Create texture for rendering
    texture = pyglet.image.Texture.create(screen_width, screen_height)

    # Create rendering buffer - ensure 3 channels for RGB
    buffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    print(f"Buffer created with shape: {buffer.shape}")

    # Handle key press - only ESC to exit
    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()

    fps_display = pyglet.window.FPSDisplay(window=win)
    frame_counter = 0
    last_frame_time = time.time()
    last_render_time = time.time()

    # Update function - now only updates the UI
    def update(dt):
        nonlocal frame_counter, last_frame_time, last_render_time
        frame_counter += 1

        # FPS calculation
        if frame_counter % 60 == 0:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            fps = 60 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}")
            last_frame_time = current_time

        # Get the latest screen from the game thread
        if game_thread.screen is not None:
            # Only update texture if enough time has passed (frame rate limiting)
            current_time = time.time()
            if current_time - last_render_time >= 1.0 / FRAME_RATE_TARGET:
                # Copy to buffer for display
                try:
                    # Get the latest screen from the game thread
                    screen = game_thread.screen

                    # Check if screen is blank (all zeros)
                    if (
                        screen.max() == 0
                        and isinstance(game_thread.obs, dict)
                        and "viewport" in game_thread.obs
                    ):
                        # Try to use viewport data instead
                        viewport = game_thread.obs["viewport"][0]

                        # Extract RGB frame from viewport (channels 45-47 for the frame with content)
                        if viewport.shape[0] == 48:
                            screen = np.stack(
                                [viewport[45], viewport[46], viewport[47]], axis=2
                            )
                            if screen.dtype != np.uint8:
                                screen = screen.astype(np.uint8)

                    # Debug print occasionally to see if we're getting valid screen data
                    if frame_counter % 300 == 0:
                        screen_stats = f"Screen shape: {screen.shape}, min: {screen.min()}, max: {screen.max()}"
                        print(screen_stats)
                        if screen.max() == 0:
                            print("WARNING: Screen still showing all black")

                    # Make sure screen has the right format
                    if screen.shape == buffer.shape:
                        np.copyto(buffer, screen)
                    else:
                        # Handle differently shaped screens
                        if len(screen.shape) == 3 and screen.shape[2] == 3:
                            # Resize to match buffer dimensions
                            resized = cv2.resize(
                                screen, (buffer.shape[1], buffer.shape[0])
                            )
                            np.copyto(buffer, resized)
                        else:
                            # Create blank image if formats don't match
                            buffer.fill(0)  # Clear buffer
                except Exception as e:
                    print(f"Error copying screen to buffer: {e}")

                # Update texture
                try:
                    image_data = pyglet.image.ImageData(
                        buffer.shape[1],  # width
                        buffer.shape[0],  # height
                        "RGB",
                        np.flipud(buffer).tobytes(),  # Flip vertically for OpenGL
                        pitch=buffer.shape[1] * 3,
                    )
                    texture.blit_into(image_data, 0, 0, 0)
                    last_render_time = current_time
                except Exception as e:
                    print(f"Error creating texture: {e}")

    # Draw function
    @win.event
    def on_draw():
        win.clear()
        # Use nearest neighbor filtering for pixelated look
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        texture.blit(0, 0, width=WINDOW_WIDTH, height=win_height)

        # Draw mode and action text
        info_text = pyglet.text.Label(
            f"AI Playing - Action: {game_thread.action} | Press ESC to exit",
            font_name="Arial",
            font_size=14,
            x=10,
            y=win_height - 10,
            color=(255, 255, 255, 255),
            anchor_x="left",
            anchor_y="top",
        )
        info_text.draw()

        fps_display.draw()

    # Schedule update with a consistent frame rate
    pyglet.clock.schedule_interval(update, 1.0 / FRAME_RATE_TARGET)

    # Run the application
    print("Running AI player. Press ESC to exit.")

    try:
        pyglet.app.run()
    finally:
        # Clean up
        game_thread.running = False
        game_thread.join(timeout=2.0)
        print("Game terminated")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
