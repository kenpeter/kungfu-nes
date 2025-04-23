import argparse
import pyglet
import sys
import ctypes
import os
import time
import numpy as np
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from kungfu_env import KUNGFU_ACTIONS, KUNGFU_ACTION_NAMES, make_env

# Constants
DEFAULT_GAME_SPEED = 1.0
MODEL_PATH = "models/kungfu_ppo/kungfu_ppo.zip"
VERBOSE = False  # Set to True only when debugging


def map_input_to_discrete_action(inputs):
    """Map keyboard inputs to a discrete action index (0-12) based on KUNGFU_ACTIONS."""
    if VERBOSE:
        print(f"Inputs: {inputs}")
    if not isinstance(inputs, dict):
        raise ValueError(f"Expected inputs to be a dictionary, got {type(inputs)}")
    if inputs.get("DOWN", False) and inputs.get("A", False):
        return 11
    if inputs.get("DOWN", False) and inputs.get("B", False):
        return 12
    if inputs.get("UP", False) and inputs.get("RIGHT", False):
        return 10
    if inputs.get("UP", False):
        return 4
    if inputs.get("DOWN", False):
        return 5
    if inputs.get("LEFT", False):
        return 6
    if inputs.get("RIGHT", False):
        return 7
    if inputs.get("A", False):
        return 8
    if inputs.get("B", False):
        return 1
    if inputs.get("START", False):
        return 3
    if inputs.get("SELECT", False):
        return 2
    return 0


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", help="scenario to use", default="scenario")
    parser.add_argument("--width", type=int, help="window width", default=800)
    parser.add_argument(
        "--speed",
        type=float,
        help="game speed (1.0 is normal)",
        default=DEFAULT_GAME_SPEED,
    )
    parser.add_argument("--model", help="path to trained model", default=MODEL_PATH)
    parser.add_argument(
        "--manual",
        action="store_true",
        help="use manual keyboard controls instead of model",
    )
    parser.add_argument("--verbose", action="store_true", help="enable verbose logging")
    parser.add_argument(
        "--simple",
        action="store_true",
        help="use simple rendering for better performance",
    )
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose

    # Initialize environment
    if VERBOSE:
        print("Creating game environment...")
    env = DummyVecEnv([lambda: make_env()])
    env = VecFrameStack(env, n_stack=4, channels_order="last")

    # Reset environment to get initial state
    obs = env.reset()

    # Input validation
    if not isinstance(obs, dict) or "viewport" not in obs:
        raise ValueError(
            f"Expected obs to be a dictionary with 'viewport' key, got {obs}"
        )

    # Extract screen dimensions
    viewport = obs["viewport"]
    screen_height, screen_width = viewport.shape[1:3]  # Shape is (1, H, W, C*n_stack)
    if VERBOSE:
        print(f"Screen dimensions: {screen_width}x{screen_height}")

    # Load model if not using manual control
    model = None
    if not args.manual:
        if os.path.exists(args.model):
            if VERBOSE:
                print(f"Loading model from {args.model}...")
            model = PPO.load(args.model, env=env)
        else:
            print(f"Model not found at {args.model}, switching to manual mode.")
            args.manual = True

    # Setup window with simple config
    win_width = args.width
    win_height = int(win_width * screen_height / screen_width)
    display = pyglet.canvas.get_display()

    # Try to get a simple config for better performance
    try:
        # config
        config = pyglet.gl.Config(double_buffer=True, sample_buffers=0, samples=0)
        # window
        win = pyglet.window.Window(
            width=win_width,
            height=win_height,
            vsync=False,  # Disable vsync for better performance measurement
            caption="KungFu-Nes",
            config=config,
            resizable=False,
        )
    except pyglet.window.NoSuchConfigException:
        # Fall back to default config
        win = pyglet.window.Window(
            width=win_width,
            height=win_height,
            vsync=False,
            caption="KungFu-Nes",
            resizable=False,
        )

    # Print manual control instructions if in manual mode
    if args.manual:
        print("\nControls:")
        print("- Arrow keys: UP (Jump), DOWN (Crouch), LEFT, RIGHT")
        print("- Z: A (Kick)")
        print("- X: B (Punch)")
        print("- ENTER: START")
        print("- SPACE: SELECT")
        print("- P: Pause/resume")
        print("- ESC: Quit\n")

    # Setup keyboard handler
    key_handler = pyglet.window.key.KeyStateHandler()
    win.push_handlers(key_handler)

    # Setup texture for game display
    texture = pyglet.image.Texture.create(screen_width, screen_height)

    # Pre-allocate buffer for viewport data
    viewport_buffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Game state variables
    paused = False
    last_frame_time = time.time()
    frames = 0
    fps_update_time = time.time()
    current_fps = 0
    last_action = 0
    accumulated_time = 0

    # Scale factor for window
    scale_x = win_width / screen_width
    scale_y = win_height / screen_height

    # Function to update the fps display
    def update_fps():
        nonlocal frames, fps_update_time, current_fps
        current_time = time.time()
        dt = current_time - fps_update_time
        if dt >= 1.0:
            current_fps = frames / dt
            fps_update_time = current_time
            frames = 0
            if VERBOSE:
                print(f"FPS: {current_fps:.1f}")

    # Function to update game state
    def update(dt):
        nonlocal viewport_buffer, paused, last_action, frames, accumulated_time, obs

        # Handle pause toggle
        if key_handler.get(keycodes.P) and not hasattr(update, "p_pressed"):
            update.p_pressed = True
            paused = not paused
            print("Game " + ("paused" if paused else "resumed"))
        elif not key_handler.get(keycodes.P):
            if hasattr(update, "p_pressed"):
                delattr(update, "p_pressed")

        # Skip updates if paused
        if paused:
            return

        # Track frame rate
        frames += 1
        update_fps()

        # Determine action based on mode
        if args.manual:
            inputs = {
                "A": key_handler.get(keycodes.Z, False),
                "B": key_handler.get(keycodes.X, False),
                "UP": key_handler.get(keycodes.UP, False),
                "DOWN": key_handler.get(keycodes.DOWN, False),
                "LEFT": key_handler.get(keycodes.LEFT, False),
                "RIGHT": key_handler.get(keycodes.RIGHT, False),
                "START": key_handler.get(keycodes.ENTER, False),
                "SELECT": key_handler.get(keycodes.SPACE, False),
            }
            discrete_action = map_input_to_discrete_action(inputs)
            action_array = [discrete_action]
            last_action = discrete_action
        else:
            # Only predict every few frames to reduce CPU load
            action_array = [last_action]  # Use previous action by default

            # Only run prediction every 4 frames to reduce load
            if frames % 4 == 0:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    discrete_action = int(action.item())  # Ensure integer action
                    action_array = [discrete_action]
                    last_action = discrete_action
                except Exception as e:
                    if VERBOSE:
                        print(f"Prediction error: {e}")

        # Step environment with chosen action
        try:
            if VERBOSE:
                print(f"Stepping with action: {action_array}")
            obs, reward, done, info = env.step(action_array)

            # Log action and reward for debugging
            if VERBOSE:
                action_name = (
                    KUNGFU_ACTION_NAMES[last_action]
                    if 0 <= last_action < len(KUNGFU_ACTION_NAMES)
                    else "Unknown"
                )
                print(
                    f"Action: {action_name} ({last_action}), Reward: {reward}, Done: {done}"
                )

            # Handle game over/reset
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
            if done:
                obs = env.reset()

            # Validate observation
            if not isinstance(obs, dict) or "viewport" not in obs:
                raise ValueError(f"Invalid observation: {obs}")

            # Extract viewport data efficiently
            viewport = obs["viewport"][0]  # First environment
            viewport_rgb = viewport[:, :, -3:]  # Get last 3 RGB channels

            # Log mean pixel value for debugging
            if VERBOSE:
                print(f"Viewport mean pixel value: {np.mean(viewport_rgb):.2f}")

            # Flip the viewport vertically to correct the orientation
            viewport_rgb = np.flipud(viewport_rgb)

            # Copy directly to our pre-allocated buffer
            np.copyto(viewport_buffer, viewport_rgb)

            # Update texture with new data
            raw_data = (ctypes.c_ubyte * viewport_buffer.nbytes).from_buffer_copy(
                viewport_buffer.tobytes()
            )
            texture.blit_into(
                pyglet.image.ImageData(
                    screen_width, screen_height, "RGB", raw_data, pitch=screen_width * 3
                ),
                0,
                0,
                0,
            )

        except Exception as e:
            if VERBOSE:
                print(f"Environment step error: {e}")
            raise

    # Draw game screen
    @win.event
    def on_draw():
        # Clear window
        win.clear()

        # Simple but fast rendering
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)

        # Draw the texture
        texture.blit(0, 0, width=win_width, height=win_height)

        # Display FPS if verbose
        if VERBOSE:
            fps_text = f"FPS: {current_fps:.1f}"
            pyglet.text.Label(
                fps_text,
                font_name="Arial",
                font_size=14,
                x=10,
                y=10,
                color=(255, 255, 0, 255),
            ).draw()

    # Handle window close
    @win.event
    def on_close():
        pyglet.app.exit()

    # Handle key press for escape to quit
    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == keycodes.ESCAPE:
            win.close()
            return pyglet.event.EVENT_HANDLED

    # Calculate update interval based on speed
    update_interval = 1 / 60.0 / args.speed

    # Schedule game update with appropriate timing
    pyglet.clock.schedule_interval(update, update_interval)

    # Display running mode
    print(f"Running in {'manual' if args.manual else 'AI'} mode at speed {args.speed}x")

    # Main loop
    try:
        pyglet.app.run()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    finally:
        # Clean up
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
