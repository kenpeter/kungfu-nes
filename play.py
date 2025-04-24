import pyglet
import numpy as np
import retro  # stable retro, not open ai retro
import time
import cv2
from pyglet.gl import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from kungfu_env import KungFuWrapper, SimpleCNN, N_STACK

# Constants
MODEL_PATH = "models/kungfu_ppo/kungfu_ppo.zip"
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480


def main():
    print("Initializing Kung Fu environment...")

    # Define the environment creation function
    def make_kungfu_env_for_vec():
        base_env = retro.make(
            "KungFu-Nes",
            use_restricted_actions=retro.Actions.ALL,
            render_mode="rgb_array",
        )
        return KungFuWrapper(base_env, n_stack=N_STACK)

    # Create the environment only once through DummyVecEnv
    env = DummyVecEnv([make_kungfu_env_for_vec])
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)

    # First reset to get initial observation
    obs = env.reset()

    # Debug observation shape - this is helpful
    print(f"Observation type: {type(obs)}")
    if isinstance(obs, dict):
        for key, value in obs.items():
            print(f"Observation key: {key}, shape: {value.shape}")
    else:
        print(f"Observation shape: {obs.shape}")

    # Get the internal environment from VecEnv
    # Let's print the wrapper hierarchy to debug
    current_env = env.envs[0]
    print(f"Environment type: {type(current_env)}")

    # Navigate through wrappers until we find RetroEnv
    # this is go down
    while hasattr(current_env, "env"):
        current_env = current_env.env
        print(f"Unwrapped to: {type(current_env)}")

    # The last unwrapped environment should be the RetroEnv
    base_env = current_env

    # Get initial screen shape from raw environment render
    screen = base_env.render()
    screen_height, screen_width = screen.shape[:2]

    # Calculate window dimensions to maintain aspect ratio
    win_height = int(WINDOW_WIDTH * screen_height / screen_width)

    # Create window
    win = pyglet.window.Window(
        width=WINDOW_WIDTH,
        height=win_height,
        caption="KungFu-Nes Direct Render",
    )

    # Create texture for rendering
    texture = pyglet.image.Texture.create(screen_width, screen_height)

    # Create rendering buffer
    buffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Variables for game state
    action = 0  # Start with no-op

    try:
        # Try to load the model
        print("Attempting to load model from", MODEL_PATH)
        custom_objects = {
            "features_extractor_class": SimpleCNN,
            "features_extractor_kwargs": {"features_dim": 256},
        }
        model = PPO.load(MODEL_PATH, custom_objects=custom_objects)
        print("Model loaded successfully")
        print(f"Playing with n_stack={N_STACK}")
        ai_mode = True
    except Exception as e:
        print(f"Couldn't load model: {e}")
        print("Running in manual mode")
        ai_mode = False

    # Keys for manual control
    keys = {
        pyglet.window.key.LEFT: 6,  # LEFT
        pyglet.window.key.RIGHT: 7,  # RIGHT
        pyglet.window.key.UP: 4,  # UP (Jump)
        pyglet.window.key.DOWN: 5,  # DOWN (Crouch)
        pyglet.window.key.Z: 1,  # B (Punch)
        pyglet.window.key.X: 8,  # A (Kick)
        pyglet.window.key.SPACE: 9,  # B + A (Punch + Kick)
    }

    key_pressed = set()

    # Handle key press
    @win.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol in keys:
            key_pressed.add(symbol)

    # Handle key release
    @win.event
    def on_key_release(symbol, modifiers):
        if symbol in key_pressed:
            key_pressed.remove(symbol)

    # Toggle AI mode with updated event handler
    @win.event
    def on_key_press(symbol, modifiers):
        nonlocal ai_mode
        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
        elif symbol in keys:
            key_pressed.add(symbol)
        elif symbol == pyglet.window.key.TAB:
            ai_mode = not ai_mode
            print(f"Switched to {'AI' if ai_mode else 'Manual'} mode")

    fps_display = pyglet.window.FPSDisplay(window=win)
    frame_counter = 0
    last_frame_time = time.time()
    frame_rate_target = 30.0  # Target frame rate

    # Display text for visible model state
    status_text = None

    # Update function
    def update(dt):
        nonlocal action, obs, frame_counter, last_frame_time, status_text
        frame_counter += 1

        # FPS calculation
        if frame_counter % 60 == 0:
            current_time = time.time()
            elapsed = current_time - last_frame_time
            fps = 60 / elapsed if elapsed > 0 else 0
            print(f"FPS: {fps:.1f}")
            last_frame_time = current_time

        # Determine action
        if ai_mode:
            # Get model prediction using vectorized observation
            action, _ = model.predict(obs, deterministic=True)
            # Convert from array to scalar for the environment
            action = action[0]
        else:
            # Use keyboard input
            if key_pressed:
                for key in key_pressed:
                    if key in keys:
                        action = keys[key]
                        break
            else:
                action = 0  # No-op

        # Step environment with chosen action
        obs, rewards, dones, infos = env.step([action])

        # Get raw screen from unwrapped env for rendering
        screen = base_env.render()

        # Copy to buffer for display
        if screen.shape == buffer.shape:
            np.copyto(buffer, screen)
        else:
            print(
                f"Warning: Shape mismatch - buffer: {buffer.shape}, screen: {screen.shape}"
            )
            # Handle resize if needed
            resized = cv2.resize(screen, (screen_width, screen_height))
            np.copyto(buffer, resized)

        # Update texture
        image_data = pyglet.image.ImageData(
            screen_width,
            screen_height,
            "RGB",
            np.flipud(buffer).tobytes(),  # Flip vertically for OpenGL
            pitch=screen_width * 3,
        )
        texture.blit_into(image_data, 0, 0, 0)

        # Update status text
        status_text = pyglet.text.Label(
            f"Mode: {'AI' if ai_mode else 'Manual'} | Action: {action} | N_Stack: {N_STACK}",
            font_name="Arial",
            font_size=12,
            x=10,
            y=win_height - 30,
            color=(255, 255, 0, 255),
        )

        # Reset if done
        if dones[0]:
            obs = env.reset()
            print("Episode finished - resetting environment")

    # Draw function
    @win.event
    def on_draw():
        win.clear()
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        texture.blit(0, 0, width=WINDOW_WIDTH, height=win_height)

        # Draw mode text
        mode_text = pyglet.text.Label(
            f"{'AI' if ai_mode else 'Manual'} Mode - Press TAB to toggle",
            font_name="Arial",
            font_size=16,
            x=10,
            y=win_height - 10,
            color=(255, 255, 255, 255),
            anchor_x="left",
            anchor_y="top",
        )
        mode_text.draw()
        if status_text:
            status_text.draw()
        fps_display.draw()

    # Schedule update with a consistent frame rate
    pyglet.clock.schedule_interval(update, 1.0 / frame_rate_target)

    # Run the application
    print(
        f"Running in {'AI' if ai_mode else 'Manual'} mode. Press TAB to toggle. Press ESC to exit."
    )
    pyglet.app.run()

    # Cleanup
    env.close()
    base_env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
