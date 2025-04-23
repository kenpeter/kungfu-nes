import pyglet
import numpy as np
from pyglet import clock
from pyglet.gl import *
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from kungfu_env import KUNGFU_ACTION_NAMES, make_env

# Constants
MODEL_PATH = "models/kungfu_ppo/kungfu_ppo.zip"
WINDOW_WIDTH = 400
GAME_SPEED = 1.0


def main():
    # Initialize environment
    env = DummyVecEnv([lambda: make_env()])
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    obs = env.reset()

    # Validate observation
    if not isinstance(obs, dict) or "viewport" not in obs:
        raise ValueError("Invalid observation")

    # Extract screen dimensions
    viewport = obs["viewport"]
    screen_height, screen_width = viewport.shape[1:3]

    # Load model
    model = PPO.load(MODEL_PATH, env=env)

    # Setup window
    win_height = int(WINDOW_WIDTH * screen_height / screen_width)
    win = pyglet.window.Window(
        width=WINDOW_WIDTH,
        height=win_height,
        vsync=False,
        caption="KungFu-Nes",
        resizable=False,
    )

    # Setup texture
    texture = pyglet.image.Texture.create(screen_width, screen_height)
    viewport_buffer = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)

    # Game state
    last_action = 0
    frame_count = 0  # Manual frame counter

    # Update game state
    def update(dt):
        nonlocal obs, last_action, frame_count

        frame_count += 1  # Increment frame counter

        # Predict action every 4 frames
        if frame_count % 4 == 0:
            action, _ = model.predict(obs, deterministic=True)
            last_action = int(action.item())
            print(
                f"Step {frame_count}: Predicted action {last_action} ({KUNGFU_ACTION_NAMES[last_action]})"
            )

        # Step environment
        obs, _, done, _ = env.step([last_action])
        if done:
            obs = env.reset()

        # Update texture
        viewport_rgb = np.flipud(obs["viewport"][0][:, :, -3:])
        np.copyto(viewport_buffer, viewport_rgb)
        raw_data = viewport_buffer.tobytes()
        texture.blit_into(
            pyglet.image.ImageData(
                screen_width, screen_height, "RGB", raw_data, pitch=screen_width * 3
            ),
            0,
            0,
            0,
        )

    # Draw game screen
    @win.event
    def on_draw():
        win.clear()
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        texture.blit(0, 0, width=WINDOW_WIDTH, height=win_height)

    # Handle window close
    @win.event
    def on_close():
        pyglet.app.exit()

    # Schedule updates
    clock.schedule_interval(update, 1 / 60.0 / GAME_SPEED)

    print("Running in AI mode")
    pyglet.app.run()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGame terminated by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
# endif
