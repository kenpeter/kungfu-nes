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
from kungfu_env import KUNGFU_ACTIONS, KUNGFU_ACTION_NAMES, make_env

# Constants
DEFAULT_GAME_SPEED = 1.0
MODEL_PATH = "models/kungfu_ppo/kungfu_ppo.zip"

def map_input_to_discrete_action(inputs):
    """Map keyboard inputs to a discrete action index (0-12) based on KUNGFU_ACTIONS."""
    if inputs.get('DOWN', False) and inputs.get('A', False):
        print("Manual Input - DOWN + A: Crouch Kick (11)")
        return 11
    if inputs.get('DOWN', False) and inputs.get('B', False):
        print("Manual Input - DOWN + B: Crouch Punch (12)")
        return 12
    if inputs.get('UP', False) and inputs.get('RIGHT', False):
        print("Manual Input - UP + RIGHT: Jump + Right (10)")
        return 10
    if inputs.get('UP', False):
        print("Manual Input - UP: Jump (4)")
        return 4
    if inputs.get('DOWN', False):
        print("Manual Input - DOWN: Crouch (5)")
        return 5
    if inputs.get('LEFT', False):
        print("Manual Input - LEFT: Left (6)")
        return 6
    if inputs.get('RIGHT', False):
        print("Manual Input - RIGHT: Right (7)")
        return 7
    if inputs.get('A', False):
        print("Manual Input - A: Kick (8)")
        return 8
    if inputs.get('B', False):
        print("Manual Input - B: Punch (1)")
        return 1
    if inputs.get('START', False):
        print("Manual Input - START: Start (3)")
        return 3
    if inputs.get('SELECT', False):
        print("Manual Input - SELECT: Select (2)")
        return 2
    print("Manual Input - No valid input: No-op (0)")
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    parser.add_argument('--width', type=int, help='window width', default=800)
    parser.add_argument('--speed', type=float, help='game speed (1.0 is normal)', default=DEFAULT_GAME_SPEED)
    parser.add_argument('--model', help='path to trained model', default=MODEL_PATH)
    parser.add_argument('--manual', action='store_true', help='use manual keyboard controls instead of model')
    args = parser.parse_args()

    # Initialize environment
    print("Creating game environment with KungFuWrapper...")
    env = make_env()
    print("Resetting environment...")
    obs, info = env.reset()
    screen_height, screen_width = obs['viewport'].shape[:2]
    print(f"Screen dimensions: {screen_width}x{screen_height} (160x160 from KungFuWrapper)")

    # Load model
    model = None
    if not args.manual:
        if os.path.exists(args.model):
            print(f"Loading model from {args.model}...")
            model = PPO.load(args.model)
            expected_actions = len(KUNGFU_ACTIONS)
            model_actions = model.policy.action_space.n
            if model_actions != expected_actions:
                print(f"WARNING: Model action space ({model_actions}) does not match environment ({expected_actions}).")
            print("Model observation space:", model.observation_space)
        else:
            print(f"Model not found at {args.model}. Falling back to manual mode.")
            args.manual = True

    # Window setup
    win_width = args.width
    win_height = win_width * screen_height // screen_width
    print(f"Creating window with dimensions: {win_width}x{win_height}")
    win = pyglet.window.Window(
        width=win_width,
        height=win_height,
        vsync=False,
        caption=f"KungFu-Nes - Playing",
        style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
    )

    if args.manual:
        print("\nControls:")
        print("- Arrow keys: UP (Jump), DOWN (Crouch), LEFT, RIGHT")
        print("- Z: A (Kick)")
        print("- X: B (Punch)")
        print("- Combos: DOWN + Z (Crouch Kick), DOWN + X (Crouch Punch), UP + RIGHT (Jump + Right)")
        print("- ENTER: START")
        print("- SPACE: SELECT")
        print("- Press CTRL+C in terminal to quit\n")

    # Handle high DPI displays
    pixel_scale = 1.0
    if hasattr(win.context, '_nscontext'):
        pixel_scale = win.context._nscontext.view().backingScaleFactor()
    win.width = int(win.width // pixel_scale)
    win.height = int(win.height // pixel_scale)

    # Input handling
    key_handler = pyglet.window.key.KeyStateHandler()
    win.push_handlers(key_handler)
    key_previous_states = {}

    # Performance monitoring
    fps_display = pyglet.window.FPSDisplay(win)
    pyglet.clock.schedule_interval(lambda dt: None, 1/60.0)

    # OpenGL texture setup
    glEnable(GL_TEXTURE_2D)
    texture_id = GLuint(0)
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    # Timing control
    last_frame_time = time.time()
    frame_duration = 1/60.0 / args.speed

    print("Starting main game loop...")
    step_count = 0
    while not win.has_exit:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
        last_frame_time = time.time()

        win.dispatch_events()
        win.clear()

        # Get action
        if args.manual:
            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)
                if not key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                key_previous_states[key_code] = pressed

            inputs = {
                'A': keycodes.Z in keys_pressed,
                'B': keycodes.X in keys_pressed,
                'UP': keycodes.UP in keys_pressed,
                'DOWN': keycodes.DOWN in keys_pressed,
                'LEFT': keycodes.LEFT in keys_pressed,
                'RIGHT': keycodes.RIGHT in keys_pressed,
                'START': keycodes.ENTER in keys_pressed,
                'SELECT': keycodes.SPACE in keys_pressed,
            }
            discrete_action = map_input_to_discrete_action(inputs)
        else:
            print(f"Observation: {obs}")
            action, _ = model.predict(obs, deterministic=False)
            discrete_action = action.item()
            print(f"Model Action: {discrete_action} ({KUNGFU_ACTION_NAMES[discrete_action]})")

        # Step environment
        obs, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated
        step_count += 1

        # Log step info
        if step_count % 10 == 0 or done:
            print(f"Step {step_count}: Action={KUNGFU_ACTION_NAMES[discrete_action]}, Reward={reward:.2f}, "
                  f"HP={info.get('hp', 0):.1f}, EnemyHit={info.get('enemy_hit', 0)}, Done={done}")

        # Handle reset
        if done:
            print("Game over detected - resetting")
            obs, info = env.reset()

        # Rendering
        glBindTexture(GL_TEXTURE_2D, texture_id)
        try:
            viewport = obs['viewport']
            print(f"Rendering viewport: shape={viewport.shape}, dtype={viewport.dtype}, min={viewport.min()}, max={viewport.max()}")
            video_buffer = ctypes.cast(viewport.tobytes(), ctypes.POINTER(ctypes.c_ubyte))
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, screen_width, screen_height, GL_RGB, GL_UNSIGNED_BYTE, video_buffer)
        except Exception as e:
            print(f"Rendering error: {e}")

        pyglet.graphics.draw(
            4,
            GL_QUADS,
            ('v2f', [0, 0, win.width, 0, win.width, win.height, 0, win.height]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

        fps_display.draw()
        win.flip()

        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)
        clock.tick()

    print("Closing environment...")
    env.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPlay terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)