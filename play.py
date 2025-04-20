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
import retro
from stable_baselines3 import PPO
from kungfu_env import KUNGFU_ACTIONS, KUNGFU_ACTION_NAMES, make_env

# Constants
DEFAULT_GAME_SPEED = 1.0
MODEL_PATH = "models/kungfu_ppo_model.zip"

def map_input_to_discrete_action(inputs, buttons, keys_pressed):
    """Map keyboard inputs to a discrete action index (0-12) based on KUNGFU_ACTIONS."""
    # Check for combo actions first
    if inputs.get('DOWN', False) and inputs.get('A', False):  # DOWN + Z for Crouch Kick
        print("DEBUG - DOWN + A pressed - using action 11 (Crouch Kick)")
        return 11  # DOWN + A
    
    if inputs.get('DOWN', False) and inputs.get('B', False):  # DOWN + X for Crouch Punch
        print("DEBUG - DOWN + B pressed - using action 12 (Crouch Punch)")
        return 12  # DOWN + B
    
    if inputs.get('UP', False) and inputs.get('RIGHT', False):  # UP + RIGHT
        print("DEBUG - UP + RIGHT pressed - using action 10")
        return 10  # UP + RIGHT
    
    # Map individual game inputs
    if inputs.get('UP', False):
        print("DEBUG - UP pressed - using action 4")
        return 4  # UP
    
    if inputs.get('DOWN', False):
        print("DEBUG - DOWN pressed - using action 5")
        return 5  # DOWN
    
    if inputs.get('LEFT', False):
        print("DEBUG - LEFT pressed - using action 6")
        return 6  # LEFT
    
    if inputs.get('RIGHT', False):
        print("DEBUG - RIGHT pressed - using action 7")
        return 7  # RIGHT
    
    if inputs.get('A', False):  # Z key
        print("DEBUG - A pressed - using action 8")
        return 8  # A
    
    if inputs.get('B', False):  # X key
        print("DEBUG - B pressed - using action 1")
        return 1  # B
    
    # Default to no-op
    return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    parser.add_argument('--width', type=int, help='window width', default=800)
    parser.add_argument('--speed', type=float, help='game speed (1.0 is normal)', default=DEFAULT_GAME_SPEED)
    parser.add_argument('--model', help='path to trained model', default=MODEL_PATH)
    parser.add_argument('--manual', action='store_true', help='use manual keyboard controls instead of model')
    args = parser.parse_args()

    # Initialize game environment
    game = 'KungFu-Nes'
    state = '1Player.Level1'
    
    print("Creating game environment...")
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        scenario=args.scenario,
        render_mode="rgb_array"
    )
    
    # Handle reset
    print("Resetting environment...")
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result
        print("Using new API (tuple return from reset)")
    else:
        obs = result
        print("Using old API (single value return from reset)")
        
    screen_height, screen_width = obs.shape[:2]
    print(f"Screen dimensions: {screen_width}x{screen_height}")
    
    # Load model if not manual mode
    model = None
    if not args.manual:
        if os.path.exists(args.model):
            print(f"Loading model from {args.model}...")
            model = PPO.load(args.model)
            if model.n_actions != len(KUNGFU_ACTIONS):
                print(f"WARNING: Model action space ({model.n_actions}) does not match environment ({len(KUNGFU_ACTIONS)}). New actions (Crouch Kick, Crouch Punch) may be ignored.")
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
        caption=f"{game} - {state} (Playing)",
        style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
    )

    # Print controls for manual mode
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

    # Input handling for manual mode
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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    # Timing control
    last_frame_time = time.time()
    frame_duration = 1/60.0 / args.speed

    print("Starting main game loop...")
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
            # Input processing
            keys_clicked = set()
            keys_pressed = set()
            for key_code, pressed in key_handler.items():
                if pressed:
                    keys_pressed.add(key_code)
                if not key_previous_states.get(key_code, False) and pressed:
                    keys_clicked.add(key_code)
                key_previous_states[key_code] = pressed

            # Convert keyboard inputs to game actions
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
            
            # Map inputs to discrete action
            discrete_action = map_input_to_discrete_action(inputs, env.buttons, keys_pressed)
        else:
            # Use model to predict action
            action, _ = model.predict(obs, deterministic=False)
            discrete_action = action.item()

        action = KUNGFU_ACTIONS[discrete_action]
        print(f"Action: {discrete_action} ({KUNGFU_ACTION_NAMES[discrete_action]}) -> {action}")

        # Step environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, rew, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, rew, done, info = step_result

        # Handle game over
        if done:
            print("Game over detected - resetting")
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result

        # Rendering
        glBindTexture(GL_TEXTURE_2D, texture_id)
        try:
            video_buffer = ctypes.cast(obs.tobytes(), ctypes.POINTER(ctypes.c_short))
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, obs.shape[1], obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)
        except Exception as e:
            print(f"Rendering error: {e}")

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
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