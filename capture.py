import argparse
import random
import pyglet
import sys
import ctypes
import os
import time
import numpy as np
from datetime import datetime
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
import retro

# Constants
SAVE_PERIOD = 60  # frames
DEFAULT_GAME_SPEED = 1.0
STATE_SAVE_DIR = "saved_states"
RECORDING_DIR = "recordings"
MIN_RECORDING_LENGTH = 60  # Minimum frames to save a recording
FRAME_SKIP = 2  # Record every Nth frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    parser.add_argument('--width', type=int, help='window width', default=800)
    parser.add_argument('--speed', type=float, help='game speed (1.0 is normal)', default=DEFAULT_GAME_SPEED)
    parser.add_argument('--frame-skip', type=int, help='record every N frames', default=FRAME_SKIP)
    args = parser.parse_args()

    # Create directories if they don't exist
    os.makedirs(STATE_SAVE_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)

    # Initialize game environment - ONLY create one environment instance
    game = 'KungFu-Nes'
    state = '1Player.Level1'
    game_state_path = os.path.join(STATE_SAVE_DIR, f"{game}_{state}.state")
    
    print("Creating game environment...")
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.ALL,
        scenario=args.scenario,
        render_mode="rgb_array"  # Make sure this is the only render call
    )
    
    # Fix for newer Gymnasium/Retro API: properly unpack tuple return value
    print("Resetting environment...")
    result = env.reset()
    if isinstance(result, tuple):
        obs, info = result  # Unpack if it's a tuple (new API)
        print("Using new API (tuple return from reset)")
    else:
        obs = result  # For backward compatibility
        print("Using old API (single value return from reset)")
        
    screen_height, screen_width = obs.shape[:2]
    print(f"Screen dimensions: {screen_width}x{screen_height}")
    
    # Verify button order matches training wrapper
    EXPECTED_BUTTON_ORDER = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'C', 'X', 'Y', 'Z', 'MODE', 'START', 'SELECT']
    print("Button order verification:")
    print(f"  Environment: {env.buttons}")
    print(f"  Expected:    {EXPECTED_BUTTON_ORDER}")
    if list(env.buttons) != EXPECTED_BUTTON_ORDER:
        print("\nWARNING: Button order mismatch! This will cause training problems!")
        print("Modify either capture.py or your training wrapper to use consistent button ordering.\n")

    # Window setup
    win_width = args.width
    win_height = win_width * screen_height // screen_width
    print(f"Creating window with dimensions: {win_width}x{win_height}")
    win = pyglet.window.Window(
        width=win_width,
        height=win_height,
        vsync=False,
        caption=f"{game} - {state} (Game Running)",
        style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
    )

    # Print controls
    print("\nWindow Controls:")
    print("- R: Toggle recording game segment for ML training")
    print("- O: Save current game state")
    print("- P: Load previously saved game state")
    print("- Game controls: Arrow keys, Z(A), X(B), C, A(X), S(Y), D(Z), ENTER(START), SPACE(SELECT), TAB(MODE)")
    print("- Press CTRL+C in terminal to force quit\n")

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

    # Recording state
    is_recording = False
    recording_frames = []
    recording_actions = []
    recording_rewards = []
    frame_counter = 0

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

    def update_caption():
        status = " [RECORDING]" if is_recording else ""
        win.set_caption(f"{game} - {state}{status} (Game Running)")

    def save_recording():
        nonlocal recording_frames, recording_actions, recording_rewards, frame_counter
        
        if len(recording_frames) < MIN_RECORDING_LENGTH:
            print(f"Recording too short ({len(recording_frames)} frames) - discarded")
            recording_frames = []
            recording_actions = []
            recording_rewards = []
            return

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        recording_path = os.path.join(RECORDING_DIR, f"{game}_{state}_{timestamp}.npz")
        
        frames_array = np.array(recording_frames, dtype=np.uint8)
        actions_array = np.array(recording_actions)
        rewards_array = np.array(recording_rewards)

        np.savez_compressed(
            recording_path,
            frames=frames_array,
            actions=actions_array,
            rewards=rewards_array,
            buttons=list(env.buttons),
            scenario=args.scenario,
            timestamp=timestamp,
            game_speed=args.speed,
            frame_skip=args.frame_skip
        )
        
        print(f"Saved recording with {len(recording_frames)} frames to {recording_path}")
        recording_frames = []
        recording_actions = []
        recording_rewards = []

    print("Starting main game loop...")
    # Main game loop
    while not win.has_exit:
        # Frame timing control
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
        last_frame_time = time.time()

        win.dispatch_events()
        win.clear()

        # Input processing
        keys_clicked = set()
        keys_pressed = set()
        for key_code, pressed in key_handler.items():
            if pressed:
                keys_pressed.add(key_code)
            if not key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
            key_previous_states[key_code] = pressed

        # Handle recording toggle
        if keycodes.R in keys_clicked:
            is_recording = not is_recording
            if is_recording:
                print("Recording started for ML training data")
                frame_counter = 0
            else:
                save_recording()
            update_caption()

        # Handle state save/load
        if keycodes.O in keys_clicked:
            print("Saving game state...")
            save_state = env.em.get_state()
            with open(game_state_path, "wb") as f:
                f.write(save_state)
            print(f"Game state saved to {game_state_path}")

        if keycodes.P in keys_clicked:
            if os.path.exists(game_state_path):
                print("Loading game state...")
                with open(game_state_path, "rb") as f:
                    save_state = f.read()
                env.em.set_state(save_state)
                print(f"Game state loaded from {game_state_path}")
            else:
                print(f"No saved state found at {game_state_path}")

        # Convert keyboard inputs to game actions
        inputs = {
            'A': keycodes.Z in keys_pressed,
            'B': keycodes.X in keys_pressed,
            'C': keycodes.C in keys_pressed,
            'X': keycodes.A in keys_pressed,
            'Y': keycodes.S in keys_pressed,
            'Z': keycodes.D in keys_pressed,
            'UP': keycodes.UP in keys_pressed,
            'DOWN': keycodes.DOWN in keys_pressed,
            'LEFT': keycodes.LEFT in keys_pressed,
            'RIGHT': keycodes.RIGHT in keys_pressed,
            'MODE': keycodes.TAB in keys_pressed,
            'START': keycodes.ENTER in keys_pressed,
            'SELECT': keycodes.SPACE in keys_pressed,
        }

        action = [inputs.get(b, False) for b in env.buttons]
        
        # Debug input handling
        if any(action):
            print("Pressed buttons:", [btn for btn, pressed in zip(env.buttons, action) if pressed])
        
        # Handle step return value for new API
        step_result = env.step(action)
        if len(step_result) == 5:  # New API (obs, rew, terminated, truncated, info)
            obs, rew, terminated, truncated, info = step_result
            done = terminated or truncated
        else:  # Old API (obs, rew, done, info)
            obs, rew, done, info = step_result

        # Handle game over during recording
        if done and is_recording:
            print("Game over detected - saving recording")
            save_recording()
            is_recording = False
            update_caption()
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, _ = reset_result

        # Frame recording with skipping
        frame_counter += 1
        if is_recording and frame_counter % args.frame_skip == 0:
            recording_frames.append(obs.copy())
            recording_actions.append(action.copy())
            recording_rewards.append(rew)

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

        # Handle Pyglet event loop
        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)
        clock.tick()

    # Cleanup
    print("Closing environment...")
    env.close()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nCapture terminated by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)