import argparse
import random
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

SAVE_PERIOD = 60  # frames
DEFAULT_GAME_SPEED = 1.0  # 1.0 is normal speed, lower values slow down the game
STATE_SAVE_DIR = "saved_states"  # Directory for saved game states
RECORDING_DIR = "recordings"     # Directory for ML training data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    parser.add_argument('--width', type=int, help='window width', default=800)
    parser.add_argument('--speed', type=float, help='game speed (1.0 is normal)', default=DEFAULT_GAME_SPEED)
    args = parser.parse_args()

    game_speed = args.speed
    print(f"Game speed set to: {game_speed}x (1.0 is normal speed)")

    # Create directories for saved states and recordings
    os.makedirs(STATE_SAVE_DIR, exist_ok=True)
    os.makedirs(RECORDING_DIR, exist_ok=True)

    # Placeholder for game and state (to be set by UI)
    game = None
    state = None
    game_state_path = None

    # Initialize environment (game and state will be set later via UI)
    env = None
    obs = None
    screen_height, screen_width = None, None

    def initialize_game(game_name, state_name):
        nonlocal env, obs, screen_height, screen_width, game, state, game_state_path
        game = game_name
        state = state_name
        game_state_path = os.path.join(STATE_SAVE_DIR, f"{game}_{state}.state")
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.ALL,
            scenario=args.scenario
        )
        obs = env.reset()
        screen_height, screen_width = obs.shape[:2]
        print("Available buttons:", env.buttons)

    # Simulate UI setting game and state (replace with actual UI logic)
    # For demonstration, we'll use KungFu-Nes and 1Player.Level1 as per your example
    initialize_game('KungFu-Nes', '1Player.Level1')

    random.seed(0)

    key_handler = pyglet.window.key.KeyStateHandler()
    win_width = args.width
    win_height = win_width * screen_height // screen_width

    win = pyglet.window.Window(
        width=win_width,
        height=win_height,
        vsync=False,
        caption=f"{game} - {state} (Game Running)",
        style=pyglet.window.Window.WINDOW_STYLE_DEFAULT
    )

    print("\nWindow Controls:")
    print("- R: Toggle recording game segment for ML training")
    print("- O: Save current game state")
    print("- P: Load previously saved game state")
    print("- Game controls: Arrow keys, Z(A), X(B), C, A(X), S(Y), D(Z), ENTER(START), SPACE(SELECT), TAB(MODE)")
    print("- Press CTRL+C in terminal to force quit\n")

    pixel_scale = 1.0
    if hasattr(win.context, '_nscontext'):
        pixel_scale = win.context._nscontext.view().backingScaleFactor()

    win.width = int(win.width // pixel_scale)
    win.height = int(win.height // pixel_scale)

    win.push_handlers(key_handler)

    key_previous_states = {}

    steps = 0
    recorded_actions = []
    recorded_states = []

    is_recording = False
    recording_frames = []
    recording_actions = []
    recording_rewards = []
    recording_count = 0

    pyglet.app.platform_event_loop.start()

    fps_display = pyglet.window.FPSDisplay(win)

    pyglet.clock.schedule_interval(lambda dt: None, 1/60.0)

    glEnable(GL_TEXTURE_2D)
    texture_id = GLuint(0)
    glGenTextures(1, ctypes.byref(texture_id))
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, screen_width, screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)

    last_frame_time = time.time()
    frame_duration = 1/60.0 / game_speed

    def update_caption():
        status = " [RECORDING]" if is_recording else ""
        win.set_caption(f"{game} - {state}{status} (Game Running)")

    def save_recording():
        nonlocal recording_frames, recording_actions, recording_rewards, recording_count
        if len(recording_frames) == 0:
            print("No frames to save!")
            return

        frames_array = np.array(recording_frames)
        actions_array = np.array(recording_actions)
        rewards_array = np.array(recording_rewards)

        recording_path = os.path.join(RECORDING_DIR, f"{game}_{state}_rec{recording_count}.npz")
        np.savez(
            recording_path,
            frames=frames_array,
            actions=actions_array,
            rewards=rewards_array
        )
        print(f"Saved recording with {len(recording_frames)} frames to {recording_path}")

        recording_frames = []
        recording_actions = []
        recording_rewards = []
        recording_count += 1

    while not win.has_exit:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)
        last_frame_time = time.time()

        win.dispatch_events()

        win.clear()

        keys_clicked = set()
        keys_pressed = set()
        for key_code, pressed in key_handler.items():
            if pressed:
                keys_pressed.add(key_code)

            if not key_previous_states.get(key_code, False) and pressed:
                keys_clicked.add(key_code)
            key_previous_states[key_code] = pressed

        if keycodes.R in keys_clicked:
            if not is_recording:
                is_recording = True
                recording_frames = []
                recording_actions = []
                recording_rewards = []
                print("Recording started for ML training data")
            else:
                is_recording = False
                save_recording()
                print("Recording stopped")

            update_caption()

        if keycodes.O in keys_clicked:
            save_state = env.em.get_state()
            with open(game_state_path, "wb") as f:
                f.write(save_state)
            print(f"Game state saved to {game_state_path}")

        if keycodes.P in keys_clicked:
            if os.path.exists(game_state_path):
                with open(game_state_path, "rb") as f:
                    save_state = f.read()
                env.em.set_state(save_state)
                print(f"Game state loaded from {game_state_path}")
            else:
                print(f"No saved state found at {game_state_path}")

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
            None: False
        }

        action = []
        for b in env.buttons:
            if b in inputs:
                action.append(inputs[b])
            else:
                action.append(False)

        if steps % SAVE_PERIOD == 0:
            recorded_states.append((steps, env.em.get_state()))
        obs, rew, done, info = env.step(action)
        recorded_actions.append(action)
        steps += 1

        if is_recording:
            recording_frames.append(obs.copy())
            recording_actions.append(action.copy())
            recording_rewards.append(rew)

        glBindTexture(GL_TEXTURE_2D, texture_id)
        video_buffer = ctypes.cast(obs.tobytes(), ctypes.POINTER(ctypes.c_short))
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, obs.shape[1], obs.shape[0], GL_RGB, GL_UNSIGNED_BYTE, video_buffer)

        x = 0
        y = 0
        h = win.height
        w = win.width

        pyglet.graphics.draw(
            4,
            pyglet.gl.GL_QUADS,
            ('v2f', [x, y, x + w, y, x + w, y + h, x, y + h]),
            ('t2f', [0, 1, 1, 1, 1, 0, 0, 0]),
        )

        fps_display.draw()

        win.flip()

        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)

        clock.tick()


if __name__ == "__main__":
    main()