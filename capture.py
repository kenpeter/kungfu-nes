import argparse
import random
import pyglet
import sys
import ctypes
import os
import time

# Import clock correctly
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *

import retro

# TODO:
# indicate to user when episode is over (hard to do without save/restore lua state)
# record bk2 directly
# resume from bk2

SAVE_PERIOD = 60  # frames
DEFAULT_GAME_SPEED = 1.0  # 1.0 is normal speed, lower values slow down the game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', help='retro game to use')
    parser.add_argument('--state', help='retro state to start from')
    parser.add_argument('--scenario', help='scenario to use', default='scenario')
    parser.add_argument('--width', type=int, help='window width', default=800)  # Smaller default width
    parser.add_argument('--speed', type=float, help='game speed (1.0 is normal)', default=DEFAULT_GAME_SPEED)
    args = parser.parse_args()

    if args.game is None:
        print('Please specify a game with --game <game>')
        print('Available games:')
        for game in sorted(retro.data.list_games()):
            print(game)
        sys.exit(1)

    if args.state is None:
        print('Please specify a state with --state <state>')
        print('Available states:')
        for state in sorted(retro.data.list_states(args.game)):
            print(state)
        sys.exit(1)

    game_speed = args.speed
    print(f"Game speed set to: {game_speed}x (1.0 is normal speed)")

    env = retro.make(game=args.game, state=args.state, use_restricted_actions=retro.Actions.ALL, scenario=args.scenario)
    obs = env.reset()
    screen_height, screen_width = obs.shape[:2]

    # Print the available buttons for debugging
    print("Available buttons:", env.buttons)

    random.seed(0)

    key_handler = pyglet.window.key.KeyStateHandler()
    win_width = args.width
    win_height = win_width * screen_height // screen_width
    
    # Create window with standard style (not always on top)
    win = pyglet.window.Window(
        width=win_width, 
        height=win_height, 
        vsync=False,
        caption=f"{args.game} - {args.state} (Press ESC to save and exit)",
        style=pyglet.window.Window.WINDOW_STYLE_DEFAULT  # Use default style, not always on top
    )

    # Print instructions
    print("\nWindow Controls:")
    print("- ESC: Save recording and exit")
    print("- R: Rewind to previous save point")
    print("- Game controls: Arrow keys, Z(A), X(B), C, A(X), S(Y), D(Z), ENTER(START), SPACE(SELECT), TAB(MODE)")
    print("- Press CTRL+C in terminal to force quit\n")

    # Initialize pixel_scale with default value
    pixel_scale = 1.0
    if hasattr(win.context, '_nscontext'):
        pixel_scale = win.context._nscontext.view().backingScaleFactor()

    # Ensure integer values
    win.width = int(win.width // pixel_scale)
    win.height = int(win.height // pixel_scale)

    win.push_handlers(key_handler)

    key_previous_states = {}

    steps = 0
    recorded_actions = []
    recorded_states = []

    pyglet.app.platform_event_loop.start()

    # Update to use FPSDisplay instead of ClockDisplay
    fps_display = pyglet.window.FPSDisplay(win)
    
    # Replace set_fps_limit with schedule_interval
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
    frame_duration = 1/60.0 / game_speed  # Target time per frame based on speed

    while not win.has_exit:
        # Control game speed
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
            if len(recorded_states) > 1:
                recorded_states.pop()
                steps, save_state = recorded_states.pop()
                recorded_states = recorded_states[:steps]
                recorded_actions = recorded_actions[:steps]
                env.em.set_state(save_state)

        if keycodes.ESCAPE in keys_pressed:
            # record all the actions so far to a bk2 and exit
            i = 0
            while True:
                movie_filename = 'human/%s/%s/%s-%s-%04d.bk2' % (args.game, args.scenario, args.game, args.state, i)
                if not os.path.exists(movie_filename):
                    break
                i += 1
            os.makedirs(os.path.dirname(movie_filename), exist_ok=True)
            env.record_movie(movie_filename)
            env.reset()
            for step, act in enumerate(recorded_actions):
                if step % 1000 == 0:
                    print('saving %d/%d' % (step, len(recorded_actions)))
                env.step(act)
            env.stop_record()
            print('complete')
            print(f"Recording saved to: {movie_filename}")
            sys.exit(0)

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
            
            # Add this to handle None buttons
            None: False
        }
        
        # Modified to handle buttons not in the inputs dictionary
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

        # process joystick events
        timeout = clock.get_sleep_time(False)
        pyglet.app.platform_event_loop.step(timeout)

        # Update the clock
        clock.tick()


if __name__ == "__main__":
    main()