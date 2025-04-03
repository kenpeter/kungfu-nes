import argparse
import retro
import numpy as np
from pynput import keyboard
import time
import logging
from gym import ActionWrapper
import gym.spaces

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class KungFuDiscreteWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(11)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 1: Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # 2: Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 3: Kick
            [0,1,0,0,0,0,0,0,0,0,0,0],  # 4: Punch
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 5: Kick+Left
            [1,0,0,0,0,0,0,0,1,0,0,0],  # 6: Kick+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # 7: Punch+Left
            [0,1,0,0,0,0,0,0,1,0,0,0],  # 8: Punch+Right
            [0,0,0,0,0,1,0,0,0,0,0,0],  # 9: Duck
            [0,0,1,0,0,0,0,0,0,0,0,0]   # 10: Jump
        ]
        self.action_names = [
            "No action", "Left", "Right", "Kick", "Punch",
            "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right",
            "Duck", "Jump"
        ]

    def action(self, action):
        if isinstance(action, (list, np.ndarray)):
            action = int(action.item() if isinstance(action, np.ndarray) else action[0])
        return self._actions[action]

def make_kungfu_env(render=True):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    if render:
        env.render()
    return env

def capture_state(args):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('capture.log'))
    
    env = make_kungfu_env(render=True)
    obs = env.reset()
    done = False
    steps = 0
    frame_time = 1/60

    # Key tracking variables
    current_keys = set()
    save_requested = False
    quit_requested = False

    def on_press(key):
        nonlocal save_requested, quit_requested
        try:
            k = key.char.lower()
            if k == 's':
                save_requested = True
            elif k == 'q':
                quit_requested = True
            elif k == 'z':
                current_keys.add('punch')
            elif k == 'x':
                current_keys.add('kick')
            elif k == 'c':
                current_keys.add('jump')
            elif k == 'down':
                current_keys.add('duck')
        except AttributeError:
            if key == keyboard.Key.left:
                current_keys.add('left')
            elif key == keyboard.Key.right:
                current_keys.add('right')
            elif key == keyboard.Key.up:
                current_keys.add('jump')
            elif key == keyboard.Key.down:
                current_keys.add('duck')

    def on_release(key):
        try:
            k = key.char.lower()
            if k == 'z':
                current_keys.discard('punch')
            elif k == 'x':
                current_keys.discard('kick')
            elif k == 'c':
                current_keys.discard('jump')
        except AttributeError:
            if key == keyboard.Key.left:
                current_keys.discard('left')
            elif key == keyboard.Key.right:
                current_keys.discard('right')
            elif key == keyboard.Key.up:
                current_keys.discard('jump')
            elif key == keyboard.Key.down:
                current_keys.discard('duck')

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("Controls:")
    print("Left/Right Arrows: Move")
    print("Z: Punch")
    print("X: Kick")
    print("Up Arrow or C: Jump")
    print("Down Arrow: Duck")
    print("Press 'S' to save state, 'Q' to quit")

    try:
        while not done and not quit_requested:
            start_time = time.time()
            env.render()
            steps += 1

            # Determine action based on pressed keys
            action = 0  # No action
            
            # Movement takes priority
            if 'left' in current_keys:
                action = 1
            elif 'right' in current_keys:
                action = 2
            
            # Attack combinations
            if 'kick' in current_keys:
                if action == 1:  # Left + Kick
                    action = 5
                elif action == 2:  # Right + Kick
                    action = 6
                elif action == 0:  # Just Kick
                    action = 3
            elif 'punch' in current_keys:
                if action == 1:  # Left + Punch
                    action = 7
                elif action == 2:  # Right + Punch
                    action = 8
                elif action == 0:  # Just Punch
                    action = 4
            
            # Special moves (override other actions)
            if 'jump' in current_keys:
                action = 10
            elif 'duck' in current_keys:
                action = 9

            obs, reward, done, info = env.step(action)
            logging.info(f"Step {steps}: Action={env.action_names[action]}")

            # Handle save requests
            if save_requested:
                with open(args.state_file, "wb") as f:
                    f.write(env.unwrapped.get_state())
                print(f"State saved to '{args.state_file}' at step {steps}")
                save_requested = False
                
            # Maintain frame rate
            elapsed = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed))

    except Exception as e:
        print(f"Error: {str(e)}")
        logging.error(f"Error: {str(e)}")
    finally:
        listener.stop()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KungFu Master State Capture")
    parser.add_argument("--state_file", default="custom_state.state", help="Save file path")
    parser.add_argument("--enable_file_logging", action="store_true", help="Enable file logging")
    args = parser.parse_args()
    capture_state(args)