import argparse
import retro
import numpy as np
import keyboard  # Install with `pip install keyboard`
import time
import logging
from gym import ActionWrapper  # For wrapping the environment
import gym.spaces  # For defining action spaces

# Configure basic logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Custom action wrapper to map discrete actions to Kung Fu Master controls
class KungFuDiscreteWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(11)  # Define discrete action space with 11 actions
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 1: Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # 2: Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 3: Kick (B)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # 4: Punch (A)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 5: Kick+Left
            [1,0,0,0,0,0,0,0,1,0,0,0],  # 6: Kick+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # 7: Punch+Left
            [0,1,0,0,0,0,0,0,1,0,0,0],  # 8: Punch+Right
            [0,0,0,0,0,1,0,0,0,0,0,0],  # 9: Down (Duck)
            [0,0,1,0,0,0,0,0,0,0,0,0]   # 10: Up (Jump)
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

# Function to set up the Retro environment with the action wrapper
def make_kungfu_env(render=True):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    if render:
        env.render_mode = 'human'
    return env

# Main function to capture the game state
def capture_state(args):
    # Enable file logging if requested
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('capture.log'))
    
    # Set up the environment
    env = make_kungfu_env(render=True)
    obs = env.reset()
    done = False
    steps = 0
    frame_time = 1 / 60  # Target 60 FPS

    # Display controls to the user
    print("Controls: Left/Right Arrows, Z (Punch), X (Kick), Up (Jump), Down (Duck)")
    print(f"Press 'S' to save state to '{args.state_file}', 'Q' to quit.")

    try:
        while not done:
            start_time = time.time()
            env.render()
            steps += 1

            # Default action is "No action"
            action = 0

            # Map keyboard inputs to actions
            if keyboard.is_pressed('left'):
                action = 1
                print("Left pressed!")
            elif keyboard.is_pressed('right'):
                action = 2
                print("Right pressed!")
            elif keyboard.is_pressed('z'):
                action = 4
                print("Z pressed!")
            elif keyboard.is_pressed('x'):
                action = 3
                print("X pressed!")
            elif keyboard.is_pressed('up'):
                action = 10
                print("Up pressed!")
            elif keyboard.is_pressed('down'):
                action = 9
                print("Down pressed!")

            # Step the environment with the selected action
            obs, reward, done, info = env.step(action)
            logging.info(f"Step {steps}: Action={env.action_names[action]}")

            # Save state when 'S' is pressed
            if keyboard.is_pressed('s'):
                with open(args.state_file, "wb") as f:
                    f.write(env.unwrapped.get_state())
                print(f"State saved to '{args.state_file}' at step {steps}")
                logging.info(f"State saved to '{args.state_file}' at step {steps}")

            # Quit when 'Q' is pressed
            if keyboard.is_pressed('q'):
                print("Quitting...")
                break

            # Maintain frame rate
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed_time))

    except Exception as e:
        print(f"Error during capture: {str(e)}")
        logging.error(f"Error during capture: {str(e)}")
    finally:
        env.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Manually play KungFu Master and capture a state")
    parser.add_argument(
        "--state_file",
        default="custom_state.state",
        help="File to save the captured state"
    )
    parser.add_argument(
        "--enable_file_logging",
        action="store_true",
        help="Enable logging to file 'capture.log'"
    )

    args = parser.parse_args()
    capture_state(args)