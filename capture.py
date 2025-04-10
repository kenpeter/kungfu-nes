import argparse
import os
import time
import numpy as np
import retro
import pyglet
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
import ctypes  # Added for OpenGL texture handling
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gym  # Added for base gym.Wrapper

# Placeholder KungFuWrapper; replace with your actual implementation
class KungFuWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def get_screen(self):
        """Returns the current screen as a 3D array [height, width, 3]."""
        return self.env.get_screen()

class KungFuRecorder:
    def __init__(self, game, state, save_dir='training_data', state_dir='saved_states'):
        """
        Initialize the KungFuRecorder with environment and display setup.

        Args:
            game (str): The retro game to load (e.g., 'KungFu-Nes').
            state (str): The initial state of the game.
            save_dir (str): Directory to save recorded segments.
            state_dir (str): Directory to save game states.
        """
        # Initialize retro environment
        base_env = retro.make(game=game, state=state, use_restricted_actions=retro.Actions.ALL)
        self.env = KungFuWrapper(base_env)
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecFrameStack(self.env, n_stack=12)  # Stack 12 frames for training
        
        # Access the base environment (KungFuWrapper) for raw screen data
        self.base_env = self.env.venv.envs[0]
        
        # Get screen dimensions from the raw screen
        screen = self.base_env.get_screen()  # Shape: [height, width, 3]
        self.screen_height, self.screen_width = screen.shape[:2]
        
        # Set up directories
        self.save_dir = save_dir
        self.state_dir = state_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Recording state
        self.recording = False
        self.current_segment = []
        self.segment_id = self._get_next_id(self.save_dir)
        self.state_id = self._get_next_id(self.state_dir)
        
        # Set up Pyglet window for display
        self.win = pyglet.window.Window(
            width=800,
            height=800 * self.screen_height // self.screen_width
        )
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.win.push_handlers(self.key_handler)
        
        # Set up OpenGL texture for rendering
        glEnable(GL_TEXTURE_2D)
        self.texture_id = GLuint(0)
        glGenTextures(1, ctypes.byref(self.texture_id))
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        # Initialize control states
        self.controls = {
            'up': False, 'down': False, 'left': False, 'right': False,
            'a': False, 's': False
        }

    def _get_next_id(self, directory):
        """Determine the next available ID for segments or states."""
        try:
            existing = [
                int(f.split('_')[1].split('.')[0])
                for f in os.listdir(directory)
                if f.startswith('segment_') or f.startswith('state_')
            ]
            return max(existing) + 1 if existing else 0
        except Exception:
            return 0

    def _update_texture(self):
        """Update the OpenGL texture with the current screen data."""
        screen = self.base_env.get_screen()  # Shape: [height, width, 3]
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGB,
            self.screen_width, self.screen_height,
            0, GL_RGB, GL_UNSIGNED_BYTE,
            screen.tobytes()
        )

    def _save_segment(self):
        """Save the current recorded segment to a .npz file."""
        if len(self.current_segment) > 0:
            segment_path = os.path.join(self.save_dir, f"segment_{self.segment_id}.npz")
            frames = np.array([f[0] for f in self.current_segment])  # Stacked observations
            actions = np.array([f[1] for f in self.current_segment])
            np.savez(segment_path, frames=frames, actions=actions)
            print(f"💿 Saved segment {self.segment_id} ({len(self.current_segment)} frames)")
            self.segment_id += 1

    def run(self):
        """Run the recording loop with environment interaction and display."""
        print("🕹️ KUNG FU RECORDER CONTROLS:")
        print("↑ ↓ ← → : Move/Jump/Duck | A: Punch | S: Kick")
        print("R: Start/Stop recording | O: Save state | P: Load state | Q: Quit")
        
        obs = self.env.reset()
        clock.set_fps_limit(60)

        @self.win.event
        def on_draw():
            self.win.clear()
            glBindTexture(GL_TEXTURE_2D, self.texture_id)
            glBegin(GL_QUADS)
            glTexCoord2f(0, 1); glVertex2f(0, 0)
            glTexCoord2f(1, 1); glVertex2f(self.win.width, 0)
            glTexCoord2f(1, 0); glVertex2f(self.win.width, self.win.height)
            glTexCoord2f(0, 0); glVertex2f(0, self.win.height)
            glEnd()

        while not self.win.has_exit:
            self.win.dispatch_events()
            
            # Update controls based on keyboard input
            self.controls['up'] = self.key_handler[keycodes.UP]
            self.controls['down'] = self.key_handler[keycodes.DOWN]
            self.controls['left'] = self.key_handler[keycodes.LEFT]
            self.controls['right'] = self.key_handler[keycodes.RIGHT]
            self.controls['a'] = self.key_handler[keycodes.A]
            self.controls['s'] = self.key_handler[keycodes.S]

            # Handle key commands
            if self.key_handler[keycodes.R]:
                if self.recording:
                    self.recording = False
                    self._save_segment()
                    print("⏹ Recording STOPPED")
                else:
                    self.recording = True
                    self.current_segment = []
                    print("⏺ Recording STARTED")
                time.sleep(0.2)  # Debounce
            if self.key_handler[keycodes.O]:
                state_path = os.path.join(self.state_dir, f"state_{self.state_id}.state")
                self.base_env.env.save_state(state_path)  # Access RetroEnv
                print(f"💾 Saved state {self.state_id}")
                self.state_id += 1
                time.sleep(0.2)
            if self.key_handler[keycodes.P]:
                load_id = max(0, self.state_id - 1)
                state_path = os.path.join(self.state_dir, f"state_{load_id}.state")
                if os.path.exists(state_path):
                    self.base_env.env.load_state(state_path)
                    obs = self.env.reset()
                    print(f"🔃 Loaded state {load_id}")
                else:
                    print("❌ No saved states found!")
                time.sleep(0.2)
            if self.key_handler[keycodes.Q]:
                break

            # Define action based on controls
            action = [
                int(self.controls['up']),
                int(self.controls['down']),
                int(self.controls['left']),
                int(self.controls['right']),
                int(self.controls['a']),
                int(self.controls['s'])
            ]

            # Step the environment
            obs, _, done, _ = self.env.step(action)
            
            # Record frame and action if recording
            if self.recording:
                self.current_segment.append((obs.copy(), action))

            # Update the display
            self._update_texture()
            pyglet.clock.tick()
            self.win.flip()

            if done:
                obs = self.env.reset()

        self.env.close()
        self.win.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='KungFu-Nes', help='Retro game to use')
    parser.add_argument('--state', default=retro.State.DEFAULT, help='State to start from')
    args = parser.parse_args()
    recorder = KungFuRecorder(game=args.game, state=args.state)
    recorder.run()