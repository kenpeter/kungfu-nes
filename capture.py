import argparse
import os
import time
import numpy as np
import retro
import pyglet
from pyglet import clock
from pyglet.window import key as keycodes
from pyglet.gl import *
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from kungfu_env import KungFuWrapper  # Assuming this is your custom wrapper

class KungFuRecorder:
    def __init__(self, game, state, save_dir='training_data', state_dir='saved_states'):
        # Initialize retro environment
        base_env = retro.make(game=game, state=state, use_restricted_actions=retro.Actions.ALL)
        self.env = KungFuWrapper(base_env)
        self.env = DummyVecEnv([lambda: self.env])
        self.env = VecFrameStack(self.env, n_stack=12)  # Match your training setup
        
        # Directories
        self.save_dir = save_dir
        self.state_dir = state_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        
        # Recording state
        self.recording = False
        self.current_segment = []
        self.segment_id = self._get_next_id(self.save_dir)
        self.state_id = self._get_next_id(self.state_dir)
        
        # Window setup
        obs = self.env.reset()
        # Assuming 'screen' is the key for the image data; adjust based on your KungFuWrapper
        screen_data = obs['screen'] if 'screen' in obs else list(obs.values())[0]  # Fallback to first value
        self.screen_height, self.screen_width = screen_data.shape[1:3]  # [n_stack, height, width, channels]
        
        self.win = pyglet.window.Window(width=800, height=800 * self.screen_height // self.screen_width)
        self.key_handler = pyglet.window.key.KeyStateHandler()
        self.win.push_handlers(self.key_handler)
        
        # OpenGL texture setup
        glEnable(GL_TEXTURE_2D)
        self.texture_id = GLuint(0)
        glGenTextures(1, ctypes.byref(self.texture_id))
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        
        # Control states
        self.controls = {'up': False, 'down': False, 'left': False, 'right': False, 'a': False, 's': False}

    def _get_next_id(self, directory):
        try:
            existing = [int(f.split('_')[1].split('.')[0]) 
                       for f in os.listdir(directory) if f.startswith('segment_') or f.startswith('state_')]
            return max(existing) + 1 if existing else 0
        except Exception:
            return 0

    def _update_texture(self, obs):
        # Extract screen data from dict; adjust key as needed
        screen_data = obs['screen'] if 'screen' in obs else list(obs.values())[0]
        frame = screen_data[0, :, :, :3]  # Use first frame, drop extra channels if any
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.screen_width, self.screen_height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame.tobytes())

    def _save_segment(self):
        if len(self.current_segment) > 0:
            segment_path = os.path.join(self.save_dir, f"segment_{self.segment_id}.npz")
            frames = np.array([f[0] for f in self.current_segment])  # Full dict saved
            actions = np.array([f[1] for f in self.current_segment])
            np.savez(segment_path, frames=frames, actions=actions)
            print(f"💿 Saved segment {self.segment_id} ({len(self.current_segment)} frames)")
            self.segment_id += 1

    def run(self):
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
            
            # Update controls
            self.controls['up'] = self.key_handler[keycodes.UP]
            self.controls['down'] = self.key_handler[keycodes.DOWN]
            self.controls['left'] = self.key_handler[keycodes.LEFT]
            self.controls['right'] = self.key_handler[keycodes.RIGHT]
            self.controls['a'] = self.key_handler[keycodes.A]
            self.controls['s'] = self.key_handler[keycodes.S]

            # Handle commands
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
                self.env.envs[0].unwrapped.save_state(state_path)
                print(f"💾 Saved state {self.state_id}")
                self.state_id += 1
                time.sleep(0.2)
            if self.key_handler[keycodes.P]:
                load_id = max(0, self.state_id - 1)
                state_path = os.path.join(self.state_dir, f"state_{load_id}.state")
                if os.path.exists(state_path):
                    self.env.envs[0].unwrapped.load_state(state_path)
                    obs = self.env.reset()
                    print(f"🔃 Loaded state {load_id}")
                else:
                    print("❌ No saved states found!")
                time.sleep(0.2)
            if self.key_handler[keycodes.Q]:
                break

            # Human action
            action = [
                int(self.controls['up']),
                int(self.controls['down']),
                int(self.controls['left']),
                int(self.controls['right']),
                int(self.controls['a']),
                int(self.controls['s'])
            ]

            # Step environment
            obs, _, done, _ = self.env.step(action)
            
            # Record if active
            if self.recording:
                self.current_segment.append((obs.copy(), action))

            # Update display
            self._update_texture(obs)
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