import retro
import numpy as np
import os
import time
import json
import cv2
from pynput import keyboard
from stable_baselines3 import PPO

class TrainingCapturer:
    def __init__(self):
        self.env = retro.make('KungFu-Nes')
        self.save_dir = 'training_data'
        self.state_dir = 'saved_states'
        self.model_dir = 'models\kungfu_ppo'  # Directory where train.py saves models
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.recording = False
        self.current_segment = []
        self.segment_id = self._get_next_id(self.save_dir)
        self.state_id = self._get_next_id(self.state_dir)
        self.ai_playing = False
        self.model = self._load_ai_model()  # Load model immediately
        self._init_controls()

    def _get_next_id(self, directory):
        """Get the next available ID number for saving"""
        try:
            existing = [int(f.split('_')[1].split('.')[0]) 
                      for f in os.listdir(directory) if f.startswith('segment_')]
            return max(existing) + 1 if existing else 0
        except Exception:
            return 0

    def _init_controls(self):
        self.controls = {
            'up': False,    # Jump
            'down': False,  # Duck
            'left': False,  # Move left
            'right': False, # Move right
            'a': False,     # Punch
            's': False      # Kick
        }

    def _load_ai_model(self):
        """Load the best model from train.py's output"""
        model_path = os.path.join(self.model_dir, 'kungfu_ppo_best.zip')
        if os.path.exists(model_path):
            print(f"Loading AI model from {model_path}")
            return PPO.load(model_path, env=self.env)
        print(f"No AI model found at {model_path}")
        return None

    def toggle_ai(self):
        if self.model:
            self.ai_playing = not self.ai_playing
            print(f"AI control {'ON' if self.ai_playing else 'OFF'}")
        else:
            print("First train a model with train.py!")

    def start_recording(self):
        if not self.recording:
            self.recording = True
            self.current_segment = []
            print("⏺ Recording STARTED")

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self._save_segment()
            print("⏹ Recording STOPPED")

    def save_game_state(self):
        state_path = os.path.join(self.state_dir, f"state_{self.state_id}.state")
        self.env.save_state(state_path)
        print(f"💾 Saved game state {self.state_id}")
        self.state_id += 1

    def load_game_state(self):
        load_id = max(0, self.state_id - 1)
        state_path = os.path.join(self.state_dir, f"state_{load_id}.state")
        if os.path.exists(state_path):
            self.env.load_state(state_path)
            print(f"🔃 Loaded game state {load_id}")
        else:
            print("❌ No saved states found!")

    def _process_frame(self, frame):
        """Process frame for AI input"""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84)) / 255.0  # Normalize

    def _save_segment(self):
        if len(self.current_segment) > 0:
            # Save frames and actions
            segment_path = os.path.join(self.save_dir, f"segment_{self.segment_id}.npz")
            frames = np.array([f[0] for f in self.current_segment])
            actions = np.array([f[1] for f in self.current_segment])
            np.savez(segment_path, frames=frames, actions=actions)
            
            # Save metadata
            metadata = {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'num_frames': len(self.current_segment),
                'ai_generated': self.ai_playing
            }
            with open(os.path.join(self.save_dir, f"segment_{self.segment_id}.json"), 'w') as f:
                json.dump(metadata, f)
            
            print(f"💿 Saved segment {self.segment_id} ({len(self.current_segment)} frames)")
            self.segment_id += 1

    def run(self):
        print("🕹️ KUNG FU MASTER TRAINER CONTROLS:")
        print("↑ ↓ ← → : Move/Jump/Duck")
        print("A: Punch | S: Kick")
        print("R: Start/Stop recording")
        print("O: Save state | P: Load state")
        print("M: Toggle AI control")
        print("Q: Quit")

        def on_press(key):
            try:
                if key.char.lower() == 'r':
                    if self.recording: self.stop_recording()
                    else: self.start_recording()
                elif key.char.lower() == 'o':
                    self.save_game_state()
                elif key.char.lower() == 'p':
                    self.load_game_state()
                elif key.char.lower() == 'm':
                    self.toggle_ai()
                elif key.char.lower() == 'q':
                    return False  # Quit
                elif key.char.lower() == 'a':
                    self.controls['a'] = True
                elif key.char.lower() == 's':
                    self.controls['s'] = True
            except AttributeError:
                if key == keyboard.Key.up: self.controls['up'] = True
                elif key == keyboard.Key.down: self.controls['down'] = True
                elif key == keyboard.Key.left: self.controls['left'] = True
                elif key == keyboard.Key.right: self.controls['right'] = True

        def on_release(key):
            try:
                if key.char.lower() == 'a':
                    self.controls['a'] = False
                elif key.char.lower() == 's':
                    self.controls['s'] = False
            except AttributeError:
                if key == keyboard.Key.up: self.controls['up'] = False
                elif key == keyboard.Key.down: self.controls['down'] = False
                elif key == keyboard.Key.left: self.controls['left'] = False
                elif key == keyboard.Key.right: self.controls['right'] = False

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        self.env.reset()
        try:
            while True:
                # Get action (AI or human)
                if self.ai_playing and self.model:
                    frame = self._process_frame(self.env.get_screen())
                    action, _ = self.model.predict(frame)
                else:
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

                # Record if enabled
                if self.recording:
                    self.current_segment.append((self._process_frame(obs), action))

                # Render at consistent speed
                self.env.render()
                time.sleep(0.02)  # ~50 FPS

                if done:
                    self.env.reset()
        finally:
            listener.stop()
            self.env.close()

if __name__ == "__main__":
    capturer = TrainingCapturer()
    capturer.run()