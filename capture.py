import retro
import numpy as np
import cv2
import os
import json
import time
from collections import deque

class KungFuStateCapturer:
    def __init__(self, game_state='KungFu-Nes', save_dir='saved_states'):
        self.env = retro.make(game=game_state, use_restricted_actions=retro.Actions.ALL)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # State tracking variables
        self.current_stage = 0
        self.player_hp = 0
        self.enemy_count = 0
        self.boss_active = False
        self.last_capture_time = 0
        
        # Frame buffer for motion detection
        self.frame_buffer = deque(maxlen=5)
        
        # Define important RAM addresses
        self.ram_addrs = {
            'stage': 0x0058,
            'player_hp': 0x04A6,
            'enemies': [0x008E, 0x008F, 0x0090, 0x0091],
            'boss_hp': 0x04A5,
            'player_x': 0x0094,
            'player_y': 0x00B6,
            'boss_x': 0x0093,
            'boss_action': 0x004E
        }
        
    def get_ram_value(self, addr):
        """Read a value from RAM"""
        return self.env.get_ram()[addr]
    
    def get_game_state(self):
        """Get current game state information"""
        ram = self.env.get_ram()
        state = {
            'stage': int(ram[self.ram_addrs['stage']]),
            'player_hp': int(ram[self.ram_addrs['player_hp']]),
            'enemies': [int(ram[addr]) for addr in self.ram_addrs['enemies']],
            'boss_hp': int(ram[self.ram_addrs['boss_hp']]) if ram[self.ram_addrs['stage']] == 5 else 0,
            'player_pos': (
                int(ram[self.ram_addrs['player_x']]), 
                int(ram[self.ram_addrs['player_y']])
            ),
            'boss_pos': int(ram[self.ram_addrs['boss_x']]) if ram[self.ram_addrs['stage']] == 5 else 0,
            'boss_action': int(ram[self.ram_addrs['boss_action']]) if ram[self.ram_addrs['stage']] == 5 else 0,
            'timestamp': time.time()
        }
        return state
    
    def is_state_interesting(self, state):
        """Determine if the current state is worth capturing"""
        # Check if we just entered a new stage
        if state['stage'] != self.current_stage:
            self.current_stage = state['stage']
            return True
        
        # Check for boss fight start
        if state['stage'] == 5 and not self.boss_active and state['boss_hp'] > 0:
            self.boss_active = True
            return True
        
        # Check for low health situations
        if state['player_hp'] < 64 and state['player_hp'] != self.player_hp:
            self.player_hp = state['player_hp']
            return True
        
        # Check for enemy patterns
        active_enemies = sum(1 for e in state['enemies'] if e > 0)
        if active_enemies != self.enemy_count:
            self.enemy_count = active_enemies
            return active_enemies > 1  # Only capture when multiple enemies
        
        # Check for time-based capture (don't capture too frequently)
        if time.time() - self.last_capture_time > 10:  # At least 10 seconds between captures
            self.last_capture_time = time.time()
            return True
            
        return False
    
    def detect_combat_movement(self, frame):
        """Use motion detection to identify combat situations"""
        if len(self.frame_buffer) < self.frame_buffer.maxlen:
            self.frame_buffer.append(frame)
            return False
            
        # Convert frames to grayscale
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in self.frame_buffer]
        
        # Calculate frame differences
        diffs = []
        for i in range(1, len(gray_frames)):
            diff = cv2.absdiff(gray_frames[i-1], gray_frames[i])
            diffs.append(diff)
        
        # Average the differences
        if diffs:
            avg_diff = np.mean(diffs)
            motion_level = np.sum(avg_diff) / (avg_diff.size * 255)
            return motion_level > 0.2  # Threshold for significant motion
        return False
    
    def capture_state(self, state_id, description=""):
        """Capture and save the current game state"""
        # Get current state
        state = self.get_game_state()
        frame = self.env.render(mode='rgb_array')
        
        # Create state directory
        state_dir = os.path.join(self.save_dir, f"state_{state_id}")
        os.makedirs(state_dir, exist_ok=True)
        
        # Save frame as image
        frame_path = os.path.join(state_dir, "frame.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Save RAM state
        ram_path = os.path.join(state_dir, "ram.bin")
        with open(ram_path, 'wb') as f:
            f.write(bytes(self.env.get_ram()))
        
        # Save metadata
        metadata = {
            'state_id': state_id,
            'description': description,
            'game_state': state,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'frame_size': frame.shape
        }
        
        metadata_path = os.path.join(state_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved state {state_id} to {state_dir}")
        return state_dir
    
    def load_state(self, state_id):
        """Load a previously saved game state"""
        state_dir = os.path.join(self.save_dir, f"state_{state_id}")
        
        # Load RAM state
        ram_path = os.path.join(state_dir, "ram.bin")
        with open(ram_path, 'rb') as f:
            ram_state = bytearray(f.read())
        
        # Create a new environment with this state
        self.env.close()
        self.env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
        
        # For retro gym, we need to use the emulator's state loading functionality
        state_path = os.path.join(state_dir, "state.state")
        if os.path.exists(state_path):
            self.env.load_state(state_path)
        else:
            # Fallback: Direct RAM injection (may not work perfectly)
            # This is a hack and may not work reliably
            for i in range(len(ram_state)):
                self.env.get_ram()[i] = ram_state[i]
        
        # Load metadata
        metadata_path = os.path.join(state_dir, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        print(f"Loaded state {state_id}: {metadata.get('description', '')}")
        return metadata
    
    def auto_capture_loop(self, max_states=50, capture_interval=5):
        """Automatically capture interesting game states"""
        self.env.reset()
        state_id = 0
        last_capture = time.time()
        
        while state_id < max_states:
            # Take a random action
            action = self.env.action_space.sample()
            obs, _, done, _ = self.env.step(action)
            
            # Get current game state
            current_state = self.get_game_state()
            
            # Detect interesting situations
            if self.is_state_interesting(current_state) or \
            (time.time() - last_capture > capture_interval and self.detect_combat_movement(obs)):
                desc = self.generate_state_description(current_state)
                self.capture_state(state_id, desc)
                state_id += 1
                last_capture = time.time()
            
            if done:
                self.env.reset()
                
        print(f"Captured {state_id} game states")
    
    def generate_state_description(self, state):
        """Generate a human-readable description of the game state"""
        if state['stage'] != self.current_stage:
            return f"Stage {state['stage']} transition"
        
        if state['stage'] == 5 and state['boss_hp'] > 0:
            return f"Boss fight - HP: {state['boss_hp']}"
        
        if state['player_hp'] < 64:
            return f"Low health: {state['player_hp']}/255"
        
        active_enemies = sum(1 for e in state['enemies'] if e > 0)
        if active_enemies > 1:
            return f"Multiple enemies ({active_enemies})"
        
        return f"General gameplay state (Stage {state['stage']})"
    
    def close(self):
        self.env.close()

def main():
    capturer = KungFuStateCapturer()
    
    print("Kung Fu Master State Capturer")
    print("1. Auto-capture interesting states")
    print("2. Capture current state")
    print("3. Load saved state")
    print("4. Exit")
    
    try:
        while True:
            choice = input("Select option: ")
            
            if choice == '1':
                count = int(input("How many states to capture? (default 50): ") or 50)
                capturer.auto_capture_loop(max_states=count)
            elif choice == '2':
                state_id = input("Enter state ID: ")
                desc = input("Enter description (optional): ")
                capturer.capture_state(state_id, desc)
            elif choice == '3':
                state_id = input("Enter state ID to load: ")
                capturer.load_state(state_id)
                
                # Show the loaded state
                frame = capturer.env.render(mode='rgb_array')
                cv2.imshow('Loaded State', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            elif choice == '4':
                break
            else:
                print("Invalid choice")
    finally:
        capturer.close()

if __name__ == "__main__":
    main()