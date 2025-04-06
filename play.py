import argparse
import os
import retro
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import cv2
import logging
import sys
import time
from gym import spaces, Wrapper

# Assuming these classes are needed from your original code
class SimpleCNN(torch.nn.Module):
    def __init__(self, observation_space, features_dim=256):
        super(SimpleCNN, self).__init__()
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict"
        
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        combat_status_size = observation_space["combat_status"].shape[0]
        
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(n_flatten + enemy_vec_size + combat_status_size, features_dim),
            torch.nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        combat_status = observations["combat_status"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
        if isinstance(combat_status, np.ndarray):
            combat_status = torch.from_numpy(combat_status)
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and (viewport.shape[3] == 4 or viewport.shape[3] == 1):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        if len(enemy_vector.shape) == 1:
            enemy_vector = enemy_vector.unsqueeze(0)
        if len(combat_status.shape) == 1:
            combat_status = combat_status.unsqueeze(0)
            
        combined = torch.cat([cnn_output, enemy_vector, combat_status], dim=1)
        return self.linear(combined)

class KungFuWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.viewport_size = (84, 84)
        
        self.actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # No-op
            [0,0,0,0,0,0,1,0,0,0,0,0],  # Punch
            [0,0,0,0,0,0,0,0,1,0,0,0],  # Kick
            [1,0,0,0,0,0,1,0,0,0,0,0],  # Right+Punch
            [0,1,0,0,0,0,1,0,0,0,0,0],  # Left+Punch
            [0,0,0,0,0,1,0,0,0,0,0,0],  # Jump
            [0,0,1,0,0,0,0,0,0,0,0,0],  # Crouch
            [0,0,0,0,0,1,1,0,0,0,0,0],  # Jump+Punch
            [0,0,1,0,0,0,1,0,0,0,0,0]   # Crouch+Punch
        ]
        self.action_names = [
            "No-op", "Punch", "Kick", "Right+Punch", "Left+Punch",
            "Jump", "Crouch", "Jump+Punch", "Crouch+Punch"
        ]
        
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (8,), np.float32),
            "combat_status": spaces.Box(0, 1, (1,), np.float32)
        })
        
        self.last_hp = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * 4
        return self._get_obs(obs)

    def step(self, action):
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = ram[0x04A6]
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        
        self.last_hp = hp
        self.last_enemies = curr_enemies
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (sum(self.action_counts) + 1e-6),
            "action_names": self.action_names
        })
        
        return self._get_obs(obs), 0, done, info

    def _get_obs(self, obs):
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114])
        viewport = cv2.resize(gray, self.viewport_size)[..., np.newaxis]
        ram = self.env.get_ram()
        
        hero_x = int(ram[0x0094])
        enemy_info = []
        for addr in [0x008E, 0x008F, 0x0090, 0x0091]:
            enemy_x = int(ram[addr])
            if enemy_x != 0:
                distance = enemy_x - hero_x
                direction = 1 if distance > 0 else -1
                enemy_info.extend([direction, min(abs(distance), 255)])
            else:
                enemy_info.extend([0, 0])
        
        enemy_vector = np.array(enemy_info, dtype=np.float32)
        combat_status = np.array([self.last_hp/255.0], dtype=np.float32)
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector,
            "combat_status": combat_status
        }

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    return logging.getLogger()

def play(args):
    logger = setup_logging(args.log_dir)
    logger.info("Starting play session")
    
    model_file = args.model_path
    if not os.path.exists(model_file + ".zip"):
        logger.error(f"No trained model found at {model_file}.zip")
        sys.exit(1)

    # Create environment
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuWrapper(env)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    
    if args.render:
        env.envs[0].env.render_mode = 'human'
    
    # Load model
    model = PPO.load(model_file, env=env, device="cuda" if args.cuda else "cpu")
    
    obs = env.reset()
    total_hits = 0
    step_count = 0
    done = False

    try:
        logger.info("Starting gameplay...")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, info = env.step(action)
            total_hits += info[0]['enemy_hit']
            step_count += 1
            
            if step_count % 10 == 0 or info[0]['enemy_hit'] > 0:
                hit_rate = total_hits / (step_count + 1e-6)
                action_pct = info[0]['action_percentages']
                action_names = info[0]['action_names']
                logger.info(
                    f"Step: {step_count}, "
                    f"Hits: {total_hits}, "
                    f"Hit Rate: {hit_rate:.2%}, "
                    f"HP: {info[0]['hp']}, "
                    f"Actions: {[f'{name}:{pct:.1%}' for name, pct in zip(action_names, action_pct)]}"
                )
                if info[0]['enemy_hit'] > 0:
                    logger.info(">>> ENEMY HIT! <<<")
            
            if args.render:
                env.envs[0].env.render()
                time.sleep(0.02)  # Control frame rate

    except KeyboardInterrupt:
        logger.info("\nPlay interrupted by user")
    
    finally:
        hit_rate = total_hits / (step_count + 1e-6)
        logger.info(f"Game ended. Steps: {step_count}, Hits: {total_hits}, Hit Rate: {hit_rate:.2%}")
        
        if args.render:
            env.envs[0].env.close()
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Kung Fu with a trained PPO model")
    parser.add_argument("--model_path", default="models/kungfu_ppo/kungfu_ppo_best", help="Path to trained model file")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    
    args = parser.parse_args()
    play(args)