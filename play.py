import argparse
import os
import retro
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import logging
import sys
import time
from gym import spaces, Wrapper
import cv2

# Define SimpleCNN to match the training model's feature extractor
class SimpleCNN(nn.Module):
    def __init__(self, observation_space, features_dim=256):
        super(SimpleCNN, self).__init__()
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict"
        
        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        with torch.no_grad():
            sample_input = torch.zeros(1, 4, 84, 84)
            n_flatten = self.cnn(sample_input).shape[1]
        
        enemy_vec_size = observation_space["enemy_vector"].shape[0]
        enemy_types_size = observation_space["enemy_types"].shape[0]
        enemy_history_size = observation_space["enemy_history"].shape[0]
        enemy_timers_size = observation_space["enemy_timers"].shape[0]
        enemy_patterns_size = observation_space["enemy_patterns"].shape[0]
        combat_status_size = observation_space["combat_status"].shape[0]
        projectile_vec_size = observation_space["projectile_vectors"].shape[0]
        enemy_proximity_size = observation_space["enemy_proximity"].shape[0]
        boss_info_size = observation_space["boss_info"].shape[0]
        
        self.linear = nn.Sequential(
            nn.Linear(
                n_flatten + enemy_vec_size + enemy_types_size + enemy_history_size +
                enemy_timers_size + enemy_patterns_size + combat_status_size +
                projectile_vec_size + enemy_proximity_size + boss_info_size,
                features_dim
            ),
            nn.ReLU()
        )
    
    def forward(self, observations):
        viewport = observations["viewport"]
        enemy_vector = observations["enemy_vector"]
        enemy_types = observations["enemy_types"]
        enemy_history = observations["enemy_history"]
        enemy_timers = observations["enemy_timers"]
        enemy_patterns = observations["enemy_patterns"]
        combat_status = observations["combat_status"]
        projectile_vectors = observations["projectile_vectors"]
        enemy_proximity = observations["enemy_proximity"]
        boss_info = observations["boss_info"]
        
        if isinstance(viewport, np.ndarray):
            viewport = torch.from_numpy(viewport)
        if isinstance(enemy_vector, np.ndarray):
            enemy_vector = torch.from_numpy(enemy_vector)
        if isinstance(enemy_types, np.ndarray):
            enemy_types = torch.from_numpy(enemy_types)
        if isinstance(enemy_history, np.ndarray):
            enemy_history = torch.from_numpy(enemy_history)
        if isinstance(enemy_timers, np.ndarray):
            enemy_timers = torch.from_numpy(enemy_timers)
        if isinstance(enemy_patterns, np.ndarray):
            enemy_patterns = torch.from_numpy(enemy_patterns)
        if isinstance(combat_status, np.ndarray):
            combat_status = torch.from_numpy(combat_status)
        if isinstance(projectile_vectors, np.ndarray):
            projectile_vectors = torch.from_numpy(projectile_vectors)
        if isinstance(enemy_proximity, np.ndarray):
            enemy_proximity = torch.from_numpy(enemy_proximity)
        if isinstance(boss_info, np.ndarray):
            boss_info = torch.from_numpy(boss_info)
            
        if len(viewport.shape) == 3:
            viewport = viewport.unsqueeze(0)
        if len(viewport.shape) == 4 and (viewport.shape[3] == 4 or viewport.shape[3] == 1):
            viewport = viewport.permute(0, 3, 1, 2)
        
        cnn_output = self.cnn(viewport)
        
        if len(enemy_vector.shape) == 1:
            enemy_vector = enemy_vector.unsqueeze(0)
        if len(enemy_types.shape) == 1:
            enemy_types = enemy_types.unsqueeze(0)
        if len(enemy_history.shape) == 1:
            enemy_history = enemy_history.unsqueeze(0)
        if len(enemy_timers.shape) == 1:
            enemy_timers = enemy_timers.unsqueeze(0)
        if len(enemy_patterns.shape) == 1:
            enemy_patterns = enemy_patterns.unsqueeze(0)
        if len(combat_status.shape) == 1:
            combat_status = combat_status.unsqueeze(0)
        if len(projectile_vectors.shape) == 1:
            projectile_vectors = projectile_vectors.unsqueeze(0)
        if len(enemy_proximity.shape) == 1:
            enemy_proximity = enemy_proximity.unsqueeze(0)
        if len(boss_info.shape) == 1:
            boss_info = boss_info.unsqueeze(0)
            
        combined = torch.cat([
            cnn_output, enemy_vector, enemy_types, enemy_history, enemy_timers,
            enemy_patterns, combat_status, projectile_vectors, enemy_proximity, boss_info
        ], dim=1)
        return self.linear(combined)

# Define KungFuWrapper to match the training environment
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
        self.max_enemies = 4
        self.history_length = 5
        self.max_projectiles = 2
        self.patterns_length = 2
        
        self.observation_space = spaces.Dict({
            "viewport": spaces.Box(0, 255, (*self.viewport_size, 1), np.uint8),
            "enemy_vector": spaces.Box(-255, 255, (self.max_enemies * 2,), np.float32),
            "enemy_types": spaces.Box(0, 5, (self.max_enemies,), np.float32),
            "enemy_history": spaces.Box(-255, 255, (self.max_enemies * self.history_length * 2,), np.float32),
            "enemy_timers": spaces.Box(0, 255, (self.max_enemies,), np.float32),
            "enemy_patterns": spaces.Box(-255, 255, (self.max_enemies * self.patterns_length,), np.float32),
            "combat_status": spaces.Box(-1, 1, (2,), np.float32),
            "projectile_vectors": spaces.Box(-255, 255, (self.max_projectiles * 4,), np.float32),
            "enemy_proximity": spaces.Box(0, 1, (1,), np.float32),
            "boss_info": spaces.Box(-255, 255, (3,), np.float32)
        })
        
        self.last_hp = 0
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.last_hp = self.env.get_ram()[0x04A6]
        self.last_hp_change = 0
        self.action_counts = np.zeros(len(self.actions))
        self.last_enemies = [0] * self.max_enemies
        self.total_steps = 0
        self.prev_frame = None
        self.enemy_history = np.zeros((self.max_enemies, self.history_length, 2), dtype=np.float32)
        self.enemy_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        self.enemy_types = np.zeros(self.max_enemies, dtype=np.float32)
        self.enemy_timers = np.zeros(self.max_enemies, dtype=np.float32)
        self.boss_info = np.zeros(3, dtype=np.float32)
        self.enemy_patterns = np.zeros((self.max_enemies, self.patterns_length), dtype=np.float32)
        return self._get_obs(obs)

    def step(self, action):
        self.total_steps += 1
        self.action_counts[action] += 1
        obs, _, done, info = self.env.step(self.actions[action])
        ram = self.env.get_ram()
        
        hp = ram[0x04A6]
        curr_enemies = [int(ram[0x008E]), int(ram[0x008F]), int(ram[0x0090]), int(ram[0x0091])]
        enemy_hit = sum(1 for p, c in zip(self.last_enemies, curr_enemies) if p != 0 and c == 0)
        hp_change_rate = (hp - self.last_hp) / 255.0
        
        self._update_enemy_info(obs, ram)
        self._update_boss_info(ram)
        
        self.last_hp = hp
        self.last_hp_change = hp_change_rate
        self.last_enemies = curr_enemies
        
        info.update({
            "hp": hp,
            "enemy_hit": enemy_hit,
            "action_percentages": self.action_counts / (self.total_steps + 1e-6),
            "action_names": self.action_names
        })
        
        return self._get_obs(obs), 0, done, info

    def _update_enemy_info(self, obs, ram):
        hero_x = int(ram[0x0094])
        new_positions = np.zeros((self.max_enemies, 2), dtype=np.float32)
        new_types = np.zeros(self.max_enemies, dtype=np.float32)
        new_timers = np.zeros(self.max_enemies, dtype=np.float32)
        
        for i, (pos_addr, timer_addr) in enumerate([
            (0x008E, 0x002B),
            (0x008F, 0x002C),
            (0x0090, 0x002D),
            (0x0091, 0x002E)
        ]):
            enemy_x = int(ram[pos_addr])
            enemy_timer = int(ram[timer_addr])
            if enemy_x != 0:
                new_positions[i] = [enemy_x, 50]
                new_types[i] = 1 if enemy_x > hero_x else 2  # Simplified type assignment
                new_timers[i] = enemy_timer / 255.0
        
        for i in range(self.max_enemies):
            if new_types[i] != 0:
                dx = new_positions[i][0] - self.enemy_positions[i][0]
                dy = new_positions[i][1] - self.enemy_positions[i][1]
                self.enemy_history[i, :-1] = self.enemy_history[i, 1:]
                self.enemy_history[i, -1] = [dx, dy]
        
        self.enemy_positions = new_positions
        self.enemy_types = new_types
        self.enemy_timers = new_timers

    def _update_boss_info(self, ram):
        stage = int(ram[0x0058])
        if stage == 5:
            boss_pos_x = int(ram[0x0093])
            boss_hp = int(ram[0x04A5])
            self.boss_info = np.array([boss_pos_x - int(ram[0x0094]), 0, boss_hp / 255.0], dtype=np.float32)
        else:
            self.boss_info = np.zeros(3, dtype=np.float32)

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
        enemy_types = self.enemy_types
        enemy_history = self.enemy_history.reshape(-1)
        enemy_timers = self.enemy_timers
        enemy_patterns = self.enemy_patterns.reshape(-1)
        combat_status = np.array([self.last_hp/255.0, self.last_hp_change], dtype=np.float32)
        projectile_vectors = np.zeros(self.max_projectiles * 4, dtype=np.float32)  # Simplified
        enemy_proximity = np.array([1.0 if any(abs(enemy_x - hero_x) <= 20 for enemy_x in [int(ram[addr]) for addr in [0x008E, 0x008F, 0x0090, 0x0091]] if enemy_x != 0) else 0.0], dtype=np.float32)
        boss_info = self.boss_info
        
        return {
            "viewport": viewport.astype(np.uint8),
            "enemy_vector": enemy_vector,
            "enemy_types": enemy_types,
            "enemy_history": enemy_history,
            "enemy_timers": enemy_timers,
            "enemy_patterns": enemy_patterns,
            "combat_status": combat_status,
            "projectile_vectors": projectile_vectors,
            "enemy_proximity": enemy_proximity,
            "boss_info": boss_info
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
            total_hits += info[0].get('enemy_hit', 0)
            step_count += 1
            
            if step_count % 10 == 0 or info[0].get('enemy_hit', 0) > 0:
                hit_rate = total_hits / (step_count + 1e-6)
                action_pct = info[0].get('action_percentages', np.zeros(len(info[0].get('action_names', []))))
                action_names = info[0].get('action_names', [])
                logger.info(
                    f"Step: {step_count}, "
                    f"Hits: {total_hits}, "
                    f"Hit Rate: {hit_rate:.2%}, "
                    f"HP: {info[0].get('hp', 0)}, "
                    f"Actions: {[f'{name}:{pct:.1%}' for name, pct in zip(action_names, action_pct)]}"
                )
                if info[0].get('enemy_hit', 0) > 0:
                    logger.info(">>> ENEMY HIT! <<<")
            
            if args.render:
                env.envs[0].env.render()
                time.sleep(0.02)

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