import argparse
import os
import gym
import retro
import numpy as np
import torch
import torch.nn as nn
import signal
import sys
import logging
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor, SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed
import keyboard
import time
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global_model = None
global_model_path = None
terminate_flag = False

def signal_handler(sig, frame):
    global terminate_flag, global_model, global_model_path
    print(f"Signal {sig} received! Preparing to terminate...")
    if global_model is not None and global_model_path is not None:
        global_model.save(f"{global_model_path}/kungfu_ppo")
        print(f"Emergency save completed: {global_model_path}/kungfu_ppo.zip")
    terminate_flag = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

class VisionTransformer(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, embed_dim=256, num_heads=8, num_layers=6, patch_size=14):
        super(VisionTransformer, self).__init__(observation_space, features_dim=embed_dim)
        self.img_size = observation_space.shape[1]
        self.in_channels = observation_space.shape[0]
        self.patch_size = patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.embed_dim = embed_dim

        self.patch_embed = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return x

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(TransformerPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=VisionTransformer,
            features_extractor_kwargs=dict(embed_dim=256, num_heads=8, num_layers=6, patch_size=14),
            **kwargs
        )

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(11)
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
            [0,0,0,0,0,1,0,0,0,0,0,0],  # 9: Down (Duck)
            [0,0,1,0,0,0,0,0,0,0,0,0]   # 10: Up (Jump)
        ]
        self.action_names = ["No action", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right", "Duck", "Jump"]

    def action(self, action):
        if isinstance(action, (list, np.ndarray)):
            action = int(action.item() if isinstance(action, np.ndarray) else action[0])
        return self._actions[action]

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
    def observation(self, obs):
        obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
        obs = np.array(Image.fromarray(obs).resize((84, 84), Image.BILINEAR))
        obs = np.expand_dims(obs, axis=-1)
        return obs.astype(np.uint8)

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env, threat_object_template_path="threat_object_template.png"):
        super().__init__(env)
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.last_hp = None
        self.total_hp_loss = 0
        self.last_hero_pos_x = 0
        self.threat_object_detections = 0
        self.threat_object_action_counts = np.zeros(11)
        self.detection_count = 0
        self.ranged_enemy_detections = 0
        self.last_ranged_detection_time = 0
        self.projectile_dodge_count = 0
        self.projectile_hit_count = 0
        
        # Load cropped threat_object template
        self.threat_object_template = cv2.imread(threat_object_template_path, cv2.IMREAD_GRAYSCALE)
        if self.threat_object_template is None:
            raise FileNotFoundError(f"Threat object template not found at {threat_object_template_path}")
        self.template_h, self.template_w = self.threat_object_template.shape
        
        # Ranged enemy template
        self.ranged_enemy_template = cv2.imread("ranged_enemy_template.png", cv2.IMREAD_GRAYSCALE)
        if self.ranged_enemy_template is None:
            logging.warning("Ranged enemy template not found, ranged enemy detection will be disabled")
            self.ranged_enemy_template = np.zeros((16, 16), dtype=np.uint8)
        self.ranged_template_h, self.ranged_template_w = self.ranged_enemy_template.shape
        
        self.ram_positions = {
            'score_1': 0x0531, 'score_2': 0x0532, 'score_3': 0x0533, 'score_4': 0x0534, 'score_5': 0x0535,
            'scroll_1': 0x00E5, 'scroll_2': 0x00D4, 'current_stage': 0x0058, 'hero_pos_x': 0x0094, 
            'hero_hp': 0x04A6, 'hero_air_mode': 0x036A,
            'enemy_type': 0x00E0, 'enemy_pos_x': 0x00E1, 'enemy_pos_y': 0x00E2
        }

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        ram = self.env.get_ram()
        self.last_hp = ram[self.ram_positions['hero_hp']]
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.total_hp_loss = 0
        self.last_hero_pos_x = 0
        self.threat_object_detections = 0
        self.threat_object_action_counts = np.zeros(11)
        self.detection_count = 0
        self.ranged_enemy_detections = 0
        self.last_ranged_detection_time = 0
        self.projectile_dodge_count = 0
        self.projectile_hit_count = 0
        return obs

    def _detect_threats(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        threats = []
        
        # Detect threat objects (projectiles)
        result = cv2.matchTemplate(gray_frame, self.threat_object_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        locations = np.where(result >= threshold)
        
        for pt in zip(*locations[::-1]):
            x, y = pt[0], pt[1]
            width, height = self.template_w, self.template_h
            center_x = x + width // 2
            center_y = y + height // 2
            
            threat_object_region = frame[y:y+height, x:x+width]
            threats.append(('projectile', center_x, center_y, width, height))
            break  # Limit to one detection per frame for simplicity
        
        # Detect ranged enemies (the throwers)
        if np.max(self.ranged_enemy_template) > 0:  # Only if template is valid
            enemy_result = cv2.matchTemplate(gray_frame, self.ranged_enemy_template, cv2.TM_CCOEFF_NORMED)
            enemy_threshold = 0.7
            enemy_locations = np.where(enemy_result >= enemy_threshold)
            
            for pt in zip(*enemy_locations[::-1]):
                x, y = pt[0], pt[1]
                width, height = self.ranged_template_w, self.ranged_template_h
                center_x = x + width // 2
                center_y = y + height // 2
                
                # Only consider enemies on the right side (approaching)
                if center_x > 112:  # Center of screen is ~112
                    threats.append(('ranged_enemy', center_x, center_y, width, height))
                    self.ranged_enemy_detections += 1
                    self.last_ranged_detection_time = time.time()
                    break  # Limit to one detection per frame
        
        return threats

    def step(self, action):
        obs, _, done, info = super().step(action)
        raw_frame = self.env.unwrapped.get_screen()  # RGB frame (224x256)
        
        ram = self.env.get_ram()
        current_score = (
            ram[self.ram_positions['score_1']] * 100000 + ram[self.ram_positions['score_2']] * 10000 +
            ram[self.ram_positions['score_3']] * 1000 + ram[self.ram_positions['score_4']] * 100 +
            ram[self.ram_positions['score_5']] * 10
        )
        current_scroll = int(ram[self.ram_positions['scroll_1']])
        current_stage = ram[self.ram_positions['current_stage']]
        hero_pos_x = ram[self.ram_positions['hero_pos_x']]
        current_hp = ram[self.ram_positions['hero_hp']]
        hero_air_mode = ram[self.ram_positions['hero_air_mode']]  # 0=ground, 1=air

        hp_loss = 0
        if self.last_hp is not None and current_hp < self.last_hp:
            hp_loss = self.last_hp - current_hp
            self.total_hp_loss += hp_loss
        self.last_hp = current_hp

        score_delta = current_score - self.last_score if current_score >= self.last_score else (999990 - self.last_score) + current_score
        scroll_delta = current_scroll - self.last_scroll if current_scroll >= self.last_scroll else (current_scroll + 256 - self.last_scroll)

        reward = 0
        if score_delta > 0:
            reward += score_delta * 5
        if scroll_delta > 0:
            reward += scroll_delta * 25
        if current_stage > self.last_stage:
            reward += 500
        if hp_loss > 0:
            reward -= hp_loss * 50
        else:
            reward += 10

        threats = self._detect_threats(raw_frame)
        PROJECTILE_DISTANCE_THRESHOLD = 30
        RANGED_ENEMY_DISTANCE_THRESHOLD = 50
        AVOIDANCE_REWARD = 5000
        APPROACH_REWARD = 2000
        RANGED_ENEMY_KILL_REWARD = 10000
        NON_AVOIDANCE_PENALTY = -1000
        PROJECTILE_HIT_PENALTY = -2000
        
        threat_detected = bool(threats)
        ranged_enemy_detected = any(t[0] == 'ranged_enemy' for t in threats)
        projectile_detected = any(t[0] == 'projectile' for t in threats)

        if threat_detected:
            self.threat_object_detections += 1
            self.threat_object_action_counts[action] += 1

            for threat_type, threat_x, threat_y, width, height in threats:
                threat_x_game = threat_x
                distance = abs(hero_pos_x - threat_x_game)
                
                if threat_type == 'projectile':
                    # Reward for dodging projectiles
                    if distance < PROJECTILE_DISTANCE_THRESHOLD:
                        if action == 10 and threat_y < 112:  # Jump over low projectile
                            reward += AVOIDANCE_REWARD
                            self.projectile_dodge_count += 1
                        elif action == 9 and threat_y > 112:  # Duck under high projectile
                            reward += AVOIDANCE_REWARD
                            self.projectile_dodge_count += 1
                        else:
                            reward += PROJECTILE_HIT_PENALTY
                            self.projectile_hit_count += 1
                    
                    # Additional reward for moving forward after dodging
                    if action in [2, 6, 8] and distance < PROJECTILE_DISTANCE_THRESHOLD * 2:
                        reward += APPROACH_REWARD / 2
                
                elif threat_type == 'ranged_enemy':
                    # Reward for approaching ranged enemies
                    if distance < RANGED_ENEMY_DISTANCE_THRESHOLD:
                        if action in [2, 6, 8]:  # Right movements
                            reward += APPROACH_REWARD
                        elif action in [3, 4, 5, 6, 7, 8]:  # Attack actions
                            reward += RANGED_ENEMY_KILL_REWARD
                    
                    # Penalty for staying still or moving away
                    if distance < RANGED_ENEMY_DISTANCE_THRESHOLD * 1.5 and action in [0, 1]:
                        reward += NON_AVOIDANCE_PENALTY / 2

                last_distance = abs(self.last_hero_pos_x - threat_x_game)
                if distance < last_distance and threat_type == 'ranged_enemy':
                    reward += 20  # Small reward for closing distance to ranged enemy

        # Additional reward for attacking when ranged enemy was recently detected
        if time.time() - self.last_ranged_detection_time < 2.0 and action in [3, 4, 5, 6, 7, 8]:
            reward += APPROACH_REWARD / 2

        self.last_hero_pos_x = hero_pos_x
        self.last_score = current_score
        self.last_scroll = current_scroll
        self.last_stage = current_stage
        
        info['score'] = current_score
        info['hero_pos_x'] = hero_pos_x
        info['current_stage'] = current_stage
        info['hp'] = current_hp
        info['total_hp_loss'] = self.total_hp_loss
        info['threat_detected'] = threat_detected
        info['projectile_detected'] = projectile_detected
        info['ranged_enemy_detected'] = ranged_enemy_detected
        info['threat_object_detections'] = self.threat_object_detections
        info['ranged_enemy_detections'] = self.ranged_enemy_detections
        info['projectile_dodge_count'] = self.projectile_dodge_count
        info['projectile_hit_count'] = self.projectile_hit_count
        info['threat_object_action_counts'] = self.threat_object_action_counts.copy()

        return obs, reward, done, info

def make_kungfu_env(render=False, seed=None, state_only=False, state_file="custom_state.state"):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    env = PreprocessFrame(env)
    env = KungFuRewardWrapper(env)
    if state_only:
        env = StateLoaderWrapper(env, state_file=state_file)
    else:
        env = TimeLimit(env, max_episode_steps=5000)
    if seed is not None:
        env.seed(seed)
    if render:
        env.render_mode = 'human'
    return env

def make_env(rank, seed=0, render=False, state_only=False, state_file="custom_state.state"):
    def _init():
        env = make_kungfu_env(render=(rank == 0 and render), seed=seed + rank, state_only=state_only, state_file=state_file)
        return env
    set_random_seed(seed)
    return _init

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RenderCallback, self).__init__(verbose)

    def _on_step(self):
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) == 1:
            self.training_env.render()
        return not terminate_flag

class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.update_interval = max(self.total_timesteps // 1000, 1)
        self.n_calls = 0

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.n_calls += 1
        if self.n_calls % self.update_interval == 0 or self.n_calls == self.total_timesteps:
            self.pbar.update(self.n_calls - self.pbar.n)
        return not terminate_flag

    def _on_training_end(self):
        self.pbar.n = self.total_timesteps
        self.pbar.close()

def train(args):
    global global_model, global_model_path
    global_model_path = args.model_path
    
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('training.log'))
    
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs("detected_threat_objects", exist_ok=True)
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    set_random_seed(args.seed)
    
    if args.num_envs > 1 and args.render:
        print("Warning: Rendering is only supported with num_envs=1. Setting num_envs to 1.")
        args.num_envs = 1
    
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed, args.render, args.state_only, args.state_file) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.render, args.state_only, args.state_file)])
    
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env, os.path.join(args.log_dir, 'monitor.csv'))
    
    expected_obs_space = spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    print(f"Current environment observation space: {env.observation_space}")
    if env.observation_space != expected_obs_space:
        print(f"Warning: Observation space mismatch. Expected {expected_obs_space}, got {env.observation_space}")

    callbacks = []
    if args.progress_bar:
        callbacks.append(TqdmCallback(total_timesteps=args.timesteps))
    if args.render:
        callbacks.append(RenderCallback())

    model_file = f"{args.model_path}/kungfu_ppo.zip"
    total_trained_steps = 0
    
    ppo_kwargs = dict(
        policy=TransformerPolicy,
        env=env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=0.01,
        tensorboard_log=args.log_dir,
        verbose=1 if args.verbose else 0,
        device=device
    )

    if os.path.exists(model_file) and args.resume:
        print(f"Resuming training from {model_file}")
        try:
            model = PPO.load(model_file, env=env, device=device, print_system_info=True)
            total_trained_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
            print(f"Total trained steps so far: {total_trained_steps}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting new training.")
            model = PPO(**ppo_kwargs)
    else:
        print("Starting new training.")
        if os.path.exists(model_file):
            print(f"Warning: {model_file} exists but --resume not specified. Overwriting.")
        model = PPO(**ppo_kwargs)
    
    global_model = model
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=False if args.resume and os.path.exists(model_file) else True
        )
        total_trained_steps = model.num_timesteps
        model.save(model_file)
        print(f"Training completed. Model saved to {model_file}")
        print(f"Overall training steps: {total_trained_steps}")
        
        evaluate(args, model=model)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        logging.error(f"Error during training: {str(e)}")
        total_trained_steps = model.num_timesteps
        model.save(model_file)
        print(f"Model saved due to error: {model_file}")
        print(f"Overall training steps: {total_trained_steps}")
    finally:
        env.close()
        if terminate_flag:
            print("Training terminated by signal.")

def play(args):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('play.log'))
    
    env = make_kungfu_env(render=args.render, state_only=args.state_only, state_file=args.state_file)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if not os.path.exists(model_file):
        print(f"No model found at {model_file}. Please train a model first.")
        return
    
    print(f"Loading model from {model_file}")
    model = PPO.load(model_file, env=env)
    total_trained_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    print(f"Overall training steps: {total_trained_steps}")
    
    episode_count = 0
    try:
        while not terminate_flag:
            obs = env.reset()
            done = False
            total_reward = 0
            steps = 0
            episode_count += 1
            print(f"Starting episode {episode_count}")
            
            while not done and not terminate_flag:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = env.step(action)
                total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
                steps += 1
                
                action_name = env.envs[0].action_names[action[0] if isinstance(action, np.ndarray) else action]
                print(f"Step {steps}: Action={action_name}, Reward={reward}, HP={info[0]['hp']}, Score={info[0]['score']}")
                
                if info[0]['projectile_detected']:
                    print(f"PROJECTILE DETECTED! Action taken: {action_name}")
                if info[0]['ranged_enemy_detected']:
                    print(f"RANGED ENEMY DETECTED! Action taken: {action_name}")
                
                if args.render:
                    env.render()
                
                if terminate_flag:
                    break
            
            print(f"Episode {episode_count} - Total reward: {total_reward}, Steps: {steps}, Final HP: {info[0]['hp']}")
            print(f"Projectile dodges: {info[0]['projectile_dodge_count']}, Projectile hits: {info[0]['projectile_hit_count']}")
            if args.episodes > 0 and episode_count >= args.episodes:
                break
    
    except Exception as e:
        print(f"Error during play: {str(e)}")
        logging.error(f"Error during play: {str(e)}")
    finally:
        env.close()

def capture(args):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('capture.log'))
    
    env = make_kungfu_env(render=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    obs = env.reset()
    done = False
    steps = 0
    frame_time = 1 / 60
    
    print("Controls: Left/Right Arrows, Z (Punch), X (Kick), Up (Jump), Down (Duck)")
    print("Press 'S' to save state, 'K' to save threat_object template, 'E' to save ranged enemy template, 'Q' to quit.")
    
    try:
        while not done:
            start_time = time.time()
            env.render()
            steps += 1
            
            action = 0
            if keyboard.is_pressed('left'):
                action = 1
            elif keyboard.is_pressed('right'):
                action = 2
            elif keyboard.is_pressed('z'):
                action = 4
            elif keyboard.is_pressed('x'):
                action = 3
            elif keyboard.is_pressed('up'):
                action = 10
            elif keyboard.is_pressed('down'):
                action = 9
            
            obs, reward, done, info = env.step([action])
            logging.info(f"Step {steps}: Action={env.envs[0].action_names[action]}, Reward={reward[0]}, HP={info[0]['hp']}, Score={info[0]['score']}")
            
            if keyboard.is_pressed('s'):
                with open(args.state_file, "wb") as f:
                    f.write(env.envs[0].unwrapped.get_state())
                print(f"State saved to '{args.state_file}' at step {steps}")
                logging.info(f"State saved to '{args.state_file}' at step {steps}")
            
            if keyboard.is_pressed('k'):
                frame = env.envs[0].unwrapped.get_screen()
                threat_object_x, threat_object_y = 112, 128  # Center of screen
                threat_object_w, threat_object_h = 32, 16    # Adjust based on threat_object size
                threat_object_region = frame[threat_object_y-threat_object_h//2:threat_object_y+threat_object_h//2, threat_object_x-threat_object_w//2:threat_object_x+threat_object_w//2]
                cv2.imwrite("threat_object_template.png", cv2.cvtColor(threat_object_region, cv2.COLOR_RGB2BGR))
                print(f"Threat object template saved to 'threat_object_template.png' at step {steps}")
                logging.info(f"Threat object template saved at step {steps}")
            
            if keyboard.is_pressed('e'):
                frame = env.envs[0].unwrapped.get_screen()
                # Capture the ranged enemy (usually on the right side)
                enemy_x, enemy_y = 180, 128  # Adjust based on enemy position
                enemy_w, enemy_h = 16, 16
                enemy_region = frame[enemy_y-enemy_h//2:enemy_y+enemy_h//2, enemy_x-enemy_w//2:enemy_x+enemy_w//2]
                cv2.imwrite("ranged_enemy_template.png", cv2.cvtColor(enemy_region, cv2.COLOR_RGB2BGR))
                print(f"Ranged enemy template saved to 'ranged_enemy_template.png' at step {steps}")
                logging.info(f"Ranged enemy template saved at step {steps}")
            
            if keyboard.is_pressed('q'):
                print("Quitting...")
                break
            
            elapsed_time = time.time() - start_time
            time.sleep(max(0, frame_time - elapsed_time))
    
    except Exception as e:
        print(f"Error during capture: {str(e)}")
        logging.error(f"Error during capture: {str(e)}")
    finally:
        env.close()

def evaluate(args, model=None, baseline_file=None):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('evaluate.log'))
    
    env = make_kungfu_env(render=False, state_only=args.state_only, state_file=args.state_file)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if model is None:
        if not os.path.exists(model_file):
            print(f"No model found at {model_file}. Please train a model first.")
            return
        print(f"Loading model from {model_file}")
        model = PPO.load(model_file, env=env)
    
    total_trained_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    
    action_counts = np.zeros(11)
    total_steps = 0
    episode_lengths = []
    episode_rewards = []
    episode_scores = []
    max_positions = []
    max_stages = []
    hp_loss_rates = []
    threat_object_detections_per_episode = []
    threat_object_action_totals = np.zeros(11)
    projectile_dodge_counts = []
    projectile_hit_counts = []
    ranged_enemy_detections_per_episode = []
    
    for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0
        episode_score = 0
        episode_max_pos_x = 0
        episode_max_stage = 0
        episode_hp_loss = 0
        episode_threat_object_detections = 0
        episode_threat_object_action_counts = np.zeros(11)
        episode_projectile_dodge_count = 0
        episode_projectile_hit_count = 0
        episode_ranged_enemy_detections = 0
        
        while not done and not terminate_flag:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            action = action[0] if isinstance(action, np.ndarray) else action
            action_counts[action] += 1
            obs, reward, done, info = env.step([action])
            episode_steps += 1
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_score = info[0].get('score', 0)
            hero_pos_x = info[0].get('hero_pos_x', 0)
            current_stage = info[0].get('current_stage', 0)
            episode_hp_loss = info[0].get('total_hp_loss', 0)
            episode_threat_object_detections = info[0].get('threat_object_detections', 0)
            episode_threat_object_action_counts = info[0].get('threat_object_action_counts', np.zeros(11))
            episode_projectile_dodge_count = info[0].get('projectile_dodge_count', 0)
            episode_projectile_hit_count = info[0].get('projectile_hit_count', 0)
            episode_ranged_enemy_detections = info[0].get('ranged_enemy_detections', 0)
            episode_max_pos_x = max(episode_max_pos_x, hero_pos_x)
            episode_max_stage = max(episode_max_stage, current_stage)
            total_steps += 1
        
        episode_lengths.append(episode_steps)
        episode_rewards.append(episode_reward)
        episode_scores.append(episode_score)
        max_positions.append(episode_max_pos_x)
        max_stages.append(episode_max_stage)
        hp_loss_rate = episode_hp_loss / episode_steps if episode_steps > 0 else 0
        hp_loss_rates.append(hp_loss_rate)
        threat_object_detections_per_episode.append(episode_threat_object_detections)
        threat_object_action_totals += episode_threat_object_action_counts
        projectile_dodge_counts.append(episode_projectile_dodge_count)
        projectile_hit_counts.append(episode_projectile_hit_count)
        ranged_enemy_detections_per_episode.append(episode_ranged_enemy_detections)

    env.close()
    
    action_percentages = (action_counts / total_steps) * 100 if total_steps > 0 else np.zeros(11)
    avg_steps = np.mean(episode_lengths)
    avg_reward = np.mean(episode_rewards)
    avg_score = np.mean(episode_scores)
    avg_max_pos_x = np.mean(max_positions)
    avg_max_stage = np.mean(max_stages)
    avg_hp_loss_rate = np.mean(hp_loss_rates)
    avg_threat_object_detections = np.mean(threat_object_detections_per_episode)
    total_threat_object_detections = np.sum(threat_object_detections_per_episode)
    threat_object_action_percentages = (threat_object_action_totals / total_threat_object_detections * 100) if total_threat_object_detections > 0 else np.zeros(11)
    action_entropy = -np.sum(action_percentages / 100 * np.log(action_percentages / 100 + 1e-10))
    avg_projectile_dodges = np.mean(projectile_dodge_counts)
    avg_projectile_hits = np.mean(projectile_hit_counts)
    avg_ranged_enemy_detections = np.mean(ranged_enemy_detections_per_episode)
    projectile_dodge_ratio = avg_projectile_dodges / (avg_projectile_dodges + avg_projectile_hits + 1e-10)
    
    report = f"Evaluation Report for {model_file} ({args.eval_episodes} episodes)\n"
    report += f"Overall Training Steps: {total_trained_steps}\n"
    report += "-" * 50 + "\n"
    report += "Action Percentages (Overall):\n"
    for i, (name, percent) in enumerate(zip(env.envs[0].action_names, action_percentages)):
        report += f"  {name}: {percent:.2f}%\n"
    report += f"\nAction Distribution Entropy: {action_entropy:.3f} (higher = more diverse)\n"
    report += f"Average Survival Time: {avg_steps:.2f} steps\n"
    report += f"Average Reward per Episode: {avg_reward:.2f}\n"
    report += f"Average Score per Episode: {avg_score:.2f}\n"
    report += f"Average Furthest Position (X): {avg_max_pos_x:.2f}\n"
    report += f"Average Highest Stage: {avg_max_stage:.2f} (out of 5)\n"
    report += f"Average HP Loss Rate: {avg_hp_loss_rate:.3f} HP/step\n"
    report += f"\nThreat Object Stats:\n"
    report += f"  Average Threat Object Detections per Episode: {avg_threat_object_detections:.2f}\n"
    report += f"  Total Threat Object Detections: {int(total_threat_object_detections)}\n"
    report += f"  Average Ranged Enemy Detections per Episode: {avg_ranged_enemy_detections:.2f}\n"
    report += f"  Average Projectile Dodges per Episode: {avg_projectile_dodges:.2f}\n"
    report += f"  Average Projectile Hits per Episode: {avg_projectile_hits:.2f}\n"
    report += f"  Projectile Dodge Ratio: {projectile_dodge_ratio:.2%}\n"
    report += "  Action Percentages When Threat Objects Detected:\n"
    for i, (name, percent) in enumerate(zip(env.envs[0].action_names, threat_object_action_percentages)):
        report += f"    {name}: {percent:.2f}%\n"
    
    print(report)
    if args.enable_file_logging:
        with open(os.path.join(args.log_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train, play, capture, or evaluate KungFu Master using PPO with Vision Transformer")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true", help="Train the model")
    mode_group.add_argument("--play", action="store_true", help="Play with the trained model")
    mode_group.add_argument("--capture", action="store_true", help="Manually play and capture a state")
    mode_group.add_argument("--evaluate", action="store_true", help="Evaluate the trained model")
    
    parser.add_argument("--model_path", default="models/kungfu_ppo", help="Path to save/load kungfu_ppo.zip")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--render", action="store_true", help="Render the game")
    parser.add_argument("--resume", action="store_true", help="Resume training from existing model")
    parser.add_argument("--timesteps", type=int, default=500000, help="Total timesteps for training")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--progress_bar", action="store_true", help="Show training progress bar")
    parser.add_argument("--eval_episodes", type=int, default=10, help="Number of episodes for evaluation")
    parser.add_argument("--episodes", type=int, default=0, help="Number of episodes to play (0 = infinite)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions in play/eval")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for PPO")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=0.1, help="PPO clip range")
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument("--enable_file_logging", action="store_true", help="Enable logging to file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose PPO output")
    parser.add_argument("--state_only", action="store_true", help="Train/play on custom state")
    parser.add_argument("--state_file", default="custom_state.state", help="File for custom state")
    
    args = parser.parse_args()
    if not any([args.train, args.play, args.capture, args.evaluate]):
        args.train = True
    
    if args.train:
        train(args)
    elif args.play:
        play(args)
    elif args.capture:
        capture(args)
    else:
        evaluate(args)