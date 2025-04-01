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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

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

# Vision Transformer for feature extraction
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

# Custom policy using Vision Transformer
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

# Discrete action space wrapper
class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(11)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 1: Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # 2: Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 3: B (Kick)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # 4: A (Punch)
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

# Frame preprocessing wrapper
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
    def observation(self, obs):
        obs = np.dot(obs[...,:3], [0.299, 0.587, 0.114])  # Grayscale
        obs = np.array(Image.fromarray(obs).resize((84, 84), Image.BILINEAR))
        obs = np.expand_dims(obs, axis=-1)
        return obs.astype(np.uint8)

class KungFuRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.last_hp = None
        self.total_hp_loss = 0
        self.prev_frame = None
        self.threats_history = []
        self.last_hero_pos_x = 0
        self.ram_positions = {
            'score_1': 0x0531, 'score_2': 0x0532, 'score_3': 0x0533, 'score_4': 0x0534, 'score_5': 0x0535,
            'scroll_1': 0x00E5, 'scroll_2': 0x00D4, 'current_stage': 0x0058, 'hero_pos_x': 0x0094, 
            'hero_hp': 0x04A6, 'hero_action': 0x0069, 'hero_air_mode': 0x036A,
        }

    def reset(self, **kwargs):
        # Reset the environment and get the initial observation
        obs = super().reset(**kwargs)
        ram = self.env.get_ram()
        self.last_hp = ram[self.ram_positions['hero_hp']]
        # Use the observation returned by reset instead of get_observation()
        self.prev_frame = self._preprocess_frame(obs)
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.total_hp_loss = 0
        self.threats_history = []
        self.last_hero_pos_x = 0
        return obs

    def _preprocess_frame(self, frame):
        # Ensure frame is in the expected format (84x84x1 from PreprocessFrame)
        if frame.shape[-1] == 1:  # Already preprocessed to grayscale
            return frame.squeeze(-1)  # Remove channel dim for processing
        else:
            # If raw frame is passed (unlikely due to PreprocessFrame wrapper), convert to grayscale
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114])
            frame = np.array(Image.fromarray(frame).resize((84, 84), Image.BILINEAR))
            return frame.astype(np.uint8)

    def _detect_threats(self, current_frame):
        if self.prev_frame is None:
            return []

        diff = np.abs(current_frame - self.prev_frame)
        threshold = 50
        speed_threshold = 5
        
        moving_pixels = diff > threshold
        if not np.any(moving_pixels):
            return []

        y_coords, x_coords = np.where(moving_pixels)
        if len(x_coords) == 0 or len(y_coords) == 0:
            return []

        threats = []
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        width = max_x - min_x
        height = max_y - min_y
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        is_fast_moving = False
        for prev_x, prev_y, _, _ in self.threats_history[-5:]:
            distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
            if distance > speed_threshold:
                is_fast_moving = True
                break

        if is_fast_moving and width > 0 and height > 0:
            threats.append((center_x, center_y, width, height))

        self.prev_frame = current_frame
        self.threats_history.append((center_x, center_y, width, height))
        self.threats_history = self.threats_history[-10:]
        return threats

    def step(self, action):
        obs, _, done, info = super().step(action)
        # Use the observation returned by step instead of get_observation()
        current_frame = self._preprocess_frame(obs)
        
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

        # Health loss
        hp_loss = 0
        if self.last_hp is not None and current_hp < self.last_hp:
            hp_loss = self.last_hp - current_hp
            self.total_hp_loss += hp_loss
        self.last_hp = current_hp

        # Score and scroll deltas
        score_delta = current_score - self.last_score if current_score >= self.last_score else (999990 - self.last_score) + current_score
        scroll_delta = current_scroll - self.last_scroll if current_scroll >= self.last_scroll else (current_scroll + 256 - self.last_scroll)

        # Base rewards
        reward = 0
        if score_delta > 0:
            reward += score_delta * 10
        if scroll_delta > 0:
            reward += scroll_delta * 50
        if current_stage > self.last_stage:
            reward += 1000
        if hp_loss > 0:
            reward -= hp_loss * 50
        else:
            reward += 10

        # Vision-based threat detection and response
        threats = self._detect_threats(current_frame)
        DISTANCE_THRESHOLD = 20
        SMALL_THREAT_SIZE = 20
        AVOIDANCE_REWARD = 100
        COMBAT_REWARD = 50
        GAP_CLOSING_BONUS = 10
        threat_detected = bool(threats)

        for threat_x, threat_y, width, height in threats:
            threat_x_game = threat_x * 256 // 84
            distance = abs(hero_pos_x - threat_x_game)

            if distance < DISTANCE_THRESHOLD:
                is_small_threat = width < SMALL_THREAT_SIZE and height < SMALL_THREAT_SIZE
                hero_action = ram[self.ram_positions['hero_action']]
                hero_air = ram[self.ram_positions['hero_air_mode']] == 1

                if is_small_threat:
                    if hero_air and threat_y < 42:
                        reward += AVOIDANCE_REWARD
                    elif hero_action == 0x05 and threat_y > 42:
                        reward += AVOIDANCE_REWARD
                else:
                    if hero_action in (0x01, 0x02):
                        reward += COMBAT_REWARD
                    elif hero_air and threat_y < 42:
                        reward += AVOIDANCE_REWARD
                    elif hero_action == 0x05 and threat_y > 42:
                        reward += AVOIDANCE_REWARD

            last_distance = abs(self.last_hero_pos_x - threat_x_game)
            if distance < last_distance:
                reward += GAP_CLOSING_BONUS

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

        return obs, reward, done, info
    
# State loader wrapper
class StateLoaderWrapper(Wrapper):
    def __init__(self, env, state_file="custom_state.state"):
        super().__init__(env)
        self.state_file = state_file
        if not os.path.exists(state_file):
            raise FileNotFoundError(f"State file {state_file} not found.")
        with open(state_file, "rb") as f:
            self.custom_state = f.read()
        self.env = TimeLimit(env, max_episode_steps=100)

    def reset(self):
        obs = super().reset()
        self.env.unwrapped.set_state(self.custom_state)
        return self.env.unwrapped.get_observation()

# Environment creation functions
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

# Callbacks
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

# Training function
def train(args):
    global global_model, global_model_path
    global_model_path = args.model_path
    
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('training.log'))
    
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
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
    
    # Define PPO hyperparameters
    ppo_kwargs = dict(
        policy=TransformerPolicy,
        env=env,
        learning_rate=args.learning_rate,  # Fixed the typo here
        n_steps=2048,
        batch_size=64,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=0.5,
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
            print(f"Loaded model num_timesteps: {model.num_timesteps}")
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

# Play function
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
                
                if args.render:
                    env.render()
                
                if terminate_flag:
                    break
            
            print(f"Episode {episode_count} - Total reward: {total_reward}, Steps: {steps}, Final HP: {info[0]['hp']}")
            if args.episodes > 0 and episode_count >= args.episodes:
                break
    
    except Exception as e:
        print(f"Error during play: {str(e)}")
        logging.error(f"Error during play: {str(e)}")
    finally:
        env.close()

# Capture function
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
    frame_time = 1 / 60  # 60 FPS
    
    print("Controls: Left/Right Arrows, Z (Punch), X (Kick), Up (Jump), Down (Duck)")
    print(f"Press 'S' to save state to '{args.state_file}', 'Q' to quit.")
    
    try:
        while not done:
            start_time = time.time()
            env.render()
            steps += 1
            
            action = 0  # Default: No action
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

# Evaluation function
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
    threat_avoidances = []
    
    for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_reward = 0
        episode_score = 0
        episode_max_pos_x = 0
        episode_max_stage = 0
        episode_hp_loss = 0
        episode_threat_avoidances = 0
        
        while not done and not terminate_flag:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            action_counts[action[0] if isinstance(action, np.ndarray) else action] += 1
            obs, reward, done, info = env.step(action)
            episode_steps += 1
            episode_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            episode_score = info[0].get('score', 0)
            hero_pos_x = info[0].get('hero_pos_x', 0)
            current_stage = info[0].get('current_stage', 0)
            episode_hp_loss = info[0].get('total_hp_loss', 0)
            if info[0].get('threat_detected', False):
                episode_threat_avoidances += 1
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
        threat_avoidances.append(episode_threat_avoidances)

    env.close()
    
    action_percentages = (action_counts / total_steps) * 100 if total_steps > 0 else np.zeros(11)
    avg_steps = np.mean(episode_lengths)
    avg_reward = np.mean(episode_rewards)
    avg_score = np.mean(episode_scores)
    avg_max_pos_x = np.mean(max_positions)
    avg_max_stage = np.mean(max_stages)
    avg_hp_loss_rate = np.mean(hp_loss_rates)
    avg_threat_avoidances = np.mean(threat_avoidances)
    action_entropy = -np.sum(action_percentages / 100 * np.log(action_percentages / 100 + 1e-10))
    
    report = f"Evaluation Report for {model_file} ({args.eval_episodes} episodes)\n"
    report += f"Overall Training Steps: {total_trained_steps}\n"
    report += "-" * 50 + "\n"
    report += "Action Percentages:\n"
    for i, (name, percent) in enumerate(zip(env.envs[0].action_names, action_percentages)):
        report += f"  {name}: {percent:.2f}%\n"
    report += f"\nAction Distribution Entropy: {action_entropy:.3f} (higher = more diverse)\n"
    report += f"Average Survival Time: {avg_steps:.2f} steps\n"
    report += f"Average Reward per Episode: {avg_reward:.2f}\n"
    report += f"Average Score per Episode: {avg_score:.2f}\n"
    report += f"Average Furthest Position (X): {avg_max_pos_x:.2f}\n"
    report += f"Average Highest Stage: {avg_max_stage:.2f} (out of 5)\n"
    report += f"Average HP Loss Rate: {avg_hp_loss_rate:.3f} HP/step\n"
    report += f"Average Threat Avoidances per Episode: {avg_threat_avoidances:.2f}\n"
    
    if baseline_file and os.path.exists(baseline_file):
        baseline_env = make_kungfu_env(render=False, state_only=args.state_only, state_file=args.state_file)
        baseline_env = DummyVecEnv([lambda: baseline_env])
        baseline_env = VecFrameStack(baseline_env, n_stack=4)
        baseline_env = VecTransposeImage(baseline_env)
        baseline_model = PPO.load(baseline_file, env=baseline_env)
        
        baseline_steps = []
        baseline_rewards = []
        baseline_scores = []
        baseline_max_positions = []
        baseline_max_stages = []
        baseline_hp_loss_rates = []
        baseline_threat_avoidances = []
        
        for _ in range(args.eval_episodes):
            obs = baseline_env.reset()
            done = False
            steps = 0
            reward_total = 0
            score = 0
            max_pos_x = 0
            max_stage = 0
            hp_loss = 0
            threat_avoidances_count = 0
            while not done:
                action, _ = baseline_model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = baseline_env.step(action)
                steps += 1
                reward_total += reward[0] if isinstance(reward, np.ndarray) else reward
                score = info[0].get('score', 0)
                max_pos_x = max(max_pos_x, info[0].get('hero_pos_x', 0))
                max_stage = max(max_stage, info[0].get('current_stage', 0))
                hp_loss = info[0].get('total_hp_loss', 0)
                if info[0].get('threat_detected', False):
                    threat_avoidances_count += 1
            baseline_steps.append(steps)
            baseline_rewards.append(reward_total)
            baseline_scores.append(score)
            baseline_max_positions.append(max_pos_x)
            baseline_max_stages.append(max_stage)
            hp_loss_rate = hp_loss / steps if steps > 0 else 0
            baseline_hp_loss_rates.append(hp_loss_rate)
            baseline_threat_avoidances.append(threat_avoidances_count)
        
        baseline_env.close()
        
        report += "\nComparison with Baseline:\n"
        report += f"  Baseline Avg Survival Time: {np.mean(baseline_steps):.2f} steps\n"
        report += f"  Baseline Avg Reward: {np.mean(baseline_rewards):.2f}\n"
        report += f"  Baseline Avg Score: {np.mean(baseline_scores):.2f}\n"
        report += f"  Baseline Avg Furthest Position: {np.mean(baseline_max_positions):.2f}\n"
        report += f"  Baseline Avg Highest Stage: {np.mean(baseline_max_stages):.2f}\n"
        report += f"  Baseline Avg HP Loss Rate: {np.mean(baseline_hp_loss_rates):.3f} HP/step\n"
        report += f"  Baseline Avg Threat Avoidances: {np.mean(baseline_threat_avoidances):.2f}\n"
    
    print(report)
    if args.enable_file_logging:
        with open(os.path.join(args.log_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)

# Main execution
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
    parser.add_argument("--baseline_file", type=str, default=None, help="Path to baseline model for comparison")
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