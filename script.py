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
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
from gym import spaces, Wrapper
from gym.wrappers import TimeLimit
from stable_baselines3.common.utils import set_random_seed

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

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, d_model=256, nhead=8, num_layers=3):  # Increased num_layers
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=d_model)
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        
        self.patch_size = 7
        self.num_patches = (84 // self.patch_size) ** 2
        self.patch_dim = self.patch_size * self.patch_size * 4  # 4 channels from frame stack
        
        self.patch_embedding = nn.Linear(self.patch_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model * self.num_patches, d_model)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = observations.shape
        x = observations.unfold(2, self.patch_size, self.patch_size)
        x = x.unfold(3, self.patch_size, self.patch_size)
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        num_patches = num_patches_height * num_patches_width
        x = x.contiguous().view(batch_size, channels, num_patches, self.patch_size * self.patch_size)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, num_patches, channels * self.patch_size * self.patch_size)
        x = self.patch_embedding(x)
        x = x + self.positional_encoding
        x = self.transformer_encoder(x)
        x = x.reshape(batch_size, -1)
        x = self.fc(x)
        return x

class TransformerPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super(TransformerPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=TransformerFeatureExtractor,
            features_extractor_kwargs=dict(d_model=256, nhead=8, num_layers=3),
            **kwargs
        )

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(10)
        self._actions = [
            [0,0,0,0,0,0,0,0,0,0,0,0],  # 0: No action
            [0,0,0,0,0,0,1,0,0,0,0,0],  # 1: Left
            [0,0,0,0,0,0,0,0,1,0,0,0],  # 2: Right
            [1,0,0,0,0,0,0,0,0,0,0,0],  # 3: B (kick)
            [0,1,0,0,0,0,0,0,0,0,0,0],  # 4: A (punch)
            [1,0,0,0,0,0,1,0,0,0,0,0],  # 5: B+Left
            [1,0,0,0,0,0,0,0,1,0,0,0],  # 6: B+Right
            [0,1,0,0,0,0,1,0,0,0,0,0],  # 7: A+Left
            [0,1,0,0,0,0,0,0,1,0,0,0],  # 8: A+Right
            [0,0,0,0,0,1,0,0,0,0,0,0]   # 9: Down (duck/dodge)
        ]
        self.action_names = ["No action", "Left", "Right", "Kick", "Punch", "Kick+Left", "Kick+Right", "Punch+Left", "Punch+Right", "Down"]

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
    def __init__(self, env):
        super().__init__(env)
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.last_hp = None
        self.total_hp_loss = 0
        self.ram_positions = {
            'score_1': 0x0531,  # Score digit 5 (hundred-thousands place)
            'score_2': 0x0532,  # Score digit 4 (ten-thousands place)
            'score_3': 0x0533,  # Score digit 3 (thousands place)
            'score_4': 0x0534,  # Score digit 2 (hundreds place)
            'score_5': 0x0535,  # Score digit 1 (tens place)
            'scroll_1': 0x00E5, # Screen Scroll 1 (primary scroll value)
            'scroll_2': 0x00D4, # Screen Scroll 2 (secondary, if needed)
            'current_stage': 0x0058,  # Current Stage (1-5)
            'hero_pos_x': 0x0094,  # Hero Screen Pos X
            'hero_hp': 0x04A6      # Hero HP
        }

    def reset(self):
        self.last_score = 0
        self.last_scroll = 0
        self.last_stage = 0
        self.total_hp_loss = 0
        obs = super().reset()
        ram = self.env.get_ram()
        self.last_hp = ram[self.ram_positions['hero_hp']]
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)
        
        ram = self.env.get_ram()
        current_score = (
            ram[self.ram_positions['score_1']] * 100000 +
            ram[self.ram_positions['score_2']] * 10000 +
            ram[self.ram_positions['score_3']] * 1000 +
            ram[self.ram_positions['score_4']] * 100 +
            ram[self.ram_positions['score_5']] * 10
        )
        current_scroll = int(ram[self.ram_positions['scroll_1']])
        current_stage = ram[self.ram_positions['current_stage']]
        hero_pos_x = ram[self.ram_positions['hero_pos_x']]
        current_hp = ram[self.ram_positions['hero_hp']]

        # Detect HP loss
        hp_loss = 0
        if self.last_hp is not None and current_hp < self.last_hp:
            hp_loss = self.last_hp - current_hp
            self.total_hp_loss += hp_loss
        self.last_hp = current_hp

        # Handle score overflow
        if current_score < self.last_score:
            score_delta = (999990 - self.last_score) + current_score
        else:
            score_delta = current_score - self.last_score

        # Calculate scroll progress
        scroll_delta = current_scroll - self.last_scroll
        if scroll_delta < 0:
            scroll_delta += 256

        # General reward function
        reward = 0
        if score_delta > 0:
            reward += score_delta * 10  # Reward for killing enemies
        if scroll_delta > 0:
            reward += scroll_delta * 50  # Reward for progress
        if current_stage > self.last_stage:
            reward += 1000  # Reward for stage completion
        if hp_loss > 0:
            reward -= hp_loss * 50  # Penalty for losing HP (encourages dodging)
        else:
            reward += 10  # Small reward for surviving each step

        # Update trackers
        self.last_score = current_score
        self.last_scroll = current_scroll
        self.last_stage = current_stage
        info['score'] = current_score
        info['hero_pos_x'] = hero_pos_x
        info['current_stage'] = current_stage
        info['hp'] = current_hp
        info['total_hp_loss'] = self.total_hp_loss

        return obs, reward, done, info
    
def make_kungfu_env(render=False, seed=None):
    env = retro.make(game='KungFu-Nes', use_restricted_actions=retro.Actions.ALL)
    env = KungFuDiscreteWrapper(env)
    env = PreprocessFrame(env)
    env = KungFuRewardWrapper(env)
    env = TimeLimit(env, max_episode_steps=5000)
    if seed is not None:
        env.seed(seed)
    if render:
        env.render_mode = 'human'
    return env

def make_env(rank, seed=0, render=False):
    def _init():
        env = make_kungfu_env(render=(rank == 0 and render), seed=seed + rank)
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
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    set_random_seed(args.seed)
    
    if args.num_envs > 1 and args.render:
        print("Warning: Rendering is only supported with num_envs=1. Setting num_envs to 1.")
        args.num_envs = 1
    
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed, args.render) for i in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed, args.render)])
    
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    env = VecMonitor(env, os.path.join(args.log_dir, 'monitor.csv'))
    
    expected_obs_space = spaces.Box(low=0, high=255, shape=(84, 84, 4), dtype=np.uint8)
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
    
    if args.resume and os.path.exists(model_file):
        print(f"Resuming training from {model_file}")
        try:
            loaded_model = PPO.load(model_file, device=device, print_system_info=True)
            old_obs_space = loaded_model.observation_space
            total_trained_steps = loaded_model.total_trained_steps if hasattr(loaded_model, 'total_trained_steps') else 0
            print(f"Total trained steps so far: {total_trained_steps}")
            
            if old_obs_space != env.observation_space:
                print("Observation space mismatch detected. Adapting model to new observation space...")
                model = PPO(
                    TransformerPolicy,
                    env,
                    learning_rate=1e-4,
                    n_steps=2048,
                    batch_size=64,
                    n_epochs=args.n_epochs,
                    gamma=args.gamma,
                    gae_lambda=args.gae_lambda,
                    clip_range=0.1,
                    ent_coef=0.2,  # Increased for more exploration
                    tensorboard_log=args.log_dir,
                    verbose=1 if args.verbose else 0,
                    device=device
                )
                old_policy_state_dict = loaded_model.policy.state_dict()
                new_policy_state_dict = model.policy.state_dict()
                for key in old_policy_state_dict:
                    if key in new_policy_state_dict and old_policy_state_dict[key].shape == new_policy_state_dict[key].shape:
                        new_policy_state_dict[key] = old_policy_state_dict[key]
                    else:
                        print(f"Skipping weight transfer for {key} due to shape mismatch.")
                model.policy.load_state_dict(new_policy_state_dict)
                model.total_trained_steps = total_trained_steps
                print("Weights transferred successfully where compatible.")
            else:
                model = PPO.load(
                    model_file,
                    env=env,
                    custom_objects={
                        "learning_rate": 1e-4,
                        "n_steps": 2048,
                        "batch_size": 64,
                        "n_epochs": args.n_epochs,
                        "gamma": args.gamma,
                        "gae_lambda": args.gae_lambda,
                        "clip_range": 0.1,
                        "ent_coef": 0.2,
                        "tensorboard_log": args.log_dir,
                        "total_trained_steps": total_trained_steps,
                        "policy": TransformerPolicy
                    },
                    device=device
                )
            print(f"Loaded model num_timesteps: {model.num_timesteps}")
        except Exception as e:
            print(f"Error during model loading/adaptation: {e}")
            print("Starting new training instead.")
            model = PPO(
                TransformerPolicy,
                env,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=args.n_epochs,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                clip_range=0.1,
                ent_coef=0.2,
                tensorboard_log=args.log_dir,
                verbose=1 if args.verbose else 0,
                device=device
            )
    else:
        print("Starting new training.")
        if os.path.exists(model_file):
            print(f"Warning: {model_file} exists but --resume not specified. Overwriting.")
        model = PPO(
            TransformerPolicy,
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=0.1,
            ent_coef=0.2,  # Increased for more exploration
            tensorboard_log=args.log_dir,
            verbose=1 if args.verbose else 0,
            device=device
        )
    
    global_model = model
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            reset_num_timesteps=False if args.resume and os.path.exists(model_file) else True
        )
        total_trained_steps = model.num_timesteps
        model.total_trained_steps = total_trained_steps
        model.save(model_file)
        print(f"Training completed. Model saved to {model_file}")
        print(f"Overall training steps: {total_trained_steps}")
        
        evaluate(args, model=model, baseline_file=args.baseline_file if hasattr(args, 'baseline_file') else None)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        logging.error(f"Error during training: {str(e)}")
        total_trained_steps = model.num_timesteps
        model.total_trained_steps = total_trained_steps
        model.save(model_file)
        print(f"Model saved to {model_file} due to error")
        print(f"Overall training steps: {total_trained_steps}")
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Error closing environment: {str(e)}")
        if terminate_flag:
            print("Training terminated by signal.")

def play(args):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('training.log'))
    
    env = make_kungfu_env(render=args.render)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    
    model_file = f"{args.model_path}/kungfu_ppo.zip"
    if not os.path.exists(model_file):
        print(f"No model found at {model_file}. Please train a model first.")
        return
    
    print(f"Loading model from {model_file}")
    model = PPO.load(model_file, env=env)
    total_trained_steps = model.total_trained_steps if hasattr(model, 'total_trained_steps') else 0
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
                total_reward += reward
                steps += 1
                
                action_name = env.envs[0].action_names[action[0]] if isinstance(action, np.ndarray) else env.envs[0].action_names[action]
                print(f"Step {steps}: Action={action_name}, HP={info[0]['hp']}, Score={info[0]['score']}")
                
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

def evaluate(args, model=None, baseline_file=None):
    if args.enable_file_logging:
        logging.getLogger().addHandler(logging.FileHandler('training.log'))
    
    env = make_kungfu_env(render=False)
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
    
    total_trained_steps = model.total_trained_steps if hasattr(model, 'total_trained_steps') else 0
    
    action_counts = np.zeros(10)
    total_steps = 0
    episode_lengths = []
    episode_scores = []
    max_positions = []
    max_stages = []
    hp_loss_rates = []
    action_history_per_episode = []
    
    for episode in range(args.eval_episodes):
        obs = env.reset()
        done = False
        episode_steps = 0
        episode_score = 0
        episode_max_pos_x = 0
        episode_max_stage = 0
        episode_hp_loss = 0
        episode_actions = []
        
        while not done and not terminate_flag:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            action_counts[action] += 1
            episode_actions.append(action)
            obs, reward, done, info = env.step(action)
            episode_steps += 1
            current_score = info[0].get('score', 0)
            episode_score = current_score
            hero_pos_x = info[0].get('hero_pos_x', 0)
            current_stage = info[0].get('current_stage', 0)
            episode_hp_loss = info[0].get('total_hp_loss', 0)
            episode_max_pos_x = max(episode_max_pos_x, hero_pos_x)
            episode_max_stage = max(episode_max_stage, current_stage)
            total_steps += 1
        
        episode_lengths.append(episode_steps)
        episode_scores.append(episode_score)
        max_positions.append(episode_max_pos_x)
        max_stages.append(episode_max_stage)
        hp_loss_rate = episode_hp_loss / episode_steps if episode_steps > 0 else 0
        hp_loss_rates.append(hp_loss_rate)
        action_history_per_episode.append(episode_actions)

    env.close()
    
    action_percentages = (action_counts / total_steps) * 100 if total_steps > 0 else np.zeros(10)
    avg_steps = np.mean(episode_lengths)
    avg_score = np.mean(episode_scores)
    avg_max_pos_x = np.mean(max_positions)
    avg_max_stage = np.mean(max_stages)
    avg_hp_loss_rate = np.mean(hp_loss_rates)
    action_entropy = -np.sum(action_percentages / 100 * np.log(action_percentages / 100 + 1e-10))
    
    baseline_stats = None
    if baseline_file and os.path.exists(baseline_file):
        baseline_env = make_kungfu_env(render=False)
        baseline_env = DummyVecEnv([lambda: baseline_env])
        baseline_env = VecFrameStack(baseline_env, n_stack=4)
        baseline_env = VecTransposeImage(baseline_env)
        baseline_model = PPO.load(baseline_file, env=baseline_env)
        
        baseline_steps = []
        baseline_scores = []
        baseline_max_positions = []
        baseline_max_stages = []
        baseline_hp_loss_rates = []
        for _ in range(args.eval_episodes):
            obs = baseline_env.reset()
            done = False
            steps = 0
            score = 0
            max_pos_x = 0
            max_stage = 0
            hp_loss = 0
            while not done:
                action, _ = baseline_model.predict(obs, deterministic=args.deterministic)
                obs, reward, done, info = baseline_env.step(action)
                steps += 1
                score = info[0].get('score', 0)
                max_pos_x = max(max_pos_x, info[0].get('hero_pos_x', 0))
                max_stage = max(max_stage, info[0].get('current_stage', 0))
                hp_loss = info[0].get('total_hp_loss', 0)
            hp_loss_rate = hp_loss / steps if steps > 0 else 0
            baseline_steps.append(steps)
            baseline_scores.append(score)
            baseline_max_positions.append(max_pos_x)
            baseline_max_stages.append(max_stage)
            baseline_hp_loss_rates.append(hp_loss_rate)
        baseline_env.close()
        baseline_stats = {
            'avg_steps': np.mean(baseline_steps),
            'avg_score': np.mean(baseline_scores),
            'avg_max_pos_x': np.mean(baseline_max_positions),
            'avg_max_stage': np.mean(baseline_max_stages),
            'avg_hp_loss_rate': np.mean(baseline_hp_loss_rates)
        }
    
    report = f"Evaluation Report for {model_file} ({args.eval_episodes} episodes)\n"
    report += f"Overall Training Steps: {total_trained_steps}\n"
    report += "-" * 50 + "\n"
    report += "Action Percentages:\n"
    for i, (name, percent) in enumerate(zip(env.envs[0].action_names, action_percentages)):
        report += f"  {name}: {percent:.2f}%\n"
    report += f"\nAction Distribution Entropy: {action_entropy:.3f} (higher = more diverse)\n"
    report += f"\nAverage Survival Time: {avg_steps:.2f} steps\n"
    report += f"Average Score per Episode: {avg_score:.2f}\n"
    report += f"Average Furthest Position (Hero Pos X): {avg_max_pos_x:.2f}\n"
    report += f"Average Highest Stage Reached: {avg_max_stage:.2f} (out of 5)\n"
    report += f"Average HP Loss Rate: {avg_hp_loss_rate:.3f} HP/step (lower = better survival)\n"
    
    if baseline_stats:
        report += "\nComparison with Baseline:\n"
        report += f"  Baseline Avg Survival Time: {baseline_stats['avg_steps']:.2f} steps\n"
        report += f"  Baseline Avg Score per Episode: {baseline_stats['avg_score']:.2f}\n"
        report += f"  Baseline Avg Furthest Position: {baseline_stats['avg_max_pos_x']:.2f}\n"
        report += f"  Baseline Avg Highest Stage: {baseline_stats['avg_max_stage']:.2f}\n"
        report += f"  Baseline Avg HP Loss Rate: {baseline_stats['avg_hp_loss_rate']:.3f} HP/step\n"
        report += f"  Plays Longer: {'Yes' if avg_steps > baseline_stats['avg_steps'] else 'No'} " \
                  f"(+{avg_steps - baseline_stats['avg_steps']:.2f} steps)\n"
        report += f"  Scores More: {'Yes' if avg_score > baseline_stats['avg_score'] else 'No'} " \
                  f"(+{avg_score - baseline_stats['avg_score']:.2f})\n"
        report += f"  Reaches Further: {'Yes' if avg_max_pos_x > baseline_stats['avg_max_pos_x'] else 'No'} " \
                  f"(+{avg_max_pos_x - baseline_stats['avg_max_pos_x']:.2f})\n"
        report += f"  Progresses Further (Stages): {'Yes' if avg_max_stage > baseline_stats['avg_max_stage'] else 'No'} " \
                  f"(+{avg_max_stage - baseline_stats['avg_max_stage']:.2f})\n"
        report += f"  Loses HP Slower: {'Yes' if avg_hp_loss_rate < baseline_stats['avg_hp_loss_rate'] else 'No'} " \
                  f"(-{baseline_stats['avg_hp_loss_rate'] - avg_hp_loss_rate:.3f} HP/step)\n"
    
    print(report)
    if args.enable_file_logging:
        with open(os.path.join(args.log_dir, 'evaluation_report.txt'), 'w') as f:
            f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play KungFu Master using PPO")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--train", action="store_true")
    mode_group.add_argument("--play", action="store_true")
    mode_group.add_argument("--evaluate", action="store_true")
    
    parser.add_argument("--model_path", default="models/kungfu_ppo")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timesteps", type=int, default=200_000)  # Increased for vision-based learning
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--progress_bar", action="store_true")
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions in play/eval mode")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.1)
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--baseline_file", type=str, default=None, help="Path to baseline model for comparison")
    parser.add_argument("--enable_file_logging", action="store_true", help="Enable logging to file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output from PPO")
    
    args = parser.parse_args()
    if not any([args.train, args.play, args.evaluate]):
        args.train = True
    
    if args.train:
        train(args)
    elif args.play:
        play(args)
    else:
        evaluate(args)