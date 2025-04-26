import os
import sys
import argparse
import numpy as np
import retro
import torch
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from datetime import datetime
from tqdm import tqdm
import multiprocessing as mp
from typing import List, Optional, Tuple, Union, Dict, Any

# Define the Kung Fu Master action space
KUNGFU_ACTIONS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # No-op
    [1, 0, 0, 0, 0, 0, 0, 0, 0],  # B (Punch)
    [0, 0, 1, 0, 0, 0, 0, 0, 0],  # SELECT
    [0, 0, 0, 1, 0, 0, 0, 0, 0],  # START
    [0, 0, 0, 0, 1, 0, 0, 0, 0],  # UP (Jump)
    [0, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN (Crouch)
    [0, 0, 0, 0, 0, 0, 1, 0, 0],  # LEFT
    [0, 0, 0, 0, 0, 0, 0, 1, 0],  # RIGHT
    [0, 0, 0, 0, 0, 0, 0, 0, 1],  # A (Kick)
    [1, 0, 0, 0, 0, 0, 0, 0, 1],  # B + A (Punch + Kick)
    [0, 0, 0, 0, 1, 0, 0, 1, 0],  # UP + RIGHT (Jump + Right)
    [0, 0, 0, 0, 0, 1, 0, 0, 1],  # DOWN + A (Crouch Kick)
    [1, 0, 0, 0, 0, 1, 0, 0, 0],  # DOWN + B (Crouch Punch)
]

KUNGFU_ACTION_NAMES = [
    "No-op",
    "Punch",
    "Select",
    "Start",
    "Jump",
    "Crouch",
    "Left",
    "Right",
    "Kick",
    "Punch + Kick",
    "Jump + Right",
    "Crouch Kick",
    "Crouch Punch",
]

# Set model path
MODEL_PATH = "model/kungfu.zip"


# Helper class for SubprocVecEnv
class CloudpickleWrapper:
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        import pickle

        self.var = pickle.loads(obs)


# Worker function for SubprocVecEnv with robust step handling
def _worker(remote, parent_remote, env_fn_wrapper, seed=0):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                step_result = env.step(data)
                if len(step_result) == 4:  # Old Gym API
                    obs, rew, done, info = step_result
                    term, trunc = done, False
                elif len(step_result) == 5:  # Gymnasium API
                    obs, rew, term, trunc, info = step_result
                else:
                    raise ValueError(f"Step returned {len(step_result)} values")
                if not isinstance(info, dict):
                    print(
                        f"Worker warning: info is {type(info)}: {info}, resetting to dict"
                    )
                    info = {}
                remote.send((obs, rew, term, trunc, info))
            elif cmd == "reset":
                try:
                    maybe_options = {}
                    if isinstance(data, tuple) and len(data) > 0:
                        seed_val = data[0] if data[0] is not None else None
                        maybe_options = data[1] if len(data) > 1 else {}
                        if seed_val is not None:
                            maybe_options["seed"] = seed_val
                    reset_result = env.reset(**maybe_options)
                    if isinstance(reset_result, tuple) and len(reset_result) == 2:
                        observation, reset_info = reset_result
                    else:
                        observation = reset_result
                        reset_info = {}
                    remote.send((observation, reset_info))
                except Exception as e:
                    print(f"Worker reset error: {e}")
                    observation = env.reset()
                    remote.send((observation, {}))
            elif cmd == "render":
                remote.send(env.render(data))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "env_method":
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(getattr(env, data))
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            break
        except Exception as e:
            print(f"Error in worker: {e}")
            remote.send((None, None))
            break


# Fixed SubprocVecEnv with corrected reset
class FixedSubprocVecEnv:
    def __init__(self, env_fns, start_method=None):
        self.waiting = False
        self.closed = False
        self.num_envs = len(env_fns)
        self._seeds = [None for _ in range(self.num_envs)]
        self._options = [{} for _ in range(self.num_envs)]

        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.num_envs)]
        )
        self.processes = []
        for i, (work_remote, remote, env_fn) in enumerate(
            zip(self.work_remotes, self.remotes, env_fns)
        ):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        options = {} if options is None else options
        if seed is not None:
            if isinstance(seed, int):
                seeds = [seed + idx for idx in range(self.num_envs)]
            else:
                seeds = seed
        else:
            seeds = [None for _ in range(self.num_envs)]

        for idx, remote in enumerate(self.remotes):
            try:
                remote.send(("reset", (seeds[idx], options)))
            except Exception as e:
                print(f"Error sending reset command to env {idx}: {e}")
                remote.send(("reset", {}))

        obs = []
        for idx, remote in enumerate(self.remotes):
            try:
                result = remote.recv()
                if isinstance(result, tuple) and len(result) == 2:
                    o, i = result
                    if i:
                        print(f"Env {idx} reset info: {i}")
                else:
                    o = result
                    i = {}
                obs.append(o)
            except Exception as e:
                print(f"Error receiving reset result from env {idx}: {e}")
                obs.append(np.zeros(self.observation_space.shape))
        return np.stack(obs)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        for i, res in enumerate(results):
            if res is None or len(res) != 5:
                print(f"Invalid result from env {i}: {res}")
                results[i] = (
                    np.zeros(self.observation_space.shape),
                    0.0,
                    True,
                    False,
                    {},
                )
        obs, rews, terms, truncs, infos = zip(*results)
        return np.stack(obs), np.array(rews), np.array(terms), np.array(truncs), infos

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True


# Custom environment wrapper for Kung Fu Master
class KungFuMasterEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        filtered_actions = [action for i, action in enumerate(KUNGFU_ACTIONS) if i != 3]
        filtered_action_names = [
            name for i, name in enumerate(KUNGFU_ACTION_NAMES) if i != 3
        ]
        self.KUNGFU_ACTIONS = filtered_actions
        self.KUNGFU_ACTION_NAMES = filtered_action_names
        self.action_space = gym.spaces.Discrete(len(self.KUNGFU_ACTIONS))
        self.prev_score = 0
        self.prev_hp = 0
        self.prev_x_pos = 0
        self.prev_boss_hp = 0
        self.prev_stage = 0
        self.prev_enemy_x = [0, 0, 0, 0]
        self.prev_enemy_actions = [0, 0, 0, 0]
        self.SCORE_REWARD_SCALE = 0.01
        self.HP_LOSS_PENALTY = -1.0
        self.STAGE_COMPLETION_REWARD = 50.0
        self.DEATH_PENALTY = -25.0
        self.PROGRESS_REWARD_SCALE = 0.05
        self.BOSS_DAMAGE_REWARD = 1.0
        self.TIME_PRESSURE_PENALTY = -0.01
        self.ENEMY_PROXIMITY_AWARENESS = 0.5
        self.ENEMY_DEFEAT_BONUS = 2.0

    def reset(self, **kwargs):
        try:
            reset_result = self.env.reset(**kwargs)
            if isinstance(reset_result, tuple) and len(reset_result) == 2:
                obs, info = reset_result
            else:
                obs = reset_result
                info = {}
        except Exception as e:
            print(f"Error in initial reset: {e}")
            obs = self.env.reset()
            info = {}

        start_action = KUNGFU_ACTIONS[3]
        max_attempts = 50
        attempt = 0
        game_started = False

        print("Attempting to start the game...")
        while attempt < max_attempts and not game_started:
            attempt += 1
            print(f"Start attempt {attempt}/{max_attempts}")
            for _ in range(10):  # Press START for 10 frames
                obs, _, _, _, info = self.env.step(start_action)
            for _ in range(20):  # Wait for 20 frames
                obs, _, _, _, info = self.env.step(KUNGFU_ACTIONS[0])
            ram = self.env.get_ram()
            current_stage = int(ram[0x0058])
            player_hp = int(ram[0x04A6])
            game_state = int(ram[0x0001])
            player_x = int(ram[0x0094])
            print(
                f"RAM Check: stage={current_stage}, hp={player_hp}, game_state={game_state}, x_pos={player_x}"
            )
            if player_hp > 0 and (current_stage > 0 or player_x != 128):
                game_started = True
                print(f"Game started successfully after {attempt} attempts")
                break
            if attempt % 10 == 0:
                print(f"Still trying to start game... attempt {attempt}")

        if not game_started:
            print("ERROR: Failed to start game after maximum attempts.")
            print(f"Final RAM state: {ram[:10]}...")

        try:
            ram = self.env.get_ram()
            self.prev_score = self._get_score(ram)
            self.prev_hp = int(ram[0x04A6])
            self.prev_x_pos = int(ram[0x0094])
            self.prev_boss_hp = int(ram[0x04A5])
            self.prev_stage = int(ram[0x0058])
            self.prev_enemy_x = [
                int(ram[0x008E]),
                int(ram[0x008F]),
                int(ram[0x0090]),
                int(ram[0x0091]),
            ]
            self.prev_enemy_actions = [
                int(ram[0x0080]),
                int(ram[0x0081]),
                int(ram[0x0082]),
                int(ram[0x0083]),
            ]
        except Exception as e:
            print(f"Error initializing state tracking: {e}")

        if kwargs.get("return_info", False):
            return obs, info
        return obs

    def step(self, action):
        if action == 3:
            action = 0
        converted_action = self.KUNGFU_ACTIONS[action]
        step_result = self.env.step(converted_action)
        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        elif len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            raise ValueError(f"Unexpected step result length: {len(step_result)}")
        if not isinstance(info, dict):
            print(f"Warning: info is {type(info)}: {info}, resetting to dict")
            info = {}
        done = terminated or truncated

        try:
            ram = self.env.get_ram()
            current_stage = int(ram[0x0058])
            current_score = self._get_score(ram)
            current_hp = int(ram[0x04A6])
            current_x_pos = int(ram[0x0094])
            current_y_pos = int(ram[0x00B6])
            current_boss_hp = int(ram[0x04A5])
            current_enemy_x = [
                int(ram[0x008E]),
                int(ram[0x008F]),
                int(ram[0x0090]),
                int(ram[0x0091]),
            ]
            current_enemy_actions = [
                int(ram[0x0080]),
                int(ram[0x0081]),
                int(ram[0x0082]),
                int(ram[0x0083]),
            ]

            shaped_reward = 0.0
            shaped_reward += reward
            score_diff = self._safe_subtract(current_score, self.prev_score)
            if score_diff > 0:
                shaped_reward += score_diff * self.SCORE_REWARD_SCALE
            hp_diff = self._safe_subtract(current_hp, self.prev_hp)
            if hp_diff < 0:
                shaped_reward += hp_diff * self.HP_LOSS_PENALTY
            direction_reward = self._calculate_direction_reward(
                current_stage, current_x_pos, self.prev_x_pos
            )
            shaped_reward += direction_reward
            boss_hp_diff = self._safe_subtract(current_boss_hp, self.prev_boss_hp)
            if boss_hp_diff < 0:
                shaped_reward += abs(boss_hp_diff) * self.BOSS_DAMAGE_REWARD
            if current_stage > self.prev_stage:
                shaped_reward += self.STAGE_COMPLETION_REWARD
            enemy_reward = self._calculate_enemy_handling_reward(
                action,
                current_x_pos,
                current_y_pos,
                current_enemy_x,
                current_enemy_actions,
                self.prev_enemy_x,
                self.prev_enemy_actions,
                current_stage,
            )
            shaped_reward += enemy_reward
            if done and "lives" in info and info["lives"] == 0:
                shaped_reward += self.DEATH_PENALTY
            shaped_reward += self.TIME_PRESSURE_PENALTY

            self.prev_score = current_score
            self.prev_hp = current_hp
            self.prev_x_pos = current_x_pos
            self.prev_boss_hp = current_boss_hp
            self.prev_stage = current_stage
            self.prev_enemy_x = current_enemy_x.copy()
            self.prev_enemy_actions = current_enemy_actions.copy()

            info["shaped_reward"] = shaped_reward
            info["current_stage"] = current_stage
            info["player_hp"] = current_hp
            info["player_x"] = current_x_pos
            info["enemy_positions"] = current_enemy_x
            info["enemy_actions"] = current_enemy_actions
            info["stage_direction"] = self._get_stage_direction(current_stage)
        except Exception as e:
            print(f"Error in step reward calculation: {e}")
            shaped_reward = reward

        return obs, shaped_reward, terminated, truncated, info

    def _get_stage_direction(self, stage):
        if stage in [1, 3, 5]:
            return -1
        elif stage in [2, 4]:
            return 1
        return 0

    def _safe_subtract(self, a, b):
        a_int, b_int = int(a), int(b)
        if abs(a_int - b_int) < 128:
            return a_int - b_int
        if a_int == 0 and b_int > 200:
            return -b_int
        if a_int < 50 and b_int > 200:
            return (a_int + 256) - b_int
        return a_int - b_int

    def _calculate_direction_reward(self, stage, current_x, prev_x):
        movement = self._safe_subtract(current_x, prev_x)
        if movement == 0:
            return 0
        stage_direction = self._get_stage_direction(stage)
        if (stage_direction < 0 and movement < 0) or (
            stage_direction > 0 and movement > 0
        ):
            return abs(movement) * self.PROGRESS_REWARD_SCALE
        return -float(abs(movement)) * self.PROGRESS_REWARD_SCALE * 0.5

    def _calculate_enemy_handling_reward(
        self,
        action,
        player_x,
        player_y,
        current_enemy_x,
        current_enemy_actions,
        prev_enemy_x,
        prev_enemy_actions,
        current_stage,
    ):
        reward = 0.0
        stage_direction = self._get_stage_direction(current_stage)
        is_attack_action = action in [1, 8, 9, 11, 12]
        if (stage_direction < 0 and action == 6) or (
            stage_direction > 0 and action == 7
        ):
            reward += 0.2
        new_enemies_appeared = sum(
            1 for i in range(4) if prev_enemy_x[i] == 0 and current_enemy_x[i] != 0
        )
        if new_enemies_appeared > 0:
            reward -= new_enemies_appeared * 0.2
        for i in range(4):
            if current_enemy_x[i] == 0 and prev_enemy_x[i] == 0:
                continue
            if prev_enemy_x[i] != 0 and current_enemy_x[i] == 0:
                if self.prev_score < self._get_score(self.env.get_ram()):
                    reward += self.ENEMY_DEFEAT_BONUS
                continue
            if current_enemy_x[i] != 0:
                enemy_distance = abs(self._safe_subtract(player_x, current_enemy_x[i]))
                enemy_direction = 1 if current_enemy_x[i] > player_x else -1
                is_approaching = False
                if prev_enemy_x[i] != 0:
                    prev_dist = abs(self._safe_subtract(prev_enemy_x[i], player_x))
                    curr_dist = abs(self._safe_subtract(current_enemy_x[i], player_x))
                    if curr_dist < prev_dist:
                        is_approaching = True
                if enemy_distance < 30:
                    if is_attack_action:
                        reward += self.ENEMY_PROXIMITY_AWARENESS
                        if is_approaching:
                            reward += self.ENEMY_PROXIMITY_AWARENESS * 0.5
                    elif (enemy_direction < 0 and action == 6) or (
                        enemy_direction > 0 and action == 7
                    ):
                        reward += self.ENEMY_PROXIMITY_AWARENESS * 0.3
                elif enemy_distance < 60:
                    if (enemy_direction < 0 and action == 6) or (
                        enemy_direction > 0 and action == 7
                    ):
                        reward += 0.1
                else:
                    if (stage_direction < 0 and action == 6) or (
                        stage_direction > 0 and action == 7
                    ):
                        reward += 0.05
        return reward

    def _get_score(self, ram):
        score = 1
        has_score = False
        for addr in [0x0531, 0x0532, 0x0533, 0x0534, 0x0535]:
            if ram[addr] > 0:
                has_score = True
                score += ram[addr]
        if not has_score:
            return 1
        return score


def make_kungfu_env(num_envs=1, is_play_mode=False):
    def make_env(rank):
        def _init():
            try:
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFu-Nes", render_mode=render_mode)
            except Exception as e:
                print(f"Failed to load KungFu-Nes: {e}")
                render_mode = "human" if is_play_mode and rank == 0 else None
                env = retro.make(game="KungFuMaster-Nes", render_mode=render_mode)
            env = KungFuMasterEnv(env)
            os.makedirs("logs", exist_ok=True)
            env = Monitor(env, os.path.join("logs", f"kungfu_{rank}"))
            return env

        return _init

    if num_envs == 1:
        env = DummyVecEnv([make_env(0)])
    else:
        env = FixedSubprocVecEnv([make_env(i) for i in range(num_envs)])
    n_stack = 4
    env = VecFrameStack(env, n_stack=n_stack)
    print(f"Environment created with frame stack of {n_stack}")
    print(f"Observation space shape: {env.observation_space.shape}")
    return env


def create_model(env, resume=False):
    policy_kwargs = dict(net_arch=[64, 64])
    if resume and os.path.exists(MODEL_PATH):
        print(f"Loading existing model from {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env)
    else:
        print("Creating new model")
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tensorboard_log=None,
        )
    return model


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps

    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc="Training progress")

    def _on_step(self):
        self.pbar.update(self.training_env.num_envs)
        return True

    def _on_training_end(self):
        self.pbar.close()
        self.pbar = None


class ImprovedKungFuModelCallback(BaseCallback):
    def __init__(
        self,
        check_freq=5000,
        model_dir="model",
        verbose=1,
        moving_avg_window=20,
        checkpoint_freq=100000,
        min_steps_between_saves=50000,
    ):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.model_dir = model_dir
        self.moving_avg_window = moving_avg_window
        self.checkpoint_freq = checkpoint_freq
        self.min_steps_between_saves = min_steps_between_saves
        os.makedirs(os.path.join(self.model_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.model_dir, "best_models"), exist_ok=True)
        self.best_reward = -np.inf
        self.best_progress = -np.inf
        self.best_weighted_score = -np.inf
        self.best_moving_avg_reward = -np.inf
        self.last_save_step = 0
        self.episode_rewards = []
        self.episode_stages = []
        self.episode_metrics = []
        self.current_ep_reward = 0
        self.current_ep_max_stage = 1
        self.current_ep_x_progress_by_stage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        self.ep_start_step = 0
        self.action_counts = {i: 0 for i in range(len(KUNGFU_ACTIONS))}
        self.total_actions = 0
        self.last_log_step = 0
        self.action_log_freq = 5000

    def _on_training_start(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"Training started at {self.timestamp}")
        print(f"Model directory: {self.model_dir}")
        print(f"Moving average window: {self.moving_avg_window} episodes")
        print(f"Checkpoint frequency: {self.checkpoint_freq} steps")
        self.ep_start_step = self.n_calls

    def _on_step(self):
        for env_idx in range(self.training_env.num_envs):
            try:
                action = self.locals["actions"][env_idx]
                self.action_counts[action] += 1
                self.total_actions += 1
            except (KeyError, IndexError):
                pass

        if self.n_calls - self.last_log_step >= self.action_log_freq:
            self._log_action_percentages()
            self.last_log_step = self.n_calls

        for env_idx in range(self.training_env.num_envs):
            try:
                info = self.locals["infos"][env_idx]
                if isinstance(info, dict):
                    if "shaped_reward" in info:
                        self.current_ep_reward += info["shaped_reward"]
                    if "current_stage" in info:
                        stage = info["current_stage"]
                        self.current_ep_max_stage = max(
                            self.current_ep_max_stage, stage
                        )
                        if "player_x" in info:
                            x_pos = info["player_x"]
                            if stage in [1, 3, 5]:
                                progress = 255 - x_pos
                            else:
                                progress = x_pos
                            self.current_ep_x_progress_by_stage[stage] = max(
                                self.current_ep_x_progress_by_stage.get(stage, 0),
                                progress,
                            )
                else:
                    print(f"Warning: info is {type(info)}: {info}")
                done = self.locals["dones"][env_idx]
                if done:
                    ep_metrics = self._calculate_episode_metrics()
                    self.episode_rewards.append(self.current_ep_reward)
                    self.episode_stages.append(self.current_ep_max_stage)
                    self.episode_metrics.append(ep_metrics)
                    if len(self.episode_rewards) > self.moving_avg_window:
                        self.episode_rewards.pop(0)
                        self.episode_stages.pop(0)
                        self.episode_metrics.pop(0)
                    if self.verbose > 0:
                        steps = self.n_calls - self.ep_start_step
                        print(f"\nEpisode finished after {steps} steps:")
                        print(f"  Reward: {self.current_ep_reward:.2f}")
                        print(f"  Max Stage: {self.current_ep_max_stage}")
                        print(f"  Weighted Score: {ep_metrics['weighted_score']:.4f}")
                    self.current_ep_reward = 0
                    self.current_ep_max_stage = 1
                    self.current_ep_x_progress_by_stage = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                    self.ep_start_step = self.n_calls
            except (KeyError, IndexError):
                pass

        if self.n_calls % self.checkpoint_freq == 0 and self.n_calls > 0:
            checkpoint_path = os.path.join(
                self.model_dir, "checkpoints", f"checkpoint_{self.n_calls:010d}.zip"
            )
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Saved periodic checkpoint to {checkpoint_path}")

        if self.n_calls % self.check_freq == 0 and len(self.episode_metrics) > 0:
            moving_avg_reward = np.mean([m["reward"] for m in self.episode_metrics])
            moving_avg_stage = np.mean([m["max_stage"] for m in self.episode_metrics])
            moving_avg_score = np.mean(
                [m["weighted_score"] for m in self.episode_metrics]
            )
            if self.verbose > 0:
                print("\n--- MOVING AVERAGES ---")
                print(f"Window size: {len(self.episode_metrics)} episodes")
                print(f"Avg Reward: {moving_avg_reward:.2f}")
                print(f"Avg Max Stage: {moving_avg_stage:.2f}")
                print(f"Avg Weighted Score: {moving_avg_score:.4f}")
                print("----------------------")
            steps_since_last_save = self.n_calls - self.last_save_step
            should_save = False
            save_reason = ""
            if (
                moving_avg_score > self.best_moving_avg_reward
                and steps_since_last_save >= self.min_steps_between_saves
            ):
                should_save = True
                save_reason = "improved moving average"
                self.best_moving_avg_reward = moving_avg_score
            if len(self.episode_metrics) >= 10:
                recent_best = max(
                    [m["weighted_score"] for m in self.episode_metrics[-10:]]
                )
                if (
                    recent_best > self.best_weighted_score * 1.2
                    and steps_since_last_save >= self.min_steps_between_saves
                ):
                    should_save = True
                    save_reason = "exceptional recent episode"
                    self.best_weighted_score = recent_best
            if should_save:
                self.last_save_step = self.n_calls
                best_model_path = os.path.join(
                    self.model_dir,
                    "best_models",
                    f"best_model_{self.timestamp}_step{self.n_calls}_score{moving_avg_score:.4f}.zip",
                )
                standard_best_path = os.path.join(self.model_dir, "best_model.zip")
                self.model.save(best_model_path)
                self.model.save(standard_best_path)
                if self.verbose > 0:
                    print(f"\nSaving new best model: {save_reason}")
                    print(f"Saved to {best_model_path}")
                    print(f"Also saved to {standard_best_path}")
                self._log_action_percentages()
        return True

    def _calculate_episode_metrics(self):
        x_progress_scores = []
        for stage in range(1, self.current_ep_max_stage + 1):
            if stage in self.current_ep_x_progress_by_stage:
                if stage in [1, 3, 5]:
                    norm_progress = self.current_ep_x_progress_by_stage[stage] / 255
                else:
                    norm_progress = self.current_ep_x_progress_by_stage[stage] / 255
                x_progress_scores.append(norm_progress)
        avg_x_progress = (
            sum(x_progress_scores) / len(x_progress_scores) if x_progress_scores else 0
        )
        weighted_score = (
            0.6 * self.current_ep_max_stage
            + 0.3 * avg_x_progress
            + 0.1 * min(1.0, self.current_ep_reward / 1000)
        )
        return {
            "reward": self.current_ep_reward,
            "max_stage": self.current_ep_max_stage,
            "avg_x_progress": avg_x_progress,
            "weighted_score": weighted_score,
        }

    def _log_action_percentages(self):
        if self.total_actions == 0:
            return
        print("\n--- ACTION DISTRIBUTION ---")
        print(f"Total actions: {self.total_actions}")
        percentages = []
        for action_idx, count in self.action_counts.items():
            percentage = (count / self.total_actions) * 100
            action_name = KUNGFU_ACTION_NAMES[action_idx]
            percentages.append((action_name, percentage))
        percentages.sort(key=lambda x: x[1], reverse=True)
        for action_name, percentage in percentages:
            print(f"{action_name:<15}: {percentage:.2f}%")
        print("---------------------------\n")

    def _on_training_end(self):
        final_model_path = os.path.join(self.model_dir, "final_model.zip")
        self.model.save(final_model_path)
        print("\n=== TRAINING COMPLETE ===")
        print(f"Total steps: {self.n_calls}")
        print(f"Best moving average score: {self.best_moving_avg_reward:.4f}")
        print(f"Final model saved to: {final_model_path}")
        if len(self.episode_metrics) > 0:
            summary_path = os.path.join(self.model_dir, "training_summary.txt")
            with open(summary_path, "w") as f:
                f.write(
                    f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Total steps: {self.n_calls}\n")
                f.write(
                    f"Final moving average (over {len(self.episode_metrics)} episodes):\n"
                )
                f.write(
                    f"  Reward: {np.mean([m['reward'] for m in self.episode_metrics]):.2f}\n"
                )
                f.write(
                    f"  Max Stage: {np.mean([m['max_stage'] for m in self.episode_metrics]):.2f}\n"
                )
                f.write(
                    f"  Weighted Score: {np.mean([m['weighted_score'] for m in self.episode_metrics]):.4f}\n"
                )
                f.write("\nFinal Action Distribution:\n")
                percentages = [
                    (KUNGFU_ACTION_NAMES[i], (count / self.total_actions) * 100)
                    for i, count in self.action_counts.items()
                ]
                percentages.sort(key=lambda x: x[1], reverse=True)
                for action_name, percentage in percentages:
                    f.write(f"  {action_name:<15}: {percentage:.2f}%\n")
            print(f"Training summary saved to: {summary_path}")


def train_model(model, timesteps):
    print(f"Training for {timesteps} timesteps...")
    model_dir = os.path.dirname(MODEL_PATH)
    os.makedirs(model_dir, exist_ok=True)
    improved_callback = ImprovedKungFuModelCallback(
        check_freq=5000,
        model_dir=model_dir,
        verbose=1,
        moving_avg_window=20,
        checkpoint_freq=100000,
        min_steps_between_saves=50000,
    )
    progress_callback = ProgressBarCallback(timesteps)
    model.learn(
        total_timesteps=timesteps, callback=[progress_callback, improved_callback]
    )
    best_model_path = os.path.join(model_dir, "best_model.zip")
    if os.path.exists(best_model_path):
        print(f"Using best model as final model: {best_model_path}")
        print(f"Final model is at: {MODEL_PATH}")
        if best_model_path != MODEL_PATH:
            import shutil

            shutil.copy(best_model_path, MODEL_PATH)
    else:
        model.save(MODEL_PATH)
        print(f"Final model saved to {MODEL_PATH}")


def play_game(env, model, episodes=5):
    obs = env.reset()
    action_counts = {i: 0 for i in range(len(KUNGFU_ACTIONS))}
    total_actions = 0
    start_screen_counter = 0

    for episode in range(episodes):
        done = [False]
        total_reward = 0
        step = 0
        current_stage = 1
        max_stage = 1
        stage_progress = {}
        steps_since_attack = 0
        force_directional_steps = 0
        force_attack_steps = 0
        pbar = tqdm(desc=f"Episode {episode+1}", leave=True)

        while not any(done):
            action, _ = model.predict(obs, deterministic=True)
            original_action = action[0]
            if action[0] == 3:
                action[0] = 0
                print("Prevented agent from pressing START button")

            info = env.get_attr("unwrapped")[0]
            in_start_screen = False
            ram = None
            x_pos = 128
            y_pos = 0

            if hasattr(info, "get_ram"):
                try:
                    ram = info.get_ram()
                    current_stage = int(ram[0x0058])
                    x_pos = int(ram[0x0094])
                    y_pos = int(ram[0x00B6])
                    if current_stage == 0 or (current_stage == 1 and ram[0x04A6] == 0):
                        in_start_screen = True
                        start_screen_counter += 1
                    else:
                        in_start_screen = False
                        start_screen_counter = 0
                except Exception as e:
                    print(f"Error reading RAM: {e}")

            if in_start_screen:
                action[0] = 0
                if start_screen_counter % 30 == 0:
                    print("In start screen - waiting for environment to press START")
            else:
                steps_since_attack += 1
                stage_direction = -1 if current_stage in [1, 3, 5] else 1
                if steps_since_attack > 15 and force_attack_steps == 0:
                    attack_actions = [1, 8, 9, 11, 12]
                    import random

                    action[0] = random.choice(attack_actions)
                    steps_since_attack = 0
                    force_attack_steps = 5
                    print(f"Forcing attack action: {KUNGFU_ACTION_NAMES[action[0]]}")
                elif force_attack_steps > 0:
                    attack_actions = [1, 8, 9, 11, 12]
                    import random

                    action[0] = random.choice(attack_actions)
                    force_attack_steps -= 1
                elif force_directional_steps > 0:
                    if stage_direction < 0:
                        action[0] = 6
                    else:
                        action[0] = 7
                    force_directional_steps -= 1
                elif current_stage > 0:
                    if step % 10 == 0:
                        force_directional_steps = 3
                        if stage_direction < 0:
                            action[0] = 6
                            print(f"Forcing LEFT movement for stage {current_stage}")
                        else:
                            action[0] = 7
                            print(f"Forcing RIGHT movement for stage {current_stage}")
                if ram is not None:
                    if x_pos < 50 and stage_direction > 0:
                        action[0] = 7
                    elif x_pos > 200 and stage_direction < 0:
                        action[0] = 6
                if step % 5 == 0 and action[0] in [0, 4, 5]:
                    useful_actions = [1, 8, 9]
                    if stage_direction < 0:
                        useful_actions.append(6)
                    else:
                        useful_actions.append(7)
                    import random

                    action[0] = random.choice(useful_actions)
                    print(
                        f"Overriding passive action with {KUNGFU_ACTION_NAMES[action[0]]}"
                    )
                if action[0] in [1, 8, 9, 11, 12]:
                    steps_since_attack = 0
            if action[0] == 3:
                action[0] = 0
                print("Prevented agent from pressing START button")
            action_counts[action[0]] += 1
            total_actions += 1
            action_name = KUNGFU_ACTION_NAMES[action[0]]
            obs, reward, done, info = env.step(action)
            pbar.set_description(
                f"Episode {episode+1} | Step: {step} | Action: {action_name} | Reward: {reward[0]:.2f} | Stage: {current_stage}"
            )
            pbar.update(1)
            total_reward += reward[0]
            step += 1

            if any(done):
                progress_scores = []
                for stage, prog in stage_progress.items():
                    normalized = prog / 255.0
                    progress_scores.append(normalized)
                avg_progress = (
                    sum(progress_scores) / len(progress_scores)
                    if progress_scores
                    else 0
                )
                print(f"\nEpisode {episode+1} finished:")
                print(f"  Total reward: {total_reward:.2f}")
                print(f"  Max stage reached: {max_stage}")
                print(f"  Average stage progress: {avg_progress:.2f}")
                print(f"  Total steps: {step}")
                print("  Progress by stage:")
                for stage in sorted(stage_progress.keys()):
                    norm_prog = stage_progress[stage] / 255.0
                    print(f"    Stage {stage}: {norm_prog:.2f}")
                pbar.close()
                obs = env.reset()
                break

    if total_actions > 0:
        print("\n--- ACTION DISTRIBUTION DURING PLAY ---")
        print(f"Total actions: {total_actions}")
        percentages = []
        for action_idx, count in action_counts.items():
            percentage = (count / total_actions) * 100
            action_name = KUNGFU_ACTION_NAMES[action_idx]
            percentages.append((action_name, percentage))
        percentages.sort(key=lambda x: x[1], reverse=True)
        for action_name, percentage in percentages:
            print(f"{action_name:<15}: {percentage:.2f}%")
        print("----------------------------------")
        action_log_path = os.path.join(
            os.path.dirname(MODEL_PATH), "action_distribution_play.txt"
        )
        with open(action_log_path, "w") as f:
            f.write(f"Action distribution during play ({episodes} episodes):\n")
            f.write(f"Total actions: {total_actions}\n\n")
            for action_name, percentage in percentages:
                f.write(f"{action_name:<15}: {percentage:.2f}%\n")
        print(f"Action distribution saved to: {action_log_path}")


def evaluate_models(episodes=3):
    import glob
    import pandas as pd

    print(f"Evaluating all saved models ({episodes} episodes each)...")
    model_dir = os.path.dirname(MODEL_PATH)
    model_files = []
    best_models = glob.glob(os.path.join(model_dir, "best_models", "*.zip"))
    model_files.extend(best_models)
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoints", "*.zip"))
    model_files.extend(checkpoints)
    for std_name in ["best_model.zip", "final_model.zip", "kungfu.zip"]:
        std_path = os.path.join(model_dir, std_name)
        if os.path.exists(std_path) and std_path not in model_files:
            model_files.append(std_path)

    if not model_files:
        print("No saved models found to evaluate.")
        return

    print(f"Found {len(model_files)} models to evaluate.")
    env = make_kungfu_env(num_envs=1, is_play_mode=False)
    results = []

    for model_path in model_files:
        model_name = os.path.basename(model_path)
        print(f"\nEvaluating model: {model_name}")
        try:
            model = PPO.load(model_path)
            episode_rewards = []
            episode_stages = []
            episode_progress = []

            for episode in range(episodes):
                obs = env.reset()
                done = [False]
                total_reward = 0
                max_stage = 1
                stage_progress = {}

                while not any(done):
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, info = env.step(action)
                    try:
                        ram = env.get_attr("unwrapped")[0].get_ram()
                        current_stage = int(ram[0x0058])
                        max_stage = max(max_stage, current_stage)
                        x_pos = ram[0x0094]
                        if current_stage not in stage_progress:
                            stage_progress[current_stage] = 0
                        if current_stage in [1, 3, 5]:
                            progress = 255 - x_pos
                        else:
                            progress = x_pos
                        stage_progress[current_stage] = max(
                            stage_progress[current_stage], progress
                        )
                    except:
                        pass
                    total_reward += reward[0]

                progress_values = list(stage_progress.values())
                avg_progress = (
                    sum(progress_values) / len(progress_values)
                    if progress_values
                    else 0
                )
                norm_progress = avg_progress / 255.0
                episode_rewards.append(total_reward)
                episode_stages.append(max_stage)
                episode_progress.append(norm_progress)
                print(
                    f"  Episode {episode+1}: Reward={total_reward:.2f}, Max Stage={max_stage}, Progress={norm_progress:.2f}"
                )

            avg_reward = sum(episode_rewards) / len(episode_rewards)
            avg_stage = sum(episode_stages) / len(episode_stages)
            avg_progress = sum(episode_progress) / len(episode_progress)
            weighted_score = (
                0.6 * avg_stage + 0.3 * avg_progress + 0.1 * min(1.0, avg_reward / 1000)
            )
            results.append(
                {
                    "model": model_name,
                    "avg_reward": avg_reward,
                    "avg_stage": avg_stage,
                    "avg_progress": avg_progress,
                    "weighted_score": weighted_score,
                }
            )
            print(f"  Average over {episodes} episodes:")
            print(f"    Reward: {avg_reward:.2f}")
            print(f"    Max Stage: {avg_stage:.2f}")
            print(f"    Progress: {avg_progress:.2f}")
            print(f"    Weighted Score: {weighted_score:.4f}")
        except Exception as e:
            print(f"  Error evaluating model {model_name}: {e}")

    env.close()
    if results:
        df = pd.DataFrame(results)
        df = df.sort_values("weighted_score", ascending=False)
        print("\n=== MODEL EVALUATION RESULTS ===")
        print(df.to_string(index=False))
        results_path = os.path.join(model_dir, "model_evaluation_results.csv")
        df.to_csv(results_path, index=False)
        print(f"\nResults saved to: {results_path}")
        best_model = df.iloc[0]
        print(f"\nBest model: {best_model['model']}")
        print(f"  Weighted Score: {best_model['weighted_score']:.4f}")
        print(f"  Avg Stage: {best_model['avg_stage']:.2f}")
        print(f"  Avg Progress: {best_model['avg_progress']:.2f}")
        print(f"  Avg Reward: {best_model['avg_reward']:.2f}")
    else:
        print("No successful model evaluations to report.")


def main():
    parser = argparse.ArgumentParser(description="Train or play Kung Fu Master with AI")
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Number of timesteps to train"
    )
    parser.add_argument(
        "--num_envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--play", action="store_true", help="Play the game with trained agent"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate multiple saved models"
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=3,
        help="Number of episodes to play for each model during evaluation",
    )
    parser.add_argument(
        "--force_actions",
        action="store_true",
        help="Force action variety during play mode",
    )
    args = parser.parse_args()

    if args.eval:
        evaluate_models(args.eval_episodes)
        return

    env = make_kungfu_env(num_envs=args.num_envs, is_play_mode=args.play)
    model = create_model(env, resume=args.resume)

    if args.play:
        if not args.force_actions:
            print("\nTIP: You can use --force_actions to improve agent behavior.")
            print("     This will make the agent use more varied actions and")
            print("     follow correct stage directions.\n")
        play_game(env, model, episodes=5)
    else:
        train_model(model, args.timesteps)

    env.close()


if __name__ == "__main__":
    required_packages = ["tqdm", "pandas"]
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            import subprocess

            subprocess.check_call(["pip", "install", package])
            print(f"{package} installed successfully.")

    if torch.cuda.is_available():
        print("CUDA is available! Training will use GPU.")
    else:
        print("CUDA not available. Training will use CPU.")

    import gymnasium as gym

    print(f"Python version: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Gymnasium version: {gym.__version__}")
    print(f"Stable-Baselines3 version: {stable_baselines3.__version__}")
    print(f"Retro version: {retro.__version__}")

    main()
