import argparse
import os
import numpy as np
import torch
import zipfile
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    DummyVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import get_linear_fn
import logging
import sys
import signal
import time
import threading
import queue
from kungfu_env import KungFuWrapper, SimpleCNN, N_STACK, KUNGFU_ACTION_NAMES, make_env

current_model = None
global_logger = None
global_model_path = None
experience_data = []


def emergency_save_handler(signum, frame):
    global current_model, global_logger, global_model_path, experience_data
    if current_model is not None and global_model_path is not None:
        try:
            current_model.save(global_model_path)
            experience_count = len(experience_data)
            if global_logger:
                global_logger.info(
                    f"Emergency save triggered. Model saved at {global_model_path}"
                )
                global_logger.info(f"Collected experience: {experience_count} steps")
            else:
                print(f"Emergency save triggered. Model saved at {global_model_path}")
                print(f"Collected experience: {experience_count} steps")

            with open(f"{global_model_path}_experience_count.txt", "w") as f:
                f.write(f"Total experience collected: {experience_count} steps\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            if global_logger:
                global_logger.error(f"Emergency save failed: {e}")
            else:
                print(f"Emergency save failed: {e}")

        if hasattr(current_model, "env"):
            try:
                current_model.env.close()
            except Exception as e:
                if global_logger:
                    global_logger.warning(f"Failed to close environment: {e}")
                else:
                    print(f"Failed to close environment: {e}")
        if args.cuda:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del current_model
        current_model = None
    sys.exit(0)


signal.signal(signal.SIGINT, emergency_save_handler)


class ExperienceCollectionCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.experience_data = []

    def _on_step(self) -> bool:
        obs = self.locals.get("new_obs")
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if all(x is not None for x in [obs, actions, rewards, dones, infos]):
            experience = {
                "observation": obs,
                "action": actions,
                "reward": rewards,
                "done": dones,
                "info": infos,
            }
            self.experience_data.append(experience)
        return True


class SaveBestModelCallback(BaseCallback):
    def __init__(self, save_path, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.best_score = 0
        self.action_names = KUNGFU_ACTION_NAMES

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])

        total_hits = sum(info.get("enemy_hit", 0) for info in infos)
        total_hp = sum(info.get("hp", 0) for info in infos)
        total_dodge_reward = sum(info.get("dodge_reward", 0) for info in infos)
        total_survival_reward = sum(
            info.get("survival_reward_total", 0) for info in infos
        )
        avg_normalized_reward = sum(
            info.get("normalized_reward", 0) for info in infos
        ) / max(1, len(infos))
        total_progression_reward = sum(
            info.get("progression_reward", 0) for info in infos
        )
        max_stage = max(info.get("stage", 0) for info in infos)

        action_diversity = 0
        if infos and "action_percentages" in infos[0]:
            action_percentages = infos[0].get("action_percentages", [])
            if len(action_percentages) > 1:
                action_diversity = -sum(
                    p * np.log(p + 1e-6) for p in action_percentages if p > 0
                )
                action_diversity = action_diversity / np.log(len(action_percentages))

        score = (
            total_hits * 10
            + total_hp / 255.0 * 20
            + total_dodge_reward * 15
            + total_survival_reward * 12
            + avg_normalized_reward * 200
            + action_diversity * 25
            + total_progression_reward * 100
            + max_stage * 100
        )

        if score > self.best_score:
            self.best_score = score
            try:
                self.model.save(self.save_path)
                if self.verbose > 0:
                    print(
                        f"Saved best model with score {self.best_score:.2f} at step {self.num_timesteps}"
                    )
                    print(
                        f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Dodge: {total_dodge_reward:.2f}, "
                        f"Survival: {total_survival_reward:.2f}, Norm. Reward: {avg_normalized_reward:.2f}, "
                        f"Progression: {total_progression_reward:.2f}, Stage: {max_stage}, "
                        f"Action Diversity: {action_diversity:.2f}"
                    )

                    if (
                        infos
                        and "action_percentages" in infos[0]
                        and "action_names" in infos[0]
                    ):
                        action_percentages = infos[0].get("action_percentages", [])
                        action_names = infos[0].get("action_names", [])
                        if len(action_percentages) == len(action_names):
                            print("  Action Percentages:")
                            for name, perc in zip(action_names, action_percentages):
                                print(f"    {name}: {perc * 100:.1f}%")
            except Exception as e:
                print(f"Failed to save model: {e}")

        if self.num_timesteps % 5000 == 0 and self.verbose > 0:
            print(f"Step {self.num_timesteps} Progress:")
            print(f"  Current Score: {score:.2f}, Best Score: {self.best_score:.2f}")
            print(
                f"  Hits: {total_hits}, HP: {total_hp:.1f}/255, Norm. Reward: {avg_normalized_reward:.2f}, "
                f"Survival: {total_survival_reward:.2f}, Progression: {total_progression_reward:.2f}, Stage: {max_stage}"
            )

        return True


class RenderCallback(BaseCallback):
    def __init__(self, render_freq=256, fps=30, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.fps = fps
        self.step_count = 0
        self.render_queue = queue.Queue(maxsize=1)
        self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self.last_render_time = time.time()
        self.render_env = None
        self.running = True
        self.render_thread.start()

    def _render_loop(self):
        while self.running:
            try:
                if not self.render_queue.empty():
                    self.render_env = self.render_queue.get(block=False)
                    self.render_queue.task_done()

                if self.render_env:
                    start_time = time.time()
                    target_frame_time = 1.0 / self.fps
                    if start_time - self.last_render_time >= target_frame_time:
                        self.render_env.render()
                        self.last_render_time = start_time
                        render_time = time.time() - start_time
                        if global_logger:
                            global_logger.debug(
                                f"Render time: {render_time:.3f}s, Queue size: {self.render_queue.qsize()}"
                            )
                            if render_time > target_frame_time:
                                global_logger.debug(
                                    f"Render time exceeds frame target: {render_time:.3f}s vs {target_frame_time:.3f}s"
                                )
                    sleep_time = max(0, target_frame_time - (time.time() - start_time))
                    time.sleep(sleep_time)
                else:
                    time.sleep(0.01)
            except queue.Empty:
                time.sleep(0.01)
            except Exception as e:
                if global_logger:
                    global_logger.warning(f"Render thread error: {e}")
                time.sleep(0.01)

    def _on_step(self) -> bool:
        self.step_count += 1
        if self.step_count % self.render_freq == 0:
            try:
                if self.render_queue.qsize() == 0:
                    if (
                        hasattr(self.training_env, "envs")
                        and len(self.training_env.envs) > 0
                    ):
                        env_to_render = self.training_env.envs[0].unwrapped
                    else:
                        env_to_render = self.training_env.unwrapped
                    self.render_queue.put(env_to_render, block=False)
                    if global_logger:
                        global_logger.debug(f"Queued render at step {self.step_count}")
            except Exception as e:
                if global_logger:
                    global_logger.warning(f"Queue error: {e}")
        return True

    def _on_training_end(self) -> None:
        self.running = False
        if self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        if self.render_env:
            try:
                if hasattr(self.render_env, "close"):
                    self.render_env.close()
                if global_logger:
                    global_logger.info("Closed render environment")
            except Exception as e:
                if global_logger:
                    global_logger.warning(f"Error closing render env: {e}")


def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "training.log")),
        ],
    )
    return logging.getLogger()


def make_kungfu_env_for_vec():
    return make_env()


def train(args):
    global current_model, global_logger, global_model_path, experience_data
    experience_data = []

    global_logger = setup_logging(args.log_dir)
    global_logger.info(
        f"Starting training with {args.num_envs} envs and {args.timesteps} total timesteps"
    )
    global_model_path = args.model_path
    current_model = None

    if args.num_envs < 1:
        raise ValueError("Number of environments must be at least 1")

    global_logger.info("Training in live reinforcement learning mode")

    policy_kwargs = {
        "features_extractor_class": SimpleCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "net_arch": dict(pi=[128, 128], vf=[256, 256]),
        "activation_fn": torch.nn.ReLU,
    }

    learning_rate_schedule = get_linear_fn(start=2.5e-4, end=1e-5, end_fraction=0.5)
    ent_coef = 0.3
    params = {
        "learning_rate": learning_rate_schedule,
        "clip_range": args.clip_range,
        "ent_coef": ent_coef,
        "n_steps": 128,
        "batch_size": 32,
        "n_epochs": 5,
    }

    def initialize_model(env):
        if (
            args.resume
            and os.path.exists(args.model_path + ".zip")
            and zipfile.is_zipfile(args.model_path + ".zip")
        ):
            global_logger.info(f"Resuming training from {args.model_path}")
            try:
                custom_objects = {
                    "features_extractor_class": SimpleCNN,
                    "features_extractor_kwargs": {"features_dim": 256},
                }
                model = PPO.load(
                    args.model_path,
                    env=env,
                    custom_objects=custom_objects,
                    device="cuda" if args.cuda else "cpu",
                )
                expected_actions = len(KUNGFU_ACTION_NAMES)
                model_actions = model.policy.action_space.n
                if model_actions != expected_actions:
                    global_logger.warning(
                        f"Model action space ({model_actions}) does not match environment ({expected_actions})."
                    )
                global_logger.info("Successfully loaded existing model")
                return model
            except Exception as e:
                global_logger.warning(
                    f"Failed to load model: {e}. Starting new training session."
                )

        global_logger.info("Starting new training session")
        return PPO(
            "MultiInputPolicy",
            env,
            learning_rate=params["learning_rate"],
            clip_range=params["clip_range"],
            ent_coef=params["ent_coef"],
            n_steps=params["n_steps"],
            batch_size=params["batch_size"],
            n_epochs=params["n_epochs"],
            gamma=0.99,
            gae_lambda=0.95,
            verbose=1,
            policy_kwargs=policy_kwargs,
            device="cuda" if args.cuda else "cpu",
        )

    # Create environment(s)
    if args.render:
        global_logger.info(
            "Rendering enabled. Using a single environment for visualization."
        )
        args.num_envs = 1
        env = DummyVecEnv([make_kungfu_env_for_vec])
    else:
        env_fns = [make_kungfu_env_for_vec for _ in range(args.num_envs)]
        env = SubprocVecEnv(env_fns)

    # Apply vectorized wrappers
    env = VecFrameStack(env, n_stack=N_STACK)
    env = VecTransposeImage(env)

    # Print observation space for debugging
    dummy_obs = env.reset()
    if isinstance(dummy_obs, dict):
        for key, value in dummy_obs.items():
            if hasattr(value, "shape"):
                global_logger.info(f"Observation {key} shape: {value.shape}")
    else:
        global_logger.info(f"Observation shape: {dummy_obs.shape}")

    global_logger.info(f"Environment setup complete with {N_STACK} frame stacking")

    # Initialize or load model
    current_model = initialize_model(env)

    # Setup callbacks
    save_callback = SaveBestModelCallback(save_path=args.model_path)
    exp_callback = ExperienceCollectionCallback()
    callbacks = [save_callback, exp_callback]

    if args.render:
        render_callback = RenderCallback(render_freq=args.render_freq, fps=30)
        callbacks.append(render_callback)

    callback = CallbackList(callbacks)

    # Learn
    try:
        current_model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            progress_bar=args.progress_bar,
        )
        experience_data.extend(exp_callback.experience_data)
    except Exception as e:
        global_logger.error(f"Training failed: {e}")
        try:
            env.close()
        except Exception as close_e:
            global_logger.warning(f"Failed to close environment: {close_e}")
        raise

    # Save final model
    try:
        current_model.save(args.model_path)
        global_logger.info(f"Final model saved at {args.model_path}")
    except Exception as e:
        global_logger.error(f"Failed to save final model: {e}")

    # Cleanup
    try:
        env.close()
    except Exception as e:
        global_logger.warning(f"Failed to close environment during cleanup: {e}")

    experience_count = len(experience_data)
    experience_data = []
    global_logger.info(
        f"Training completed. Total experience collected: {experience_count} steps"
    )

    with open(f"{args.model_path}_experience_count.txt", "w") as f:
        f.write(f"Total experience collected: {experience_count} steps\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"Training parameters: num_envs={args.num_envs}, total_timesteps={args.timesteps}, mode=live\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a PPO model for Kung Fu with live RL"
    )
    parser.add_argument(
        "--model_path",
        default="models/kungfu_ppo/kungfu_ppo",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Total timesteps for training"
    )
    parser.add_argument(
        "--clip_range", type=float, default=0.2, help="Default clip range for PPO"
    )
    parser.add_argument(
        "--num_envs", type=int, default=4, help="Number of parallel environments"
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument(
        "--progress_bar", action="store_true", help="Show progress bar during training"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from the saved model"
    )
    parser.add_argument("--log_dir", default="logs", help="Directory for logs")
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during training"
    )
    parser.add_argument(
        "--render_freq", type=int, default=256, help="Render every N steps"
    )

    args = parser.parse_args()
    try:
        train(args)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if current_model and hasattr(current_model, "env"):
            try:
                current_model.env.close()
            except Exception as e:
                if global_logger:
                    global_logger.error(
                        f"Failed to close environment during final cleanup: {e}"
                    )
                else:
                    print(f"Failed to close environment during final cleanup: {e}")
        if args.cuda:
            torch.cuda.empty_cache()
