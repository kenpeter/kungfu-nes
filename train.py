import gym
import retro
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from gym import spaces

class KungFuDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # Simplified action space: 8 discrete actions
        self.action_space = spaces.Discrete(8)
        # Predefined button combinations for Kung Fu Master
        self._actions = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 0: NOOP
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: A (Punch)
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 2: B (Jump)
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 3: UP
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 4: DOWN
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # 5: LEFT
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 6: RIGHT
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 7: A+B
        ]

    def action(self, action):
        # Convert Discrete(8) action to a button array
        return self._actions[action]

def make_env():
    # Use retro.Actions.ALL to accept button arrays
    env = retro.make(
        game="KungFu-Nes",
        use_restricted_actions=retro.Actions.ALL
    )
    env = KungFuDiscreteWrapper(env)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    tensorboard_log="./logs/",
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./checkpoints/",
    name_prefix="kungfu_ppo",
)

try:
    model.learn(total_timesteps=1_000_000, callback=checkpoint_callback)
finally:
    model.save("kungfu_ppo_final")
    env.close()

print("Training complete!")