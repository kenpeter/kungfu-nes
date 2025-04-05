


python script.py --train --num_envs 16 --cuda --progress_bar --resume --skip_optuna false --timesteps 10000


python script.py --train --render --enable_file_logging --num_envs 1 --resume





python script.py --play --render



tensorboard --logdir=logs





==



python -c "import gym; print(gym.__version__)"




python capture_state.py --state_file knife_state.state --enable_file_logging











python script.py --train --render --num_envs 1 --timesteps 5000 --enable_file_logging




python script.py --train --render --num_envs 1 --timesteps 2000 --state_only --state_file knife_state.state --resume --enable_file_logging


python script.py --train --render --num_envs 1 --timesteps 5000 --resume --enable_file_logging


python train.py --play --render --enable_file_logging


python train.py --play --render --state_only --state_file knife_state.state --enable_file_logging



===


[I 2025-04-05 09:59:42,518] Trial 11 finished with value: 92.30204 and parameters: {'learning_rate': 0.0008584314714364342, 'n_steps': 16384, 'batch_size': 256, 'n_epochs': 4, 'gamma': 0.90061830637976, 'clip_range': 0.10005299463300507}. Best is trial 10 with value: 95.0305.
57344it [03:18, 288.75it/s]
[I 2025-04-05 10:03:01,290] Trial 12 finished with value: 78.97787 and parameters: {'learning_rate': 0.0008987615494759457, 'n_steps': 14336, 'batch_size': 256, 'n_epochs': 3, 'gamma': 0.9007832232799664, 'clip_range': 0.10511801243525405}. Best is trial 10 with value: 95.0305.
C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\ppo\ppo.py:149: UserWarning: You have specified a mini-batch size of 192, but because the `RolloutBuffer` is of size `n_steps * n_envs = 16384`, after every 85 untruncated mini-batches, there will be a truncated mini-batch of size 64
We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.
Info: (n_steps=16384 and n_envs=1)
  f"You have specified a mini-batch size of {batch_size},"
65536it [04:25, 246.62it/s]
[I 2025-04-05 10:07:27,198] Trial 13 finished with value: 88.51826 and parameters: {'learning_rate': 0.0005998282996820877, 'n_steps': 16384, 'batch_size': 192, 'n_epochs': 7, 'gamma': 0.9154599365261985, 'clip_range': 0.2588107938041782}. Best is trial 10 with value: 95.0305.
Best hyperparameters: {'learning_rate': 0.0007181399768445935, 'n_steps': 16384, 'batch_size': 256, 'n_epochs': 4, 'gamma': 0.9051620100876205, 'clip_range': 0.1003777999324603}
Best value: 95.0305
  0%|                                                                                                                       | 0/10000 [00:00<?, ?it/s]Traceback (most recent call last):
  File "script.py", line 440, in <module>
    train(args)
  File "script.py", line 404, in train
    reset_num_timesteps=not args.resume
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\ppo\ppo.py", line 314, in learn
    progress_bar=progress_bar,
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\common\on_policy_algorithm.py", line 250, in learn
    continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\common\on_policy_algorithm.py", line 184, in collect_rollouts
    if callback.on_step() is False:
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\common\callbacks.py", line 104, in on_step
    return self._on_step()
  File "script.py", line 238, in _on_step
    env = self.training_env.envs[0]
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\common\vec_env\base_vec_env.py", line 313, in __getattr__
    return self.getattr_recursive(name)
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\stable_baselines3\common\vec_env\base_vec_