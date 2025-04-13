python train.py --num_envs 4 --cuda --progress_bar --timesteps 1000 --resume

python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 1000 --resume




python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 50 --resume --render





python play.py


python capture.py


tensorboard --logdir=tensorboard_logs





==



python -c "import gym; print(gym.__version__)"




python capture_state.py --state_file knife_state.state --enable_file_logging











python script.py --train --render --num_envs 1 --timesteps 5000 --enable_file_logging




python script.py --train --render --num_envs 1 --timesteps 2000 --state_only --state_file knife_state.state --resume --enable_file_logging


python script.py --train --render --num_envs 1 --timesteps 5000 --resume --enable_file_logging


python train.py --play --render --enable_file_logging


python train.py --play --render --state_only --state_file knife_state.state --enable_file_logging

==





===

python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 10000








figo2@kenpeter-pc-2 MINGW64 ~/work/gym-retro/kungfu-nes (main-simple)
$ python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 10000 --resume
2025-04-13 07:42:18,706 - INFO - Starting training with 4 envs and 10000 timesteps
2025-04-13 07:42:18,706 - INFO - Maximum number of enemies: 5
2025-04-13 07:42:18,706 - INFO - Training mode: mimic
2025-04-13 07:42:18,706 - INFO - Loading NPZ data from: recordings
2025-04-13 07:42:20,874 - INFO - Resuming training from models/kungfu_ppo/kungfu_ppo_best
Using cuda device
Traceback (most recent call last):
  File "train.py", line 383, in <module>
    train(args)
  File "train.py", line 334, in train
    model.policy.load_state_dict(old_model.policy.state_dict())
  File "C:\Users\figo2\anaconda3\envs\gym-retro\lib\site-packages\torch\nn\modules\module.py", line 1672, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for MultiInputActorCriticPolicy:
        size mismatch for features_extractor.non_visual.0.weight: copying a param with shape torch.Size([256, 100]) from checkpoint, the shape in current model is torch.Size([256, 144]).
        size mismatch for pi_features_extractor.non_visual.0.weight: copying a param with shape torch.Size([256, 100]) from checkpoint, the shape in current model is torch.Size([256, 144]).
        size mismatch for vf_features_extractor.non_visual.0.weight: copying a param with shape torch.Size([256, 100]) from checkpoint, the shape in current model is torch.Size([256, 144]).
        size mismatch for action_net.weight: copying a param with shape torch.Size([11, 128]) from checkpoint, the shape in current model is torch.Size([9, 128]).
        size mismatch for action_net.bias: copying a param with shape torch.Size([11]) from checkpoint, the shape in current model is torch.Size([9]).
(gym-retro)
figo2@kenpeter-pc-2 MINGW64 ~/work/gym-retro/kungfu-nes (main-simple)
$
