


python script.py --train --num_envs 16 --cuda --progress_bar --resume --timesteps 10000




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

