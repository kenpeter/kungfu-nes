python script.py --train --num_envs 4 --cuda --progress_bar --resume --timesteps 50000


python script.py --train --render --enable_file_logging --num_envs 1 --resume





python script.py --play --render






==



python script.py --capture --state_file knife_state.state --enable_file_logging




python capture_state.py --state_file knife_state.state --enable_file_logging






python script.py --train --render --num_envs 1 --timesteps 5000 --enable_file_logging




python script.py --train --render --num_envs 1 --timesteps 2000 --state_only --state_file knife_state.state --resume --enable_file_logging


python script.py --train --render --num_envs 1 --timesteps 5000 --resume --enable_file_logging


python train.py --play --render --enable_file_logging


python train.py --play --render --state_only --state_file knife_state.state --enable_file_logging
