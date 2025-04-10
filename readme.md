python train.py --num_envs 4 --cuda --progress_bar --timesteps 10000


python train.py --num_envs 4 --cuda --progress_bar --resume --timesteps 10000





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



===



        print("Controls:")
        print("R - Start recording segment")
        print("S - Stop recording segment")
        print("P - Save game state")
        print("L - Load last state")
        print("M - Toggle AI control")
        print("+/- - Increase/Decrease game speed")
        print("Q - Quit")
