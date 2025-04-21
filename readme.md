python train.py --num_envs 4 --cuda --progress_bar --timesteps 1000 --resume

python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 1000 --resume





nohup python train.py --num_envs 4 --npz_dir recordings --cuda --progress_bar --timesteps 400000 --resume & 


tail -f nohup.out  # Check output (if still writing, it's running)
ps aux | grep "long-task.sh"  # Check if process exists







python train.py  --cuda --progress_bar --timesteps 50 --resume --render










python playback.py recordings/KungFu-Nes_1Player.Level1_k.back.npz
python playback.py recordings/KungFu-Nes_1Player.Level1_k.punch.npz






python playback.py recordings/your_recording.npz --speed 0.5  # Half speed
python playback.py recordings/your_recording.npz --speed 2.0  # Double speed








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










(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        16Gi        11Gi        57Mi       2.4Gi        13Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        17Gi        10Gi        57Mi       2.4Gi        12Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        18Gi       9.8Gi        57Mi       2.4Gi        11Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        19Gi       8.9Gi        57Mi       2.4Gi        10Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        20Gi       8.1Gi        57Mi       2.4Gi        10Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        21Gi       7.2Gi        57Mi       2.4Gi       9.1Gi
Swap:          8.0Gi          0B       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        27Gi       279Mi        77Mi       3.6Gi       3.4Gi
Swap:          8.0Gi       256Ki       8.0Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        30Gi       241Mi        36Mi       753Mi       541Mi
Swap:          8.0Gi       4.4Gi       3.6Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi        30Gi       112Mi        36Mi       456Mi       119Mi
Swap:          8.0Gi       6.7Gi       1.3Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ sudo free -h
               total        used        free      shared  buff/cache   available
Mem:            30Gi       1.6Gi        29Gi        23Mi       365Mi        28Gi
Swap:          8.0Gi       2.1Gi       5.9Gi
(base) kenpeter@kenpeter-ubuntu:~/work/kungfu-nes$ 
