

python script.py --timesteps 1000 --resume


python script.py --play

===

python train.py --num_envs 4 --cuda --progress_bar --timesteps 10000






python train.py --num_envs 4 --cuda --progress_bar --timesteps 10000 --resume




python train.py --num_envs 4 --npz_dir recordings_filtered --cuda --progress_bar --timesteps 10000 --resume





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



requirements for reward function:





game stage
==
0058	b	u	0	RAM	Current Stage
stage 1 agent should go left
stage 2 agent should go right
stage 3 agent go left
stage 4 go right
stage 5 go left



agent's score
==
0531	b	h	0	RAM	Score 5
0532	b	h	0	RAM	Score 4
0533	b	h	0	RAM	Score 3
0534	b	h	0	RAM	Score 2
0535	b	h	0	RAM	Score 1


agent's hp etc
04A6	b	u	0	RAM	Hero HP (drop rapid, need to react)
0094	b	u	0	RAM	Hero Screen Pos X (progress further, more reward)
00B6	b	u	0	RAM	Hero Pos Y

game clock is like 2 minutes for agent to finish the game (so there is urgency to finish the game quick)



boss
==
0090	b	h	0	RAM	Boss Pos X
004E	b	u	0	RAM	Boss Action
0093	b	u	0	RAM	Boss Pos X
04A5	b	u	0	RAM	Boss HP



enemy
==
008E	b	u	0	RAM	Enemy 1 Pos X
008F	b	u	0	RAM	Enemy 2 Pos X
0090	b	u	0	RAM	Enemy 3 Pos X
0091	b	u	0	RAM	Enemy 4 Pos X
0080	b	h	0	RAM	Enemy 1 Action
0081	b	h	0	RAM	Enemy 2 Action
0082	b	h	0	RAM	Enemy 3 Action
0083	b	h	0	RAM	Enemy 4 Action