

python script.py --timesteps 10000 --num_envs 4


python script.py --play





best model
python script.py --eval --eval_episodes 5












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