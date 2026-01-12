# Kung Fu Master AI

Deep reinforcement learning agent that masters the NES game Kung Fu using PPO and custom neural networks.

## What I Built

- **Custom RL Agent**: PPO with multi-input policy (visual + game state)
- **Neural Architecture**: Custom CNN for vision + dense layers for game state
- **Distributed Training**: Multi-process parallelization with GPU acceleration
- **Training Tools**: Recording/playback system, checkpointing, metrics tracking

## Tech Stack

Python • PyTorch • Stable-Baselines3 • CUDA • OpenAI Gym-Retro

## Key Features

**Smart Reward Design** - Shaped rewards for combat, survival, and positioning
**GPU Optimized** - CUDA training with multi-environment parallelization
**Production Ready** - Error handling, logging, emergency saves, checkpoint recovery
**Imitation Learning** - Record human gameplay to bootstrap training

## Quick Start

```bash
# Train agent (GPU)
python train.py --num_envs 4 --cuda --timesteps 10000

# Resume training
python train.py --cuda --timesteps 10000 --resume

# Watch trained agent
python train.py --render --resume

# Record human gameplay
python capture.py --state_file gameplay.state
```

## Technical Highlights

**Multi-Modal Input Processing** - Combines CNN visual features with structured game state (enemy positions, HP, projectiles)

**Distributed Training** - Parallel environment processing for faster learning

**Reward Engineering** - Solved sparse reward problem with dense signals for combat effectiveness and survival

## Skills Demonstrated

Deep RL • Neural Network Design • GPU Optimization • System Engineering • Python

---

Built end-to-end: from environment design and reward shaping to distributed training and model deployment.
