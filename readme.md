# Kung Fu Master AI - Reinforcement Learning Agent

A sophisticated deep reinforcement learning system that trains an AI agent to master the classic NES game Kung Fu using Proximal Policy Optimization (PPO) and custom neural network architectures.

## Technical Highlights

- **Advanced RL Implementation**: Custom PPO agent with multi-input actor-critic policy for handling both visual and non-visual state spaces
- **Custom CNN Architecture**: Purpose-built convolutional neural network for efficient visual feature extraction from game frames
- **Distributed Training**: Multi-environment parallelization using SubprocVecEnv for accelerated learning
- **GPU Acceleration**: CUDA-optimized training pipeline with efficient memory management
- **Experience Replay System**: Recording and playback capabilities for analyzing agent behavior and debugging
- **State Management**: Checkpoint/resume functionality with emergency save mechanisms for training stability
- **Real-time Visualization**: Live rendering during training with performance metrics tracking

## Core Technologies

- **Python** - Primary development language
- **PyTorch** - Deep learning framework
- **Stable-Baselines3** - Reinforcement learning algorithms
- **OpenAI Gym-Retro** - Classic game environment interface
- **CUDA** - GPU acceleration for neural network training
- **NumPy** - Numerical computing and data manipulation

## Architecture Overview

The system implements a sophisticated multi-component architecture:

### 1. Custom Environment Wrapper (`kungfu_env.py`)
- Observation space preprocessing and normalization
- Multi-enemy tracking system (up to 5 simultaneous enemies)
- Projectile detection and tracking
- Reward shaping for combat efficiency, survival, and strategic positioning
- Frame stacking for temporal awareness

### 2. Neural Network Architecture
- **Visual Processing**: Custom CNN for screen input
  - Convolutional layers for spatial feature extraction
  - Adaptive pooling for dimensionality reduction
- **Non-Visual Processing**: Dense layers for game state vectors
  - Enemy positions and health
  - Player stats and positioning
  - Projectile tracking
- **Multi-Input Fusion**: Combined visual and state-based decision making

### 3. Training Pipeline (`train.py`)
- Dynamic learning rate scheduling
- Multiple training modes: standard RL, imitation learning from recordings
- Experience collection with comprehensive metrics
- Best model selection based on composite performance score
- Robust checkpoint management and recovery

### 4. Utility Systems
- **Gameplay Recording** (`capture.py`): Capture human gameplay for imitation learning
- **Playback Engine** (`playback.py`): Visualize recorded episodes with variable playback speed
- **Interactive Play** (`play.py`): Test trained models in real-time

## Key Features

### Intelligent Reward Engineering
The agent learns through a carefully designed reward system:
- Combat effectiveness (enemy hits, strategic positioning)
- Survival optimization (HP preservation, dodge mechanics)
- Exploration encouragement (distance-based rewards)
- Action diversity for robust strategies

### Performance Optimization
- Multi-process vectorized environments for parallel experience collection
- CUDA memory management for stable long-running training
- Efficient frame stacking with VecTransposeImage
- Emergency save handlers for training interruption recovery

### Comprehensive Metrics
Real-time tracking of:
- Enemy hit rates and combat efficiency
- Health preservation and survival duration
- Action distribution and policy diversity
- Reward normalization and learning curves
- Distance-based tactical awareness

## Usage Examples

### Train Agent (GPU Accelerated)
```bash
python train.py --num_envs 4 --cuda --progress_bar --timesteps 10000
```

### Resume Training from Checkpoint
```bash
python train.py --num_envs 4 --cuda --progress_bar --timesteps 10000 --resume
```

### Train with Imitation Learning
```bash
python train.py --num_envs 4 --npz_dir recordings --cuda --timesteps 10000
```

### Watch Trained Agent Play
```bash
python train.py --render --resume
```

### Record Human Gameplay for Imitation Learning
```bash
python capture.py --state_file gameplay.state
```

### Playback Recorded Episodes
```bash
python playback.py recordings/episode.npz --speed 1.0
```

## Project Outcomes

This project demonstrates:
- **Deep RL Expertise**: Implementation of state-of-the-art PPO algorithm with custom modifications
- **Neural Network Design**: Creating specialized architectures for multi-modal input processing
- **System Engineering**: Building robust, production-ready ML training pipelines
- **Performance Optimization**: GPU acceleration and multi-process training
- **Research Methodology**: Systematic experimentation with reward functions and hyperparameters

## Technical Challenges Solved

1. **Multi-Modal State Representation**: Combining pixel observations with structured game state
2. **Sparse Reward Problem**: Designed dense reward signals for effective learning
3. **Action Space Complexity**: Handling discrete action combinations in real-time gameplay
4. **Training Stability**: Implemented checkpointing and emergency save mechanisms
5. **Sample Efficiency**: Parallel environment processing and experience replay

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure gym-retro ROM files are properly configured
```

## Requirements

- Python 3.8+
- PyTorch with CUDA support (for GPU training)
- 8GB+ RAM recommended
- NVIDIA GPU with CUDA (optional but recommended for faster training)

---

**Why This Project Matters**: This project showcases the ability to build complex AI systems from scratch, optimize for performance, and solve real-world challenges in reinforcement learning. The codebase demonstrates production-quality software engineering practices including error handling, logging, checkpointing, and modular design - skills directly transferable to building AI systems in industry.
