# Humanoid Control with PPO and Pose-Informed Initialization

A reinforcement learning project that trains a MuJoCo Humanoid to walk using Proximal Policy Optimization (PPO), with the ability to initialize arm poses from real human images using MediaPipe.

## Key Features

* **PPO Reinforcement Learning** - Train a MuJoCo Humanoid agent to walk using state-of-the-art RL
* **Pose Initialization with MediaPipe** - Extract 2D skeleton from images → convert to joint angles → apply to humanoid arms
* **Stabilization Mechanism** - Forward shoulder force for smoother and more stable walking
* **Pre-trained Model Support** - Quick testing with pre-trained checkpoints
* **Complete Training Pipeline** - Full training and evaluation scripts with TensorBoard logging and video output

## Project Structure

```
PPO-Humanoid/
├── train_ppo.py          # Training script
├── test_ppo.py           # Testing script with pose initialization
├── requirements.txt      # Python dependencies
├── pose.jpg             # Sample pose image (add your own)
├── logs/                # TensorBoard logs
├── checkpoints/         # Model checkpoints
└── out.jpg             # Generated pose visualization
```

## Setup

### Prerequisites
- Python 3.12 or higher
- Virtual environment support

### Installation

1. **Clone or navigate to the project directory:**
```bash
cd PPO-Humanoid
```

2. **Create a virtual environment:**
```bash
python3.12 -m venv venv
```

3. **Activate the virtual environment:**

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Test with a Custom Pose

1. Place an image (e.g., `pose.jpg`) in the project folder
2. Run the test script:

```bash
python test_ppo.py
```

**What happens:**
- Extracts pose from the image using MediaPipe
- Generates `out.jpg` with visualized skeleton
- Applies arm pose to the humanoid
- Runs the PPO policy in MuJoCo simulation

### Train a New PPO Model

```bash
python train_ppo.py --n-epochs=1000
```

**Training outputs:**
- Training logs saved to `logs/`
- Model checkpoints saved to `checkpoints/`
- TensorBoard metrics for monitoring progress

**Monitor training with TensorBoard:**
```bash
tensorboard --logdir=logs/
```

## Pipeline Overview

### 1. Pose Estimation
1. Load input image
2. Detect human skeleton using MediaPipe
3. Convert 2D keypoints to humanoid joint angles
4. Apply only arm angles to maintain stability (legs use PPO learned behavior)

### 2. PPO Training
- **Architecture**: Actor-Critic neural network
- **Advantage Estimation**: Generalized Advantage Estimation (GAE)
- **Optimization**: PPO clipped objective function
- **Parallelization**: Multi-environment rollout for faster training
- **Logging**: Real-time metrics via TensorBoard

### 3. Testing & Execution
1. Apply custom pose to humanoid arms
2. Add small forward stabilization force to shoulders
3. Run trained PPO agent in MuJoCo environment
4. Render simulation in real-time

## Requirements

Key dependencies (see `requirements.txt` for full list):
- `mujoco` - Physics simulation
- `gymnasium` - RL environment interface
- `stable-baselines3` - PPO implementation
- `mediapipe` - Pose estimation
- `opencv-python` - Image processing
- `tensorboard` - Training visualization

## Tips

- **For faster training**: Increase the number of parallel environments
- **For better stability**: Adjust the stabilization force in `test_ppo.py`
- **For custom poses**: Use clear, well-lit images with visible arms
- **For evaluation**: Record videos by enabling rendering during test runs


