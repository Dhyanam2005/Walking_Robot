# Humanoid Control with PPO and Pose-Informed Initialization

## Key Features
- **PPO Reinforcement Learning** for training a MuJoCo Humanoid to walk.
- **Pose Initialization** using MediaPipe: extract 2D skeleton → convert to joint angles → apply to humanoid arms.
- **Stabilization Mechanism** adding forward shoulder force for smoother walking.
- **Pre-trained Model Support** for quick testing.
- **Complete Training & Evaluation Scripts** with TensorBoard logging and video output.

## Project Structure
```markdown

## Setup
```bash
cd PPO-Humanoid
python3.12 -m venv venv
# Activate:
# Windows: venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
pip install -r req.txt
How to Use
Test Using a Custom Pose

Place an image (e.g., pose.jpg) in the project folder.

Run:

python test_ppo.py


The script extracts pose → generates out.jpg → applies arm pose → runs PPO policy in MuJoCo.

Train a New PPO Model
python train_ppo.py --n-epochs=1000


Training logs appear in logs/

Model checkpoints saved in checkpoints/

Pipeline Summary
Pose Estimation

Load image

Detect human skeleton (MediaPipe)

Convert keypoints to humanoid joint angles

Apply only arm angles to maintain stability

PPO Training

Actor–Critic architecture

GAE-based advantage estimation

PPO clipped objective

Multi-environment rollout + TensorBoard logging

Testing Execution

Apply custom pose to humanoid

Add small forward stabilization force

Run PPO agent in MuJoCo and render the simulation