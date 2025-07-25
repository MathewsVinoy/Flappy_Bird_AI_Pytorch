# Flappy Bird AI

An AI-powered Flappy Bird game implementation using Deep Q-Learning (DQN) with PyTorch and Pygame. Watch as the AI learns to play and master the classic Flappy Bird game through reinforcement learning.

## 🎮 Overview

This project implements a Flappy Bird game where an AI agent learns to play using Deep Q-Network (DQN) reinforcement learning. The agent starts with no knowledge of the game and gradually improves its performance through trial and error, eventually achieving superhuman performance.

## 🚀 Features

- **AI Agent**: Deep Q-Learning implementation that learns to play Flappy Bird
- **Game Environment**: Full Flappy Bird game built with Pygame
- **Real-time Training**: Watch the AI learn and improve in real-time
- **Neural Network**: Custom PyTorch model for Q-value approximation
- **Training Visualization**: Track progress with score monitoring
- **Replay Memory**: Experience replay for stable learning

## 📁 Project Structure

```
flap_bird/
├── agent.py       # DQN Agent implementation
├── game.py        # Flappy Bird game environment
├── model.py       # Neural network model (PyTorch)
├── Qtrainer.py    # Q-learning trainer class
└── README.md      # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

Install the required packages using pip:

```bash
pip install pygame torch torchvision numpy matplotlib
```

Or install from requirements file:

```bash
pip install -r requirements.txt
```

### Requirements.txt

```
pygame>=2.1.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
```

## 🎯 Quick Start

1. **Clone or download the project files**

2. **Install dependencies**:

   ```bash
   pip install pygame torch numpy matplotlib
   ```

3. **Run the training**:

   ```bash
   python agent.py
   ```

4. **Watch the AI learn!** The game window will open and you'll see the AI bird learning to navigate through pipes.

## 📋 File Descriptions

### `agent.py`

Main training script containing:

- DQN Agent class with epsilon-greedy exploration
- Memory replay buffer
- Training loop and game interaction
- Model saving/loading functionality

### `game.py`

Flappy Bird game implementation:

- Game physics and mechanics
- Collision detection
- Rendering with Pygame
- State representation for AI

### `model.py`

Neural network architecture:

- Deep Q-Network (DQN) model
- PyTorch implementation
- Forward pass definition
- Model architecture configuration

### `Qtrainer.py`

Training utilities:

- Q-learning algorithm implementation
- Loss calculation and backpropagation
- Optimizer configuration
- Training step execution

## 🧠 How It Works

### Deep Q-Learning (DQN)

The AI uses Deep Q-Learning to learn optimal actions:

1. **State Representation**: The game state includes:

   - Bird's vertical position and velocity
   - Distance to next pipe
   - Height of pipe gaps

2. **Actions**: The agent can choose between:

   - Flap (jump)
   - Do nothing (fall)

3. **Reward System**:

   - +10 for passing through pipes
   - -10 for hitting pipes or ground
   - +1 for staying alive

4. **Learning Process**:
   - Exploration vs Exploitation (ε-greedy)
   - Experience replay for stability
   - Target network for stable learning

### Training Process

1. **Initial Phase**: Random actions (high exploration)
2. **Learning Phase**: Gradual shift to learned policy
3. **Mastery Phase**: Consistent high performance

## 📊 Monitoring Training

The agent will display:

- Current game score
- Training episode number
- Epsilon value (exploration rate)
- Average score over recent episodes

Watch as the scores gradually improve over hundreds/thousands of episodes!

## ⚙️ Configuration

Key parameters you can adjust in `agent.py`:

```python
LEARNING_RATE = 0.001    # Neural network learning rate
EPSILON_DECAY = 0.995    # Exploration decay rate
MEMORY_SIZE = 10000      # Replay buffer size
BATCH_SIZE = 32          # Training batch size
GAMMA = 0.9              # Discount factor
```

## 💾 Model Persistence

The trained model is automatically saved as:

- `model.pth` - Neural network weights
- Automatically loads existing model if present

## 🎮 Playing Manually

To play the game manually (without AI), modify the game loop in `game.py` to accept keyboard input instead of AI actions.

## 🔧 Troubleshooting

### Common Issues:

1. **"No module named 'pygame'"**

   ```bash
   pip install pygame
   ```

2. **"No module named 'torch'"**

   ```bash
   pip install torch
   ```

3. **Slow training**: Reduce the game's FPS or increase epsilon decay

4. **Poor performance**: Adjust neural network architecture or hyperparameters

## 📈 Expected Results

- **Episodes 0-100**: Random performance, scores typically 0-5
- **Episodes 100-500**: Gradual improvement, occasional good runs
- **Episodes 500+**: Consistent performance, scores of 20+ common
- **Well-trained agent**: Can achieve scores of 100+ consistently

## 🤝 Contributing

Feel free to fork this project and experiment with:

- Different neural network architectures
- Alternative RL algorithms (A3C, PPO, etc.)
- Enhanced game features
- Improved visualization

## 📝 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Inspired by the original Flappy Bird game
- Built using PyTorch for deep learning
- Pygame for game development
- Deep Q-Learning algorithm by DeepMind

---

**Happy Learning!** 🐦🤖

Watch your AI bird go from crash-prone beginner to Flappy Bird master!
