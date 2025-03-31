# Snake-Python-DQLN: Usage Guide

This document provides a comprehensive guide on how to use the Snake-Python-DQLN software across all phases of operation, from installation to advanced training and visualization.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Basic Usage](#basic-usage)
3. [Training a DQN Agent](#training-a-dqn-agent)
4. [Evaluating a Trained Agent](#evaluating-a-trained-agent)
5. [Playing the Game](#playing-the-game)
6. [Configuration Management](#configuration-management)
7. [Customization Options](#customization-options)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (optional but recommended for faster training)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/FedericoBaratti/Snake-Python-DQLN.git
   cd Snake-Python-DQLN
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python main.py --help
   ```

## Basic Usage

The software has three main execution modes:

1. **Training mode**: Train a new DQN agent or continue training an existing one
2. **Evaluation mode**: Test a trained agent's performance
3. **Play mode**: Play the Snake game yourself or watch an agent play

### Starting the Software

The main entry point is `main.py`:

```bash
# Display help information
python main.py --help

# Start training with default configuration
python main.py train

# Start the game to play manually
python main.py play

# Evaluate a trained agent
python main.py evaluate --model_path checkpoints/best_model.pth
```

## Training a DQN Agent

### Basic Training

To train an agent with default parameters:

```bash
python train.py
```

This will:
- Initialize a new DQN agent
- Train for the default number of episodes (1000)
- Save checkpoints to the `checkpoints/` directory

### Advanced Training Options

#### Configuring the Training Process

```bash
# Train with specific hyperparameters
python train.py --learning_rate 0.001 --epsilon_start 1.0 --epsilon_end 0.01 --epsilon_decay 0.995

# Train with a specific configuration file
python train.py --config config/custom_training_config.json

# Train with a specific random seed for reproducibility
python train.py --seed 42

# Continue training from a saved model
python train.py --load_model checkpoints/model_500.pth
```

#### Monitoring Training Progress

During training, the system will display:
- Episode number
- Current score
- Average score over recent episodes
- Epsilon value (exploration rate)
- Loss value from the most recent optimization step
- Training time per episode

### Training Parameters Explained

The most important training parameters include:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `learning_rate` | Learning rate for the optimizer | 0.0005 |
| `gamma` | Discount factor for future rewards | 0.99 |
| `epsilon_start` | Initial exploration rate | 1.0 |
| `epsilon_end` | Final exploration rate | 0.01 |
| `epsilon_decay` | Rate of exploration decay | 0.995 |
| `batch_size` | Number of transitions per optimization step | 64 |
| `target_update` | Frequency of target network updates | 10 |
| `buffer_size` | Capacity of the replay buffer | 100000 |
| `hidden_dim` | Dimensions of hidden layers in the neural network | [128, 128] |
| `num_episodes` | Total number of training episodes | 1000 |
| `prioritized_replay` | Whether to use prioritized experience replay | False |

## Evaluating a Trained Agent

To evaluate a trained agent's performance:

```bash
# Evaluate a trained model
python evaluate.py --model_path checkpoints/best_model.pth --num_episodes 100

# Evaluate with visualization
python evaluate.py --model_path checkpoints/best_model.pth --render true

# Evaluate with slower rendering for better visualization
python evaluate.py --model_path checkpoints/best_model.pth --render true --render_delay 100
```

### Evaluation Metrics

The evaluation will output:
- Mean score across all evaluation episodes
- Highest score achieved
- Average length of an episode (in steps)
- Success rate (percentage of episodes where the snake ate at least one food item)
- Visualization of score distribution

## Playing the Game

### Manual Play

To play the game yourself:

```bash
python play.py --mode human
```

Controls:
- **Arrow keys**: Change snake direction
- **P**: Pause/Resume
- **Q** or **Escape**: Quit the game

### Watching an Agent Play

To watch a trained agent play:

```bash
python play.py --mode ai --model_path checkpoints/best_model.pt --speed 10
```

Options:
- `--speed`: Controls the game speed (higher means faster)
- `--render_mode`: Visual style (`human`, `rgb_array`, or `console`)

## Configuration Management

### Configuration Files

The software uses JSON configuration files located in the `config/` directory:

```
config/
├── training_config.json    # Default training parameters
├── evaluation_config.json  # Default evaluation parameters
└── play_config.json        # Default game parameters
```

### Creating Custom Configurations

You can create custom configuration files by:

1. Copying and modifying an existing configuration file:
   ```bash
   cp config/training_config.json config/my_custom_config.json
   # Edit my_custom_config.json with your preferred settings
   ```

2. Using the software to generate a configuration file from command-line parameters:
   ```bash
   python main.py --generate_config my_config.json --learning_rate 0.001 --epsilon_decay 0.99
   ```

### Configuration Priority

The system loads configurations in the following order (later ones override earlier ones):
1. Default hardcoded values
2. Values from specified configuration file
3. Command-line arguments

## Customization Options

### Modifying the Reward System

The reward system can be customized through configuration:

```bash
python train.py --food_reward 1.0 --death_penalty -1.0 --move_closer_reward 0.1 --move_away_penalty -0.1 
```

### Changing the Neural Network Architecture

```bash
# Use a dueling DQN architecture
python train.py --model_type dueling

# Use a noisy DQN for parameter-space noise exploration
python train.py --model_type noisy

# Customize network dimensions
python train.py --hidden_dim 256,256,128
```

### Environment Customization

```bash
# Change the grid size
python train.py --grid_size 15

# Modify game speed during training visualization
python train.py --render --render_delay 0.05
```

## Performance Optimization

### Hardware Utilization

```bash
# Specify CPU usage
python train.py --device cpu

# Use CUDA GPU acceleration (Not tested because i have a RX7900XT)
python train.py --device cuda

# Use mixed precision training for faster computation (requires compatible GPU)
python train.py --mixed_precision true
```

### Memory Management

```bash
# Reduce memory usage by limiting buffer size
python train.py --buffer_size 50000

```

### Training Efficiency

```bash
# Increase batch size for better GPU utilization
python train.py --batch_size 128

# Use smaller network for faster training
python train.py --hidden_dim 64,64

```

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory Errors**:
   ```bash
   # Reduce batch size
   python train.py --batch_size 32
   
   # Decrease buffer size
   python train.py --buffer_size 50000
   
   # Use smaller network
   python train.py --hidden_dim "[64, 64]"
   ```

2. **Slow Training**:
   ```bash
   # Ensure you're using GPU if available
   python train.py --device cuda
   
   # Increase optimization frequency
    python train.py --target_update 5
   ```

3. **Agent Not Learning**:
   ```bash
   # Try different learning rate
   python train.py --lr 0.0001
   
   # Adjust exploration parameters
   python train.py --eps_decay 0.99 --model_type dueling
   
   # Try prioritized replay
   python train.py --buffer prioritized
   ```

4. **Game Crashes During Play**:
   ```bash
   # Run with debug mode to get detailed error information
   python play.py --debug true
   
   ```

### Logging and Debugging

Use the logging system to diagnose issues:

```bash
# Enable debug logging
python train.py --debug

# Save detailed logs to file
python train.py --log_file training_debug.log --log_level debug
```

For additional assistance, please file an issue on the GitHub repository with your error log and a description of your system configuration. 