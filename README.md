# Snake-Python-DQLN

<div align="center">
  <img src="images/snake.png" alt="Snake Game" width="400">
  <br>
  <h3>🐍 Advanced Deep Q-Learning Network for Snake Game 🧠</h3>
  <br>
  <p>
    <a href="#overview">Overview</a> •
    <a href="#features">Features</a> •
    <a href="#installation">Installation</a> •
    <a href="#usage">Usage</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#results">Results</a> •
    <a href="#changelog">Changelog</a>
  </p>
</div>

## Overview

**Snake-Python-DQLN** is a sophisticated implementation of the classic Snake game powered by state-of-the-art Deep Q-Learning techniques. This project demonstrates how modern reinforcement learning algorithms can master complex sequential decision-making tasks through a highly modular and extensible software architecture.

<div align="center">
  <table>
    <tr>
      <td align="center"><b>🧠 Deep Learning</b></td>
      <td align="center"><b>🎮 Classic Gameplay</b></td>
      <td align="center"><b>📊 Advanced Metrics</b></td>
    </tr>
    <tr>
      <td align="center">Cutting-edge neural networks</td>
      <td align="center">Nostalgic yet challenging</td>
      <td align="center">Comprehensive performance tracking</td>
    </tr>
  </table>
</div>

## Features

### 🚀 Advanced DQN Variants

- **Double DQN** - Eliminates value overestimation bias
- **Dueling DQN** - Separates state value and action advantage estimation
- **Noisy Networks** - Implements parameterized exploration strategies
- **Prioritized Experience Replay** - Focuses learning on informative transitions
- **N-step Returns** - Accelerates reward propagation through the network

### 🛠️ Technical Components

<div align="center">
  <table>
    <tr>
      <td align="center" width="33%"><b>Game Engine</b></td>
      <td align="center" width="33%"><b>RL Framework</b></td>
      <td align="center" width="33%"><b>Visualization</b></td>
    </tr>
    <tr>
      <td>
        • Configurable grid size<br>
        • Customizable reward function<br>
        • Real-time collision detection
      </td>
      <td>
        • Tensor-based state representation<br>
        • GPU-accelerated training<br>
        • Checkpoint management system
      </td>
      <td>
        • Interactive gameplay<br>
        • Performance dashboards<br>
        • Learning progression graphs
      </td>
    </tr>
  </table>
</div>

## Installation

### Prerequisites

- Python 3.6+
- CUDA-compatible GPU (recommended)

### Quick Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Snake-Python-DQLN.git
cd Snake-Python-DQLN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

<details>
<summary><b>Detailed installation instructions</b></summary>

For GPU acceleration:
```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

For visualization tools:
```bash
pip install pygame matplotlib seaborn
```

See [docs/installation.md](docs/installation.md) for complete installation guide.
</details>

## Usage

### Training the Agent

```bash
python train.py --agent dqn --model dueling --buffer prioritized --double True --n_steps 3
```

<details>
<summary><b>Training parameters</b></summary>

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--agent` | Agent type (dqn) | dqn |
| `--model` | Network architecture (dqn, dueling, noisy, noisy_dueling) | dqn |
| `--buffer` | Experience buffer type (standard, prioritized) | standard |
| `--double` | Enable Double DQN | False |
| `--n_steps` | Number of steps for N-step returns | 1 |
| `--hidden_dim` | Hidden layer dimensions | "128,128" |
| `--lr` | Learning rate | 1e-4 |
| `--gamma` | Discount factor | 0.99 |
| `--total_episodes` | Total training episodes | 10000 |

Run `python train.py --help` for all options.
</details>

### Playing the Game

```bash
# Watch AI play
python play.py --mode ai --model_path results/best_model.pt --speed 15

# Play yourself
python play.py --mode human --grid_size 20
```

### Interactive Mode

```bash
# Launch interactive interface
python main.py
```

## Architecture

The project follows a modular architecture with clean separation of concerns:

<style>
        body {
            font-family: 'Source Code Pro', monospace;
            background-color: #121212;
            color: white;
            margin: 0;
            padding: 0;
            font-size: 16px;
        }
        h1 {
            text-align: center;
            margin-top: 40px;
            color: #58a6ff;
        }
        .container {
            display: flex;
            justify-content: center;
            margin: 40px 0;
        }
        .layer {
            display: flex;
            flex-direction: column;
            margin: 0 20px;
            border: 1px solid #444;
            padding: 20px;
            width: 250px;
            border-radius: 8px;
        }
        .layer-header {
            font-weight: bold;
            margin-bottom: 10px;
            text-transform: uppercase;
            text-align: center;
        }
        .item {
            margin-bottom: 10px;
            padding: 5px;
            background-color: #333;
            border-radius: 5px;
            text-align: center;
        }
        .item:hover {
            background-color: #444;
        }
        .arrow {
            text-align: center;
            font-size: 24px;
            margin: 20px 0;
            color: #888;
            display: flex;
            align-items: anchor-center;

        }
        .line {
            border-left: 2px solid #888;
            margin-left: 10px;
            height: 100%;
            position: absolute;
            left: 50%;
            top: 100px;
        }
    </style>
</head>
<body>

    <h1>Architecture Diagram</h1>

    <div class="container">
        <!-- Application Layer -->
        <div class="layer">
            <div class="layer-header">Application Layer</div>
            <div class="item">main.py</div>
            <div class="item">train.py</div>
            <div class="item">play.py</div>
        </div>

        <!-- Arrow between layers -->
        <div class="arrow">==></div>

        <!-- Coordination Layer -->
        <div class="layer">
            <div class="layer-header">Coordination Layer</div>
            <div class="item">agent.py</div>
            <div class="item">buffer.py</div>
            <div class="item">network.py</div>
        </div>

        <!-- Arrow between layers -->
        <div class="arrow">==></div>

        <!-- Domain Layer -->
        <div class="layer">
            <div class="layer-header">Domain Layer</div>
            <div class="item">snake_game.py</div>
            <div class="item">core/environment.py</div>
        </div>

        <!-- Arrow between layers -->
        <div class="arrow">==></div>

        <!-- Infrastructure Layer -->
        <div class="layer">
            <div class="layer-header">Infrastructure Layer</div>
            <div class="item">utils/config.py</div>
            <div class="item">utils/common.py</div>
            <div class="item">utils/hardware.py</div>
        </div>
    </div>

<details>
<summary><b>Project Structure</b></summary>

```
Snake-Python-DQLN/
├── agents/                  # RL algorithm implementations
│   ├── dqn/                 # Deep Q-Network implementation
│   │   ├── agent.py         # DQN agent logic
│   │   ├── buffer.py        # Experience memory buffer
│   │   ├── network.py       # Neural network definitions
│   │   └── __init__.py
│   └── __init__.py
├── core/                    # Application core
│   ├── environment.py       # Environmental wrapper for RL
│   ├── snake_game.py        # Core game implementation
│   └── __init__.py
├── utils/                   # Utility components
│   ├── common.py            # Common functions
│   ├── config.py            # Configuration management
│   ├── hardware_utils.py    # Hardware optimization
│   └── __init__.py
├── config/                  # Configurations
├── docs/                    # Documentation
├── results/                 # Experiment results
├── main.py                  # Entry point
├── train.py                 # Training pipeline
├── play.py                  # UI and visualization
└── requirements.txt         # Dependencies
```
</details>

For comprehensive architecture details, see [docs/architecture.md](docs/architecture.md).



### Performance Metrics

| Model | Avg. Score | Max Score | Training Time |
|-------|------------|-----------|---------------|
| DQN   | 32.5       | 78        | 2.5 hours     |
| Double DQN | 45.3  | 112       | 3.2 hours     |
| Dueling DQN | 62.7 | 165       | 3.8 hours     |
| Full (All Features) | 87.2 | 243 | 4.5 hours    |

## Practical Applications

Beyond the classic Snake game, this architecture demonstrates techniques applicable to:

- **Robotics Navigation** - Path planning and obstacle avoidance
- **Resource Management** - Optimizing sequential allocation decisions
- **Autonomous Systems** - Self-improving decision making in complex environments

## Changelog

### Version 2.0.1 (Current)
- **Architecture Redesign**: Complete restructuring of project architecture with cleaner separation of concerns
- **Enhanced DQN Implementation**: 
  - Added Dueling DQN architecture
  - Implemented Noisy Networks for improved exploration strategies
  - Added support for N-step Returns
  - Improved Double DQN implementation
- **Performance Optimization**: 
  - GPU acceleration for neural network training
  - Hardware-optimized tensor operations
  - Improved checkpoint management system
- **Documentation Improvements**:
  - Comprehensive architecture documentation
  - Detailed installation guides
  - Extended usage documentation
- **UI Enhancements**:
  - New interactive gameplay mode
  - Performance dashboards for real-time monitoring
  - Learning progression visualization
- **Core Refactoring**:
  - Redesigned snake game core with improved collision detection
  - Configurable environment parameters
  - More sophisticated reward function options
- **Code Quality**:
  - Improved modularity and extensibility
  - Better separation of concerns
  - More comprehensive test coverage
  - English localization of all documentation and UI elements

### Version 2.0.1
- Initial introduction of visual model selection interface
- Dynamic checkpoint loading system
- Improved UI with advanced visual feedback
- Added quick-start parameter `--select-model`
- Updated documentation and user manual

### Version 1.4.2
- Initial beta release with basic DQN implementation
- Simple game environment with fixed grid size
- Basic visualization capabilities
- Limited training configuration options
- Eng documentation and UI

### Version 2.0
- Added visual model selection interface
- Implemented dynamic checkpoint loading system
- Improved user interface with advanced visual feedback
- Added quick start parameter --select-model
- Updated documentation and user manual

### Version 1.0
- Initial release with basic features
- Implementation of Snake game with manual controls
- DQN agent with autoplay mode support
- Training system with checkpoints
- Graphical interface in Pygame

## Contributors

- Federico Baratti

## License

Released under the MIT License.


