# Installation Guide

This guide provides detailed instructions for installing and configuring the Snake-Python-DQLN project on different platforms.

## System Requirements

### Hardware Requirements

- **CPU**: Multi-core processor (recommended)
- **RAM**: Minimum 4 GB, 8 GB or more recommended
- **GPU**: Optional, but recommended for faster training (CUDA compatible)
- **Disk space**: At least 500 MB free

### Required Software

- **Python**: Version 3.6 or higher
- **Operating system**: 
  - Windows 10/11
  - macOS 10.14 or higher
  - Linux (Ubuntu 18.04 or higher, or equivalent distributions)

## Step-by-Step Installation

### 1. Installing Python

#### Windows

1. Download Python 3.6+ from the [official website](https://www.python.org/downloads/windows/)
2. Launch the installer and select "Add Python to PATH" during installation
3. Complete the installation by following the instructions

#### macOS

1. Install Homebrew (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python
   ```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### 2. Cloning the Repository

```bash
# Clone the repository
git clone https://github.com/username/Snake-Python-DQLN.git

# Navigate to the project directory
cd Snake-Python-DQLN
```

### 3. Creating a Virtual Environment (Recommended)

It's recommended to use a virtual environment to isolate the project's dependencies.

#### Windows

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
venv\Scripts\activate
```

#### macOS and Linux

```bash
# Create the virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```

### 4. Installing Dependencies

With the virtual environment activated, install the required dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:
- numpy (>= 1.19.0): For mathematical operations and array handling
- torch (>= 1.7.0): For deep learning
- pygame (>= 2.0.0): For graphical visualization of the game
- matplotlib (>= 3.3.0): For results visualization
- tqdm (>= 4.50.0): For progress bars during training
- gymnasium (>= 0.26.0): For reinforcement learning interfaces

### 5. GPU Configuration (Optional)

To accelerate training with NVIDIA GPU:

#### Windows and Linux

1. Verify your GPU's compatibility with CUDA
2. Install the latest NVIDIA drivers
3. Install the appropriate version of PyTorch with CUDA support:

```bash
# For CUDA 11.6
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# For CUDA 11.7
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
```

#### macOS

On macOS, PyTorch uses the CPU since CUDA is not supported. If you have a Mac with M1/M2 chip, install the specific version:

```bash
pip install torch torchvision torchaudio
```

## Verifying the Installation

After installation, run the test to verify that everything is working correctly:

```bash
python main.py --test
```

You should see a confirmation message indicating that the installation was successful.

## Running the Project

### Starting the Game in Manual Mode

```bash
python play.py --mode human
```

### Training the Agent

```bash
python train.py
```

### Playing with a Trained Agent

```bash
python play.py --mode ai --model_path results/latest_trained_model/best_model.pt
```

## Advanced Configuration

### Configuration File

The behavior of the game and training can be customized by modifying the configuration file in `config/training_config.json`.

Common parameters you might want to modify:
- `grid_size`: Size of the game grid
- `eps_decay`: Exploration decay rate
- `total_episodes`: Total number of training episodes
- `learning_rate`: Agent's learning rate

### Command Line Arguments

Many parameters can be specified directly from the command line, overriding those in the configuration file. For example:

```bash
python train.py --grid_size 15 --eps_decay 5000 --total_episodes 10000
```

For a complete list of available arguments:

```bash
python train.py --help
```

## Troubleshooting

### Common Errors

1. **ImportError: No module named 'torch'**
   - Make sure you installed all dependencies with `pip install -r requirements.txt`
   - Verify that the virtual environment is activated

2. **CUDA Errors**
   - Verify that NVIDIA drivers are up to date
   - Verify that the installed PyTorch version is compatible with your CUDA version

3. **Pygame Errors**
   - On Linux, you might need to install additional dependencies:
     ```bash
     sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libportmidi-dev libswscale-dev libavformat-dev libavcodec-dev
     ```

### Contacting Support

If you encounter problems not listed here, you can:
1. Open an issue on GitHub
2. Contact the author via email

## Supported Platforms

The project has been tested and works correctly on the following platforms:

- Windows 10/11 with Python 3.8+
- Ubuntu 20.04 LTS with Python 3.8+
- macOS Big Sur with Python 3.9+

Training is significantly faster on systems with CUDA-compatible NVIDIA GPUs. 