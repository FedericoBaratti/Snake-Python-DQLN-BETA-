# Source Code Overview

This section provides an overview of the Snake-Python-DQLN project source code, illustrating the main classes, functions, and modules, and explaining the general flow of the application.

## Code Organization

The code is organized in a modular structure that clearly separates different responsibilities:

```
Snake-Python-DQLN/
├── agents/                  # Agent implementations
│   ├── dqn/                 # Deep Q-Network agent
│   │   ├── agent.py         # DQN agent implementation
│   │   ├── buffer.py        # Memory buffer implementation
│   │   ├── network.py       # Neural network architectures
│   │   └── __init__.py
│   └── __init__.py
├── core/                    # Core components
│   ├── environment.py       # Learning environment wrapper
│   ├── snake_game.py        # Game logic implementation
│   └── __init__.py
├── utils/                   # Utilities
│   ├── common.py            # Common functions
│   ├── config.py            # Configuration management
│   ├── hardware_utils.py    # Hardware utilities (GPU/CPU)
│   └── __init__.py
├── config/                  # Configuration files
│   └── training_config.json # Default training configuration
├── main.py                  # Main entry point
├── train.py                 # Training script
├── play.py                  # Playing script
```

## Main Components

### 1. Core: Snake Game and Environment

#### `snake_game.py`

This file implements the basic logic of the Snake game through the `SnakeGame` class. Here are the main functionalities:

```python
class SnakeGame:
    """Implements the Snake game logic with advanced features."""
    
    def __init__(self, config=None):
        # Initialize the game with customizable configurations
        
    def reset(self):
        # Reset the game to an initial state
        
    def step(self, action):
        # Execute an action and update the game state
        # Returns: new state, reward, termination flag, info
        
    def get_state(self):
        # Generate the game state representation
        
    def place_food(self):
        # Place food on the grid
        
    def is_collision(self, position):
        # Check if a position causes a collision
```

The `SnakeGame` class also includes a customizable reward system (`RewardSystem`) that defines rewards for different game situations.

#### `environment.py`

This file implements a wrapper that adapts the Snake game to a standard reinforcement learning interface:

```python
class SnakeEnvironment:
    """Wrapper for the Snake game as a reinforcement learning environment."""
    
    def __init__(self, config=None):
        # Initialize the environment
        
    def reset(self):
        # Reset the environment and return the initial state
        
    def step(self, action):
        # Execute an action in the environment
        # Returns: new state, reward, termination flag, info
        
    def render(self, mode='human'):
        # Display the current state of the environment
        
    def close(self):
        # Close the environment and free resources
```

### 2. Agents: Reinforcement Learning Agents

#### `agents/dqn/agent.py`

Implements the DQN agent that learns to play Snake:

```python
class DQNAgent:
    """Agent using Deep Q-Learning with advanced features."""
    
    def __init__(self, state_dim, action_dim, config):
        # Initialize the agent and its components
        
    def select_action(self, state, training=True):
        # Select an action using ε-greedy policy
        
    def store_transition(self, state, action, next_state, reward, done):
        # Store a transition in the buffer
        
    def optimize_model(self):
        # Update the network weights based on experiences
        
    def train(self, num_updates=1):
        # Perform a training step
        
    def update_target_network(self):
        # Update the target network with the policy network weights
        
    def save(self, path, save_memory=False, save_optimizer=True):
        # Save the model to disk
        
    @classmethod
    def load(cls, path, device=None, load_memory=False):
        # Load a model from disk
```

#### `agents/dqn/buffer.py`

Implements memory buffers for experience replay:

```python
class ReplayBuffer:
    """Standard buffer for storing experiences."""
    
    def __init__(self, capacity, state_dim, device):
        # Initialize the buffer with fixed capacity
        
    def push(self, state, action, next_state, reward, done):
        # Add a transition to the buffer
        
    def sample(self, batch_size):
        # Sample a random batch of transitions
        
    def __len__(self):
        # Return the current size of the buffer

class PrioritizedReplayBuffer(ReplayBuffer):
    """Buffer with experience prioritization."""
    
    def __init__(self, capacity, state_dim, device, alpha=0.6, beta=0.4, beta_increment=0.001):
        # Initialize the buffer with prioritization
        
    def push(self, state, action, next_state, reward, done, error=None):
        # Add a transition with priority
        
    def sample(self, batch_size):
        # Sample a batch based on priorities
        
    def update_priorities(self, indices, errors):
        # Update priorities based on TD error
```

#### `agents/dqn/network.py`

Defines neural network architectures used by the agent:

```python
class DQNModel(nn.Module):
    """Base neural network for DQN."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=[128, 128], activation=F.relu):
        # Initialize the network with fully connected layers
        
    def forward(self, state):
        # Forward pass of the network

class DuelingDQNModel(nn.Module):
    """Dueling DQN neural network."""
    
    def __init__(self, state_dim, action_dim, hidden_dim=[128, 128], activation=F.relu):
        # Initialize the network with dueling architecture
        
    def forward(self, state):
        # Forward pass with separate streams for value and advantage

class NoisyLinear(nn.Module):
    """Linear layer with parameterized noise for exploration."""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        # Initialize the layer with noisy parameters
        
    def forward(self, x):
        # Forward pass with noise during training

# Other implementations: NoisyDQNModel, NoisyDuelingDQNModel
```

### 3. Utils: Utilities and Configuration

#### `utils/config.py`

Manages system configuration:

```python
class Config:
    """Class to manage system configuration."""
    
    def __init__(self, **kwargs):
        # Initialize with default or custom values
        
    @classmethod
    def from_json(cls, file_path):
        # Load configuration from a JSON file
        
    def save(self, file_path):
        # Save configuration to file
        
    def update(self, other_config):
        # Update configuration with another one
```

#### `utils/hardware_utils.py`

Manages hardware-related operations:

```python
def get_device():
    """Determines the best available device (GPU/CPU)."""
    
def set_device(device=None):
    """Sets the device for computation."""
    
def optimize_for_inference(model):
    """Optimizes a model for inference."""
```

#### `utils/common.py`

General utility functions:

```python
def set_seed(seed):
    """Sets random seeds for reproducibility."""
    
def timeit(method):
    """Decorator to measure execution time."""
    
def create_checkpoint_dir(base_dir, experiment_name):
    """Creates a directory for checkpoints."""
    
def moving_average(values, window):
    """Calculates moving average of values."""
    
def save_metrics(metrics, file_path):
    """Saves performance metrics to a file."""
```

## Entry Points

### `main.py`

The main entry point that provides a unified interface for all operations:

```python
def parse_args():
    """Parses command-line arguments."""
    
def setup_environment(config):
    """Sets up the game environment based on configuration."""
    
def main():
    """Main function to coordinate different modes."""
    
if __name__ == "__main__":
    main()
```

### `train.py`

Implements the training pipeline:

```python
def train(agent, env, config):
    """Trains the agent in the environment."""
    
def evaluate(agent, env, config):
    """Evaluates agent performance."""
    
def log_metrics(metrics, path):
    """Logs and visualizes training metrics."""
    
def main():
    """Runs the training process."""
```

### `play.py`

Implements the user interface for playing the game:

```python
def play_human(env, config):
    """Allows human play via keyboard controls."""
    
def play_ai(agent, env, config):
    """Shows AI agent playing the game."""
    
def render_game(env, mode, info):
    """Renders the game state."""
    
def main():
    """Main function for game playing interface."""
```

## Data Flow and Execution

The typical execution flow in the system follows this sequence:

1. **Initialization**:
   - Load configuration (from file/CLI arguments)
   - Set up environment and agent
   - Initialize hardware (CPU/GPU settings)

2. **Training Mode**:
   - Loop through episodes
   - For each step in episode:
     - Agent selects action
     - Environment executes action
     - Agent stores transition
     - Agent learns from experiences
   - Periodically evaluate and save model
   - Log performance metrics

3. **Playing Mode**:
   - Load trained model (or use human input)
   - Loop through episodes
   - For each step:
     - Agent/human selects action
     - Environment executes action
     - Render state
   - Display final statistics

## Key Algorithms

### Deep Q-Learning Core Algorithm

```python
# Initialize networks and replay buffer
policy_net = DQNModel(state_dim, action_dim)
target_net = DQNModel(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
buffer = ReplayBuffer(capacity, state_dim)

for episode in range(num_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Select action with ε-greedy policy
        if random.random() < epsilon:
            action = random.randint(0, action_dim - 1)
        else:
            q_values = policy_net(state)
            action = q_values.argmax().item()
        
        # Execute action in environment
        next_state, reward, done, info = env.step(action)
        
        # Store transition in buffer
        buffer.push(state, action, next_state, reward, done)
        
        # Learn from experiences if buffer has enough samples
        if len(buffer) >= batch_size:
            # Sample batch
            states, actions, next_states, rewards, dones = buffer.sample(batch_size)
            
            # Compute Q(s, a) from policy network
            q_values = policy_net(states).gather(1, actions)
            
            # Compute V(s') for next states using target network
            next_q_values = target_net(next_states).max(1, keepdim=True)[0]
            
            # Compute expected Q values
            expected_q_values = rewards + gamma * next_q_values * (1 - dones)
            
            # Compute loss and update policy network
            loss = F.smooth_l1_loss(q_values, expected_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Update state
        state = next_state
        
    # Periodically update target network
    if episode % target_update == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

### Dueling Network Architecture

```python
class DuelingDQNModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[128, 128]):
        super().__init__()
        
        # Feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU()
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[1]//2),
            nn.ReLU(),
            nn.Linear(hidden_dim[1]//2, 1)
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[1]//2),
            nn.ReLU(),
            nn.Linear(hidden_dim[1]//2, action_dim)
        )
    
    def forward(self, state):
        features = self.features(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
```

## Testing and Evaluation

The system includes extensive testing and evaluation capabilities:

```python
def evaluate(agent, env, num_episodes=10, render=False):
    """Evaluates agent performance over multiple episodes."""
    
    scores = []
    steps_list = []
    
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        score = 0
        steps = 0
        
        while not done:
            # Select best action (no exploration)
            action = agent.select_action(state, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Update metrics
            score += reward
            steps += 1
            state = next_state
            
            # Render if requested
            if render:
                env.render()
        
        # Record episode results
        scores.append(score)
        steps_list.append(steps)
    
    # Return evaluation metrics
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'mean_steps': np.mean(steps_list),
        'std_steps': np.std(steps_list)
    }
```

This source code overview provides a comprehensive understanding of the different components, their interactions, and the core algorithms implemented in the Snake-Python-DQLN project. 