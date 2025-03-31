# Design of the Snake-Python-DQLN Project

## Architectural and Design Patterns

Snake-Python-DQLN implements a combination of architectural and design patterns to create a modular, extensible, and highly configurable system. The main patterns used include:

### 1. Model-View-Controller (MVC)

The system follows a modified MVC architecture:
- **Model**: Core business logic (`snake_game.py` and DQN components)
- **View**: Rendering system (`play.py` and `render()` methods)
- **Controller**: Control logic (`train.py` and coordination components)

This separation allows modifying the user interface without affecting the business logic, and vice versa.

### 2. Strategy Pattern

Implemented for learning strategies and interchangeable components:
- Neural network strategies (standard DQN, Dueling, Noisy)
- Buffer strategies (Standard, Prioritized)
- Configurable reward systems

Example of implementation in model selection:
```python
def create_model(model_type: str, state_dim: int, action_dim: int, **kwargs):
    if model_type == "dqn":
        return DQNModel(state_dim, action_dim, **kwargs)
    elif model_type == "dueling":
        return DuelingDQNModel(state_dim, action_dim, **kwargs)
    elif model_type == "noisy":
        return NoisyDQNModel(state_dim, action_dim, **kwargs)
    elif model_type == "noisy_dueling":
        return NoisyDuelingDQNModel(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 3. Factory Method

Used to create instances of complex components such as agents and environments:
```python
def get_agent(agent_type: str, state_dim: int, action_dim: int, config: Dict[str, Any]):
    if agent_type == "dqn":
        return DQNAgent(state_dim, action_dim, config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
```

### 4. Singleton Pattern (modified)

Implemented for components that should have only one instance, such as the configuration manager:
```python
class Config:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```

### 5. Dependency Injection

Widely used to inject dependencies into components:
```python
def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
    # Configuration injected from outside
    self.batch_size = config.get("batch_size", 32)
    self.learning_rate = config.get("lr", 1e-3)
    # ...
```

## Advanced Design Decisions

### 1. Multi-channel Tensor System

The game state is represented as a multi-channel tensor (N×N×3) that encodes:
1. **Channel 0**: Binary map of snake presence (1.0 = snake, 0.0 = empty)
2. **Channel 1**: Food position (1.0 = food, 0.0 = empty)
3. **Channel 2**: Directional information encoded as normalized values [0,1]

This representation allows the neural network to develop specialized convolutional filters for each aspect of the game during training.

State generation implementation code:
```python
def get_state(self) -> np.ndarray:
    # Initialize empty state
    state = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.float32)
    
    # Channel 0: Snake position
    for segment in self.snake:
        x, y = segment
        if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
            state[y, x, 0] = 1.0
    
    # Channel 1: Food position
    food_x, food_y = self.food
    if 0 <= food_x < self.grid_size and 0 <= food_y < self.grid_size:
        state[food_y, food_x, 1] = 1.0
    
    # Channel 2: Directional information
    head_x, head_y = self.snake[0]
    # Encode direction as normalized values [0,1]
    direction_value = self.direction.value / 3.0  # 0, 0.33, 0.66, 1.0
    state[head_y, head_x, 2] = direction_value
    
    return state
```

### 2. Parameterizable Reward System

The reward system design uses a dedicated class with 9 configurable parameters that allow fine-tuning incentives for the agent:

```python
self.default_config = {
    "reward_food": 10.0,           # Reward for eating food
    "reward_death": -10.0,         # Penalty for death
    "reward_step": -0.01,          # Penalty per step (efficiency)
    "reward_closer_to_food": 0.1,  # Moving closer to food
    "reward_farther_from_food": -0.1, # Moving away from food
    "reward_circular_movement": -0.5, # Penalty for circular movements
    "reward_efficient_path": 0.2,   # Reward for efficient path
    "reward_survival": 0.001,      # Survival
    "reward_growth": 0.5           # Snake growth
}
```

This reward scheme is the result of extensive experiments that have demonstrated an optimal balance between:
- Objective-based rewards (eating food)
- Progress-based rewards (moving closer to food)
- Penalties for undesired behaviors (circular movements)

The system also implements a circular movement detection algorithm through historical position analysis:

```python
def check_circular_movement(self, new_position: Tuple[int, int]) -> float:
    self.position_history.append(new_position)
    if len(self.position_history) > self.max_history_size:
        self.position_history.pop(0)
    
    # Detect positions visited multiple times in a short period
    if len(self.position_history) >= 8:
        position_counts = {}
        for pos in self.position_history:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        if max(position_counts.values()) > 2:
            return self.config["reward_circular_movement"]
    return 0.0
```

### 3. Specialized Neural Network Architectures

#### Dueling DQN Architecture

The implementation of the Dueling DQN architecture follows the original paper [Dueling Network Architectures for Deep Reinforcement Learning (Wang et al., 2016)](https://arxiv.org/abs/1511.06581), separating the calculation of state value V(s) and advantage of each action A(s,a):

```python
class DuelingDQNModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=[128, 128]):
        super(DuelingDQNModel, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU()
        )
        
        # Value stream (V)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[1]//2),
            nn.ReLU(),
            nn.Linear(hidden_dim[1]//2, 1)
        )
        
        # Advantage stream (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim[1], hidden_dim[1]//2),
            nn.ReLU(),
            nn.Linear(hidden_dim[1]//2, action_dim)
        )
    
    def forward(self, state):
        features = self.features(state)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combination with advantage normalization
        # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum_a' A(s,a'))
        return value + advantage - advantage.mean(dim=1, keepdim=True)
```

This architecture has empirically shown to improve learning, especially for tasks where the value of some actions can be independent of the state.

#### Noisy Layer for Parameterized Exploration

The implementation of the `NoisyLinear` layer replaces traditional ε-greedy exploration with parameterized exploration through noisy weights:

```python
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Factorized noise parameters
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        # Initialize parameters
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        # Generate new noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorized Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x):
        # Forward with noise
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
```

### 4. N-step Return Implementation

Implementation of N-step returns for more efficient value propagation:

```python
def compute_n_step_returns(self, rewards, next_values, dones, gamma=0.99, n_steps=1):
    """Compute n-step returns for each transition."""
    batch_size = len(rewards)
    n_step_returns = torch.zeros_like(rewards)
    
    for idx in range(batch_size):
        # Calculate return for each position in the batch
        G = 0.0
        # Calculate up to n steps or until terminal state
        for step in range(min(n_steps, len(rewards) - idx)):
            if dones[idx + step]:
                # Terminal state, only include reward
                G += (gamma ** step) * rewards[idx + step]
                break
            else:
                # Non-terminal state, include reward and bootstrapped value
                if step < n_steps - 1:
                    # Intermediate step
                    G += (gamma ** step) * rewards[idx + step]
                else:
                    # Last step, include bootstrapped value
                    G += (gamma ** step) * (rewards[idx + step] + gamma * next_values[idx + step])
        
        n_step_returns[idx] = G
    
    return n_step_returns
```

### 5. Double DQN Implementation

Implementation of Double DQN to reduce overestimation bias:

```python
def compute_target_values(self, next_states, rewards, dones):
    """Compute target values using Double DQN method."""
    with torch.no_grad():
        if self.double_dqn:
            # Select actions using policy network
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Evaluate actions using target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
        else:
            # Standard DQN: directly use max Q-value from target network
            next_q_values = self.target_net(next_states).max(1, keepdim=True)[0]
        
        # Compute target values
        target_values = rewards + (self.gamma * next_q_values * (1 - dones))
    
    return target_values
```

### 6. Experience Replay Implementation

#### Standard Replay Buffer

```python
class ReplayBuffer:
    def __init__(self, capacity, state_dim, device):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0
        
        # Pre-allocate memory for all components
        state_shape = (capacity, state_dim)
        self.states = np.zeros(state_shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.next_states = np.zeros(state_shape, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state, action, next_state, reward, done):
        """Store a transition."""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.next_states[self.position] = next_state
        self.rewards[self.position] = reward
        self.dones[self.position] = float(done)
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        states = torch.tensor(self.states[indices], device=self.device)
        actions = torch.tensor(self.actions[indices], device=self.device).unsqueeze(1)
        next_states = torch.tensor(self.next_states[indices], device=self.device)
        rewards = torch.tensor(self.rewards[indices], device=self.device).unsqueeze(1)
        dones = torch.tensor(self.dones[indices], device=self.device).unsqueeze(1)
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self):
        return self.size
```

#### Prioritized Replay Buffer

```python
class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, state_dim, device, alpha=0.6, beta=0.4, beta_increment=0.001, epsilon=1e-6):
        super().__init__(capacity, state_dim, device)
        self.alpha = alpha  # How much to prioritize
        self.beta = beta    # Importance sampling correction
        self.beta_increment = beta_increment  # Annealing rate
        self.epsilon = epsilon  # Small value to avoid zero priority
        
        # Initialize sum tree for priorities
        self.tree_capacity = 1
        while self.tree_capacity < capacity:
            self.tree_capacity *= 2
        
        self.sum_tree = np.zeros(2 * self.tree_capacity - 1)
        self.max_priority = 1.0
    
    def _propagate(self, idx, change):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.sum_tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _update(self, idx, priority):
        """Update priority at index."""
        change = priority - self.sum_tree[idx]
        self.sum_tree[idx] = priority
        self._propagate(idx, change)
    
    def _get_leaf(self, value):
        """Find index of leaf with cumulative sum > value."""
        idx = 0
        
        while idx < self.tree_capacity - 1:  # Non-leaf nodes
            left = 2 * idx + 1
            right = left + 1
            
            if value <= self.sum_tree[left]:
                idx = left
            else:
                value -= self.sum_tree[left]
                idx = right
        
        return idx - (self.tree_capacity - 1)  # Convert to data index
    
    def push(self, state, action, next_state, reward, done, error=None):
        """Store transition with priority."""
        # Use max priority for new experiences
        priority = self.max_priority if error is None else (abs(error) + self.epsilon) ** self.alpha
        
        # Store data as in standard buffer
        super().push(state, action, next_state, reward, done)
        
        # Update priority in sum tree
        idx = self.position - 1
        if idx < 0:  # Handle wrap-around
            idx = self.capacity - 1
            
        tree_idx = idx + self.tree_capacity - 1
        self._update(tree_idx, priority)
    
    def sample(self, batch_size):
        """Sample batch based on priorities."""
        indices = np.zeros(batch_size, dtype=np.int32)
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Segment tree into batch_size parts
        segment = self.sum_tree[0] / batch_size
        
        # Annealing beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate min priority for normalization
        p_min = np.min(self.sum_tree[self.tree_capacity-1:self.tree_capacity-1+self.size]) / self.sum_tree[0]
        
        for i in range(batch_size):
            # Get value from segment
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get leaf index
            idx = self._get_leaf(value)
            
            # Get normalized priority
            p_sample = self.sum_tree[idx + self.tree_capacity - 1] / self.sum_tree[0]
            
            # Calculate weight for importance sampling correction
            weights[i] = (p_sample / p_min) ** (-self.beta)
            indices[i] = idx
        
        # Normalize weights to be between 0 and 1
        weights /= np.max(weights)
        
        # Sample data as in standard buffer
        states = torch.tensor(self.states[indices], device=self.device)
        actions = torch.tensor(self.actions[indices], device=self.device).unsqueeze(1)
        next_states = torch.tensor(self.next_states[indices], device=self.device)
        rewards = torch.tensor(self.rewards[indices], device=self.device).unsqueeze(1)
        dones = torch.tensor(self.dones[indices], device=self.device).unsqueeze(1)
        weights = torch.tensor(weights, device=self.device).unsqueeze(1)
        
        return states, actions, next_states, rewards, dones, indices, weights
    
    def update_priorities(self, indices, errors):
        """Update priorities for indices based on TD errors."""
        for idx, error in zip(indices, errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update priority in sum tree
            tree_idx = idx + self.tree_capacity - 1
            self._update(tree_idx, priority)
```

### 7. Hyperparameter Selection

The choice of hyperparameters is based on extensive experimentation and ablation studies. The most critical hyperparameters and their selected values:

| Hyperparameter | Value | Justification |
|----------------|-------|---------------|
| Learning rate | 1e-4 | Provides stable learning without divergence |
| Discount factor (γ) | 0.99 | Strongly weights future rewards for long-term planning |
| ε-greedy decay | 10000 steps | Balances exploration and exploitation over time |
| Replay buffer size | 100000 | Large enough to avoid forgetting important experiences |
| Batch size | 64 | Balances computational efficiency and gradient stability |
| Target network update | 1000 steps | Frequent enough for learning stability without slowing down |
| Hidden layer dimensions | [128, 128] | Complex enough for the task without overfitting |
| Prioritization alpha | 0.6 | Moderate prioritization strength |
| Prioritization beta | 0.4 → 1.0 | Gradually increases importance sampling correction |

## Performance Optimization Techniques

### 1. Memory Optimization

```python
def optimize_memory_usage(self):
    """Optimize memory usage for large replay buffers."""
    # Use 32-bit floats instead of 64-bit
    self.states = self.states.astype(np.float32)
    self.next_states = self.next_states.astype(np.float32)
    self.rewards = self.rewards.astype(np.float32)
    
    # Use byte for boolean flags
    self.dones = self.dones.astype(np.uint8)
    
    # Use int8 for actions if possible
    if np.max(self.actions) < 128:
        self.actions = self.actions.astype(np.int8)
    
    # Pin memory for faster CPU->GPU transfer
    if self.device.type == 'cuda':
        self.pinned_states = torch.tensor(self.states).pin_memory()
        self.pinned_next_states = torch.tensor(self.next_states).pin_memory()
```

### 2. Inference Optimization

```python
def select_action(self, state, training=True):
    """Select action with optimized inference."""
    with torch.no_grad():  # Disable gradient computation
        state_tensor = torch.tensor(state, device=self.device).float().unsqueeze(0)
        
        if training and random.random() < self.epsilon:
            # Exploration: random action
            action = random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: best action from Q-values
            q_values = self.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
    
    return action
```

### 3. Batch Processing Optimization

```python
def process_batch(self, batch):
    """Process batch with optimized operations."""
    if self.double_dqn:
        # Vectorized double DQN computation
        with torch.no_grad():
            # Get actions from policy network
            next_q_values = self.policy_net(batch.next_states)
            next_actions = next_q_values.max(1, keepdim=True)[1]
            
            # Get Q-values for these actions from target network
            next_state_values = self.target_net(batch.next_states).gather(1, next_actions)
            
            # Compute target Q-values
            expected_q_values = batch.rewards + self.gamma * next_state_values * (1 - batch.dones)
    else:
        # Standard DQN computation
        with torch.no_grad():
            # Get maximum Q-value for next states
            next_state_values = self.target_net(batch.next_states).max(1, keepdim=True)[0]
            
            # Compute target Q-values
            expected_q_values = batch.rewards + self.gamma * next_state_values * (1 - batch.dones)
    
    # Compute Q-values for current states and actions
    current_q_values = self.policy_net(batch.states).gather(1, batch.actions)
    
    # Compute loss
    return self.loss_fn(current_q_values, expected_q_values)
```

## Performance Metrics and Evaluation

### 1. Agent Performance Metrics

The system tracks the following metrics during training and evaluation:

- **Score**: Total reward accumulated per episode
- **Episode Length**: Number of steps survived in each episode
- **Food Collected**: Number of food items eaten
- **Normalized Score**: Score divided by episode length
- **Q-value Statistics**: Min, max, mean, and standard deviation of Q-values
- **TD Error**: Mean squared TD error during training

### 2. Visualization Tools

```python
def plot_training_curves(metrics, save_path=None):
    """Plot training performance curves."""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot scores
    episodes = range(len(metrics['scores']))
    axs[0, 0].plot(episodes, metrics['scores'], 'b-')
    axs[0, 0].plot(episodes, np.convolve(metrics['scores'], np.ones(100)/100, mode='same'), 'r-')
    axs[0, 0].set_title('Episode Scores')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Score')
    
    # Plot episode lengths
    axs[0, 1].plot(episodes, metrics['episode_lengths'], 'g-')
    axs[0, 1].plot(episodes, np.convolve(metrics['episode_lengths'], np.ones(100)/100, mode='same'), 'r-')
    axs[0, 1].set_title('Episode Lengths')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Steps')
    
    # Plot Q-values
    axs[1, 0].plot(range(len(metrics['q_means'])), metrics['q_means'], 'y-')
    axs[1, 0].fill_between(range(len(metrics['q_means'])),
                           np.array(metrics['q_means']) - np.array(metrics['q_stds']),
                           np.array(metrics['q_means']) + np.array(metrics['q_stds']),
                           alpha=0.2)
    axs[1, 0].set_title('Q-value Statistics')
    axs[1, 0].set_xlabel('Training Step (×1000)')
    axs[1, 0].set_ylabel('Q-value')
    
    # Plot loss
    axs[1, 1].plot(range(len(metrics['losses'])), metrics['losses'], 'm-')
    axs[1, 1].set_title('TD Loss')
    axs[1, 1].set_xlabel('Training Step (×1000)')
    axs[1, 1].set_ylabel('Loss')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()
```

### 3. Agent Behavior Analysis

```python
def generate_heatmap(agent, env, num_episodes=10):
    """Generate heatmap of agent visitations."""
    grid_size = env.grid_size
    visitation = np.zeros((grid_size, grid_size))
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, _, done, _ = env.step(action)
            
            # Update visitation map
            head_pos = env.game.snake[0]
            x, y = head_pos
            visitation[y, x] += 1
            
            state = next_state
    
    # Normalize and plot
    visitation = visitation / num_episodes
    plt.figure(figsize=(10, 8))
    sns.heatmap(visitation, cmap='viridis', annot=False)
    plt.title('Agent Visitation Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()
    
    return visitation
```

## Conclusion

This detailed design document outlines the technical architecture, design patterns, and algorithms implemented in the Snake-Python-DQLN project. The system combines advanced reinforcement learning techniques with efficient software engineering practices to create a modular and extensible framework for studying deep reinforcement learning. 