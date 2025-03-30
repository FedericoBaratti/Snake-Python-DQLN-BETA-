# 💻 Specifica Tecnica e Implementazione

Questo documento contiene le specifiche tecniche dettagliate dell'implementazione del framework Snake con UI e Deep Q-Learning, con focus particolare sui pattern architetturali, le astrazioni algoritmiche e le ottimizzazioni a livello di codice.

## 📂 Architettura Software

Il sistema è implementato seguendo un'architettura esagonale (ports and adapters) con componenti disaccoppiati che comunicano attraverso interfacce ben definite:

```
snake-rl-project/
├── main.py                # Entry point con dependency injection e bootstrapping
├── backend/
│   ├── snake_game.py      # Core domain logic con invarianti di sistema
│   ├── environment.py     # Adapter per interfaccia Gymnasium
│   └── utils.py           # Infrastructure utilities e system services
├── frontend/
│   ├── ui.py              # Presentation layer con MVC pattern
│   ├── input_handler.py   # Command processor e event dispatcher
│   └── renderer.py        # Graphics pipeline e rendering engine
├── dqn_agent/
│   ├── models.py          # Neural network domain con factory methods
│   ├── dqn_agent.py       # RL strategy implementations
│   └── config.py          # Configuration management e dependency resolution
├── pretraining/
│   ├── synthetic_env.py   # Domain-specific lightweight environment
│   └── pretrain.py        # Training pipeline orchestrator
├── training/
│   └── train.py           # Distributed training coordinator
└── autoplay/
    └── autoplay.py        # Inference runtime e execution engine
```

## 🎮 Backend: Core Domain Implementation

### `snake_game.py`

Implementazione del core domain con pattern Entity-Component:

```python
class SnakeGame:
    """
    Implementazione del core domain con stato immutabile e transizioni pure.
    Applica pattern Command per le modifiche di stato e Observer per notifiche.
    
    Attributes:
        grid_size (int): Dimensione della griglia di gioco (invariante)
        snake (list): Posizioni discrete del serpente come list di tuple (x, y)
        direction (tuple): Vettore di direzione corrente (dx, dy)
        food (tuple): Posizione corrente del cibo
        score (int): Punteggio corrente
        game_over (bool): Flag di terminazione
        observers (list): Observer registrati per notifiche di eventi
    """
    
    def __init__(self, grid_size=10, seed=None):
        """
        Inizializza lo stato di gioco con invarianti di sistema.
        
        Args:
            grid_size (int): Dimensione della griglia di gioco (min: 5, max: 100)
            seed (int, optional): Seed per generazione deterministica
        
        Raises:
            ValueError: Se grid_size non rispetta le invarianti di dominio
        """
        if not 5 <= grid_size <= 100:
            raise ValueError(f"Grid size deve essere tra 5 e 100, ricevuto: {grid_size}")
            
        self.rng = np.random.RandomState(seed)
        self.grid_size = grid_size
        self.snake = [(grid_size // 2, grid_size // 2)]  # Posizione iniziale centrata
        self.direction = (1, 0)  # Inizializzazione direzione verso destra
        self.food = self._generate_food()
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.observers = []
        self.max_steps_without_food = grid_size * grid_size * 2  # Limite per prevenire loop infiniti
        self.steps_without_food = 0
        
    def register_observer(self, observer):
        """Registra un observer per notifiche di eventi (Pattern Observer)."""
        self.observers.append(observer)
        
    def notify_observers(self, event_type, data=None):
        """Notifica tutti gli observer registrati di un evento."""
        for observer in self.observers:
            observer.on_game_event(event_type, data)
        
    def step(self, action):
        """
        Esegue un passo di gioco applicando l'azione specificata (Pattern Command).
        L'azione è interpretata come cambiamento relativo di direzione:
        - 0: mantieni direzione
        - 1: gira a destra (90° orario)
        - 2: gira a sinistra (90° antiorario)
        
        Returns:
            tuple: (reward, game_over, score)
        """
        if self.game_over:
            return 0, True, self.score
            
        # Applicazione action come trasformazione della direzione
        self.direction = self._get_new_direction(action)
        
        # Calcolo della nuova posizione della testa
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction
        new_head = ((head_x + dir_x) % self.grid_size, 
                    (head_y + dir_y) % self.grid_size)
        
        # Controllo collisione con se stesso
        if new_head in self.snake:
            self.game_over = True
            self.notify_observers("collision", {"position": new_head, "type": "self"})
            return -1.0, True, self.score
            
        # Aggiungi la nuova testa
        self.snake.insert(0, new_head)
        
        # Controllo se il serpente ha mangiato il cibo
        ate_food = new_head == self.food
        reward = 0
        
        if ate_food:
            self.score += 1
            self.food = self._generate_food()
            reward = 1.0
            self.steps_without_food = 0
            self.notify_observers("food_eaten", {"position": new_head, "score": self.score})
        else:
            # Rimuovi l'ultimo segmento del serpente (la coda)
            self.snake.pop()
            reward = -0.01  # Piccola penalità per incoraggiare efficienza
            self.steps_without_food += 1
            
            # Controllo stallo
            if self.steps_without_food >= self.max_steps_without_food:
                self.game_over = True
                self.notify_observers("stalled", {"steps": self.steps_without_food})
                return -1.0, True, self.score
                
        self.steps += 1
        self.notify_observers("step", {
            "position": new_head, 
            "direction": self.direction,
            "steps": self.steps
        })
        
        return reward, self.game_over, self.score
        
    def _get_new_direction(self, action):
        """
        Calcola la nuova direzione in base all'azione.
        Implementa le regole di invarianti di dominio che impediscono inversioni di direzione.
        
        Args:
            action (int): Azione da eseguire (0: avanti, 1: destra, 2: sinistra)
            
        Returns:
            tuple: Nuova direzione come vettore (dx, dy)
        """
        dx, dy = self.direction
        
        if action == 1:  # Destra (rotazione 90° orario)
            return (dy, -dx)
        elif action == 2:  # Sinistra (rotazione 90° antiorario)
            return (-dy, dx)
        else:  # Avanti (mantieni direzione)
            return (dx, dy)
            
    def _generate_food(self):
        """
        Genera una nuova posizione per il cibo, evitando sovrapposizioni col serpente.
        Utilizza un algoritmo ottimizzato per grid sparse.
        
        Returns:
            tuple: Coordinate (x,y) del nuovo cibo
        
        Note:
            Per griglie molto grandi con serpenti lunghi, l'algoritmo si adatta
            automaticamente per ottimizzare la performance.
        """
        if len(self.snake) > self.grid_size * self.grid_size * 0.5:
            # Con serpenti molto lunghi (>50% della griglia), usa approccio diretto
            empty_cells = [
                (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                if (x, y) not in self.snake
            ]
            if not empty_cells:
                # Teoricamente impossibile in quanto il serpente può occupare
                # al massimo grid_size^2 celle, ma gestiamo comunque il caso
                return self.snake[-1]  # Ultima posizione della coda come fallback
                
            return empty_cells[self.rng.randint(0, len(empty_cells))]
        else:
            # Con serpenti più corti, usa approccio randomizzato con retry
            while True:
                food = (
                    self.rng.randint(0, self.grid_size),
                    self.rng.randint(0, self.grid_size)
                )
                if food not in self.snake:
                    return food
```

### `environment.py`

Adapter per l'interfaccia Gymnasium che isola il core domain:

```python
class SnakeEnv(gym.Env):
    """
    Environment compatibile con Gymnasium per interfacciare il gioco Snake
    con algoritmi di Reinforcement Learning.
    
    Implementa l'interfaccia standard Gym (reset, step, render) e
    fornisce metodi per la codifica dello stato e il calcolo delle ricompense.
    
    Attributes:
        observation_space (gym.spaces.Box): Spazio di osservazione
        action_space (gym.spaces.Discrete): Spazio delle azioni
        game (SnakeGame): Istanza incapsulata del core domain
        reward_shaping (bool): Flag per abilitare reward engineering
    """
    
    metadata = {"render_modes": ["human", "rgb_array", "ascii"], "render_fps": 10}
    
    def __init__(self, grid_size=10, reward_shaping=False, seed=None, render_mode=None):
        """
        Inizializza l'ambiente con gli spazi di osservazione e azione.
        
        Args:
            grid_size (int): Dimensione della griglia
            reward_shaping (bool): Se True, usa ricompense ingegnerizzate
            seed (int, optional): Seed per riproducibilità
            render_mode (str, optional): Modalità di rendering
        """
        super().__init__()
        
        # Definizione degli spazi (3 canali: serpente, cibo, testa)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(grid_size, grid_size, 3), dtype=np.float32
        )
        
        # Spazio azioni: 0=avanti, 1=destra, 2=sinistra
        self.action_space = spaces.Discrete(3)
        
        # Inizializzazione del core game
        self.game = SnakeGame(grid_size=grid_size, seed=seed)
        self.grid_size = grid_size
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Per reward shaping
        self._prev_distance_to_food = self._calculate_distance_to_food()
        
    def reset(self, seed=None, options=None):
        """
        Resetta l'ambiente ad uno stato iniziale e restituisce l'osservazione iniziale.
        
        Args:
            seed (int, optional): Seed per reset riproducibile
            options (dict, optional): Opzioni aggiuntive (non utilizzate)
            
        Returns:
            tuple: (observation, info dict)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.game = SnakeGame(grid_size=self.grid_size, seed=seed)
        else:
            self.game = SnakeGame(grid_size=self.grid_size)
            
        self._prev_distance_to_food = self._calculate_distance_to_food()
        
        obs = self._get_observation()
        info = {"score": 0}
        
        return obs, info
        
    def step(self, action):
        """
        Esegue un'azione nell'ambiente e restituisce il risultato.
        
        Args:
            action (int): Azione da eseguire (0=avanti, 1=destra, 2=sinistra)
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Esegui l'azione nel core game
        base_reward, done, score = self.game.step(action)
        
        # Calcola reward engineered se abilitato
        if self.reward_shaping:
            reward = self._calculate_shaped_reward(base_reward, done)
        else:
            reward = base_reward
            
        # Calcola nuova osservazione e info aggiuntive
        next_observation = self._get_observation()
        info = {"score": score}
        
        # Rendering se abilitato
        if self.render_mode == "human":
            self.render()
            
        return next_observation, reward, done, False, info
```

## 🧠 DQN Agent: Neural Engine Implementation

### `models.py`

Il modulo definisce le architetture neurali utilizzate, implementando diversi pattern di design:

```python
class ModelFactory:
    """
    Factory Method per creare modelli DQN con configurazioni specifiche.
    Implementa pattern Singleton per la registry dei modelli disponibili.
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name):
        """Decorator per registrare nuove architetture nel factory."""
        def decorator(model_class):
            cls._registry[name] = model_class
            return model_class
        return decorator
        
    @classmethod
    def create(cls, name, input_dim, output_dim, **kwargs):
        """
        Crea un'istanza del modello specificato.
        
        Args:
            name (str): Nome del modello registrato
            input_dim (int): Dimensionalità dell'input
            output_dim (int): Dimensionalità dell'output
            **kwargs: Parametri aggiuntivi specifici per il modello
            
        Returns:
            nn.Module: Istanza del modello richiesto
            
        Raises:
            ValueError: Se il nome del modello non è registrato
        """
        if name not in cls._registry:
            raise ValueError(f"Modello '{name}' non registrato. "
                           f"Modelli disponibili: {list(cls._registry.keys())}")
                           
        return cls._registry[name](input_dim, output_dim, **kwargs)


@ModelFactory.register("base")
class BaseDQN(nn.Module):
    """
    Modello DQN base con architettura fully-connected.
    
    Implementa un'architettura semplice con circa 3k parametri,
    ottimizzata per l'esecuzione su CPU e per training rapido.
    """
    
    def __init__(self, input_dim, output_dim, hidden_layers=None):
        """
        Inizializza il modello DQN base.
        
        Args:
            input_dim (int): Dimensionalità dell'input (stato flattened)
            output_dim (int): Dimensionalità dell'output (numero di azioni)
            hidden_layers (list, optional): Dimensioni dei layer nascosti
        """
        super(BaseDQN, self).__init__()
        
        # Default hidden layers
        if hidden_layers is None:
            hidden_layers = [64, 32]
            
        # Costruzione sequenziale dei layer
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Inizializzazione dei pesi per convergenza più rapida
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inizializzazione dei pesi con Xavier/Glorot."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        """Forward pass attraverso la rete."""
        return self.network(x)


@ModelFactory.register("dueling")
class DuelingDQN(nn.Module):
    """
    Implementazione dell'architettura Dueling DQN che separa
    il valore dello stato dall'advantage delle azioni.
    
    Architecture reference:
    Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning", 2016
    """
    
    def __init__(self, input_dim, output_dim, hidden_dims=None):
        """
        Inizializza il modello Dueling DQN.
        
        Args:
            input_dim (int): Dimensionalità dell'input
            output_dim (int): Dimensionalità dell'output (numero di azioni)
            hidden_dims (list, optional): Dimensioni dei layer nascosti
        """
        super(DuelingDQN, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 128]
            
        # Feature extractor condiviso
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        
        # Stream per il valore dello stato V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Stream per l'advantage A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
        # Inizializzazione dei pesi
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Inizializzazione dei pesi con Xavier/Glorot."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
                
    def forward(self, x):
        """
        Forward pass con separazione value-advantage e
        aggregazione robusta con sottrazione della media.
        
        Args:
            x (torch.Tensor): Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Q-values per ogni azione [batch_size, output_dim]
        """
        features = self.feature_layer(x)
        
        # Calcolo di valore e advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Aggregazione Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # La sottrazione della media serve per stabilità numerica e unicità della decomposizione
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
```

### `dqn_agent.py`

Implementazione core dell'agente DQN con tecniche avanzate di Deep Reinforcement Learning:

```python
class ReplayBuffer:
    """
    Implementazione ottimizzata di Experience Replay Buffer con supporto
    per sampling efficiente e prioritizzazione.
    
    Attributes:
        capacity (int): Capacità massima del buffer
        buffer (list): Lista circolare di esperienze
        position (int): Posizione corrente per l'inserimento
        priorities (numpy.ndarray): Priorità di ogni esperienza (per PER)
        weights (numpy.ndarray): Pesi per importance sampling (per PER)
        max_priority (float): Priorità massima per nuove esperienze
    """
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        """
        Inizializza un buffer di esperienze con supporto per Prioritized Experience Replay.
        
        Args:
            capacity (int): Dimensione massima del buffer
            alpha (float): Esponente per la prioritizzazione (0 = uniforme)
            beta (float): Esponente per importance sampling (1 = no correzione)
            beta_increment (float): Incremento di beta ad ogni sampling
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
    def push(self, state, action, next_state, reward, done):
        """
        Aggiunge una transizione al buffer con priorità massima.
        
        Args:
            state: Stato corrente
            action: Azione eseguita
            next_state: Stato risultante
            reward: Ricompensa ricevuta
            done: Flag di terminazione
        """
        experience = (state, action, next_state, reward, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        # Assegna priorità massima alle nuove esperienze per garantire
        # che vengano campionate almeno una volta
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """
        Campiona un batch di esperienze uniformemente.
        
        Args:
            batch_size (int): Dimensione del batch
            
        Returns:
            list: Batch di esperienze
        """
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def sample_prioritized(self, batch_size):
        """
        Campiona un batch di esperienze con probabilità proporzionale alle priorità.
        
        Args:
            batch_size (int): Dimensione del batch
            
        Returns:
            tuple: (esperienze, indici, importance_weights)
        """
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:len(self.buffer)]
            
        # Calcola probabilità di sampling P(i) ∝ p_i^α
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        # Campiona indici in base alle probabilità
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        # Calcola importance sampling weights
        # w_i = (N * P(i))^(-β)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalizzazione
        
        # Incrementa beta per la convergenza a 1
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        """
        Aggiorna le priorità delle esperienze in base al TD error.
        
        Args:
            indices (list): Indici delle esperienze
            priorities (list): Nuove priorità (tipicamente |TD error| + ε)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        """Restituisce il numero di esperienze nel buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Implementazione di un agente basato su Deep Q-Network con supporto per
    tecniche avanzate come Double DQN, Dueling DQN e Prioritized Experience Replay.
    
    Attributes:
        policy_net (nn.Module): Rete per la policy corrente
        target_net (nn.Module): Rete target per stabilizzare il training
        optimizer (torch.optim.Optimizer): Ottimizzatore
        memory (ReplayBuffer): Buffer per experience replay
        batch_size (int): Dimensione del batch per l'aggiornamento
        gamma (float): Fattore di sconto per ricompense future
        target_update (int): Frequenza di aggiornamento della rete target
        use_double_dqn (bool): Flag per abilitare Double DQN
        use_prioritized (bool): Flag per abilitare Prioritized Experience Replay
        device (torch.device): Dispositivo per il training (CPU/GPU)
    """
    
    def __init__(self, state_dim, action_dim, cfg):
        """
        Inizializza l'agente DQN con configurazione specificata.
        
        Args:
            state_dim (int): Dimensionalità dello stato
            action_dim (int): Numero di azioni possibili
            cfg (dict): Configurazione dei parametri dell'agente
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = cfg.get('gamma', 0.99)
        self.batch_size = cfg.get('batch_size', 128)
        self.target_update = cfg.get('target_update', 10)
        self.use_double_dqn = cfg.get('double_dqn', False)
        self.use_prioritized = cfg.get('prioritized_replay', False)
        self.prioritized_alpha = cfg.get('prioritized_alpha', 0.6)
        self.prioritized_beta = cfg.get('prioritized_beta', 0.4)
        self.n_step = cfg.get('n_step_returns', 1)
        self.max_grad_norm = cfg.get('gradient_clip', 1.0)
        self.use_amp = cfg.get('use_amp', False)
        
        # Inizializzazione hardware-aware
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cfg.get('force_cpu', False) else "cpu")
        
        # Creazione delle reti neurali
        model_type = cfg.get('model_type', 'base')
        self.policy_net = ModelFactory.create(
            model_type, 
            state_dim, 
            action_dim, 
            hidden_layers=cfg.get('hidden_layers', None)
        ).to(self.device)
        
        self.target_net = ModelFactory.create(
            model_type, 
            state_dim, 
            action_dim, 
            hidden_layers=cfg.get('hidden_layers', None)
        ).to(self.device)
        
        # Inizializzazione della rete target con gli stessi pesi della policy net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target net sempre in modalità valutazione
        
        # Configurazione dell'ottimizzatore
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), 
            lr=cfg.get('learning_rate', 1e-3),
            weight_decay=cfg.get('weight_decay', 0)
        )
        
        # Learning rate scheduler
        lr_scheduler = cfg.get('lr_scheduler', None)
        if lr_scheduler == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=cfg.get('lr_step_size', 1000),
                gamma=cfg.get('lr_gamma', 0.9)
            )
        elif lr_scheduler == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.get('lr_T_max', 10000),
                eta_min=cfg.get('lr_eta_min', 1e-5)
            )
        else:
            self.scheduler = None
            
        # Configurazione della memoria di replay
        if self.use_prioritized:
            self.memory = ReplayBuffer(
                cfg.get('buffer_size', 100000),
                alpha=self.prioritized_alpha,
                beta=self.prioritized_beta
            )
        else:
            self.memory = ReplayBuffer(cfg.get('buffer_size', 100000))
            
        # Mixed precision training setup
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Stats tracking
        self.train_step = 0
        
    def select_action(self, state, epsilon=0.1):
        """
        Seleziona un'azione usando la policy ε-greedy.
        
        Args:
            state: Stato corrente
            epsilon (float): Probabilità di esplorazione
            
        Returns:
            int: Azione selezionata
        """
        # Esplorazione: scelta casuale con probabilità epsilon
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        # Sfruttamento: scelta dell'azione con Q-value massimo
        with torch.no_grad():
            # Ottimizzazione: minimizza i trasferimenti CPU-GPU
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif isinstance(state, torch.Tensor) and state.device != self.device:
                state = state.to(self.device)
                
            q_values = self.policy_net(state)
            return q_values.argmax(dim=1).item()
            
    def optimize_model(self):
        """
        Aggiorna la rete neurale in base alle esperienze memorizzate.
        Implementa Double DQN e Prioritized Experience Replay se abilitati.
        
        Returns:
            float: Valore della loss
        """
        # Skip optimization if buffer isn't sufficiently filled
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Sample from experience buffer
        if self.use_prioritized:
            transitions, indices, weights = self.memory.sample_prioritized(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)
        
        # Unpack batch and convert to tensors
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch.done)).unsqueeze(1).to(self.device)
        
        # Mixed precision training context
        with torch.cuda.amp.autocast() if self.use_amp and torch.cuda.is_available() else nullcontext():
            # Compute current Q values
            current_q_values = self.policy_net(state_batch).gather(1, action_batch)
            
            # Compute next Q values using Double DQN or standard DQN
            with torch.no_grad():
                if self.use_double_dqn:
                    # Double DQN: usa policy_net per selezionare l'azione,
                    # target_net per valutarne il valore
                    next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
                    next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
                else:
                    # Standard DQN: usa target_net sia per selezione che valutazione
                    next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
                
                # Compute target Q values
                target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
            
            # Compute loss with importance sampling weights for prioritized replay
            td_errors = target_q_values - current_q_values
            loss = (weights * td_errors.pow(2)).mean()
            
        # Optimize model
        self.optimizer.zero_grad(set_to_none=True)  # Ottimizzazione di memoria
        
        if self.use_amp and torch.cuda.is_available():
            # Mixed precision training path
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard training path
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
        # Update learning rate if scheduler is defined
        if self.scheduler is not None:
            self.scheduler.step()
            
        # Update priorities in replay buffer if using prioritized replay
        if self.use_prioritized:
            new_priorities = torch.abs(td_errors).detach().cpu().numpy() + 1e-6  # small constant for stability
            self.memory.update_priorities(indices, new_priorities)
            
        self.train_step += 1
        
        # Update target network periodically
        if self.train_step % self.target_update == 0:
            self.update_target_network()
            
        return loss.item()
        
    def update_target_network(self):
        """
        Aggiorna la rete target copiando i parametri dalla policy network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def save(self, path):
        """
        Salva lo stato dell'agente su disco.
        
        Args:
            path (str): Percorso dove salvare il checkpoint
        """
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_step': self.train_step,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'use_double_dqn': self.use_double_dqn,
                'use_prioritized': self.use_prioritized,
                # Altri parametri di configurazione
            }
        }
        torch.save(checkpoint, path)
        
    def load(self, path):
        """
        Carica lo stato dell'agente da disco.
        
        Args:
            path (str): Percorso del checkpoint
            
        Returns:
            bool: True se il caricamento è riuscito, False altrimenti
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.train_step = checkpoint.get('train_step', 0)
            
            # Opzionalmente carica anche i parametri di configurazione
            config = checkpoint.get('config', {})
            if config:
                # Verifica compatibilità delle dimensioni
                if config.get('state_dim') != self.state_dim or config.get('action_dim') != self.action_dim:
                    print(f"AVVISO: Dimensioni del modello non corrispondenti. "
                         f"Caricato: state_dim={config.get('state_dim')}, action_dim={config.get('action_dim')}. "
                         f"Corrente: state_dim={self.state_dim}, action_dim={self.action_dim}.")
                    
            return True
        except Exception as e:
            print(f"Errore durante il caricamento del checkpoint: {e}")
            return False
```

### `config.py`

Gestione delle configurazioni per gli agenti DQN:

```python
class ModelConfig:
    """
    Configurazioni per i modelli DQN con parametri predefiniti
    e validazione dei valori.
    """
    
    # Configurazioni predefinite per diversi livelli di complessità
    CONFIGS = {
        "base": {
            "model_type": "base",
            "hidden_layers": [64, 32],
            "learning_rate": 1e-3,
            "batch_size": 64,
            "buffer_size": 10000,
            "gamma": 0.95,
            "target_update": 5,
            "double_dqn": False,
            "prioritized_replay": False,
            "n_step_returns": 1,
            "gradient_clip": 1.0,
            "use_amp": False,
            "weight_decay": 0.0,
            "lr_scheduler": None
        },
        "avanzato": {
            "model_type": "base",  # Stessa architettura ma più grande
            "hidden_layers": [128, 64, 32],
            "learning_rate": 5e-4,
            "batch_size": 128,
            "buffer_size": 50000,
            "gamma": 0.97,
            "target_update": 10,
            "double_dqn": True,
            "prioritized_replay": False,
            "n_step_returns": 1,
            "gradient_clip": 1.0,
            "use_amp": False,
            "weight_decay": 1e-5,
            "lr_scheduler": "StepLR",
            "lr_step_size": 1000,
            "lr_gamma": 0.9
        },
        "complesso": {
            "model_type": "dueling",
            "hidden_layers": [256, 128, 64, 32],
            "learning_rate": 1e-4,
            "batch_size": 256,
            "buffer_size": 100000,
            "gamma": 0.99,
            "target_update": 20,
            "double_dqn": True,
            "prioritized_replay": True,
            "prioritized_alpha": 0.6,
            "prioritized_beta": 0.4,
            "n_step_returns": 3,
            "gradient_clip": 0.5,
            "use_amp": True,
            "weight_decay": 1e-5,
            "lr_scheduler": "CosineAnnealingLR",
            "lr_T_max": 10000,
            "lr_eta_min": 1e-5
        },
        "perfetto": {
            "model_type": "dueling_attention",
            "hidden_layers": [512, 256, 128, 64],
            "learning_rate": 5e-5,
            "batch_size": 512,
            "buffer_size": 500000,
            "gamma": 0.995,
            "target_update": 50,
            "double_dqn": True,
            "prioritized_replay": True,
            "prioritized_alpha": 0.7,
            "prioritized_beta": 0.4,
            "n_step_returns": 5,
            "gradient_clip": 0.1,
            "use_amp": True,
            "weight_decay": 1e-6,
            "lr_scheduler": "CosineAnnealingLR",
            "lr_T_max": 20000,
            "lr_eta_min": 1e-6
        }
    }
    
    @classmethod
    def get_config(cls, complexity, override=None):
        """
        Ottiene la configurazione per il livello di complessità specificato,
        con possibilità di override di parametri specifici.
        
        Args:
            complexity (str): Livello di complessità (base, avanzato, complesso, perfetto)
            override (dict, optional): Parametri da sovrascrivere
            
        Returns:
            dict: Configurazione completa
            
        Raises:
            ValueError: Se il livello di complessità non è valido
        """
        if complexity not in cls.CONFIGS:
            valid_options = list(cls.CONFIGS.keys())
            raise ValueError(f"Livello di complessità '{complexity}' non valido. "
                           f"Opzioni valide: {valid_options}")
                           
        # Copia la configurazione base
        config = cls.CONFIGS[complexity].copy()
        
        # Applica override se specificati
        if override:
            for key, value in override.items():
                config[key] = value
                
        # Validazione della configurazione
        cls._validate_config(config)
        
        return config
        
    @staticmethod
    def _validate_config(config):
        """
        Valida i parametri della configurazione e imposta valori sicuri se necessario.
        
        Args:
            config (dict): Configurazione da validare
        """
        # Validazione batch_size
        if config.get('batch_size', 0) <= 0:
            config['batch_size'] = 64
            print("AVVISO: batch_size non valido, impostato a 64")
            
        # Validazione learning_rate
        lr = config.get('learning_rate', 0)
        if lr <= 0 or lr > 1:
            config['learning_rate'] = 1e-3
            print("AVVISO: learning_rate non valido, impostato a 1e-3")
            
        # Validazione gamma
        gamma = config.get('gamma', 0)
        if gamma <= 0 or gamma >= 1:
            config['gamma'] = 0.99
            print("AVVISO: gamma non valido, impostato a 0.99")
            
        # Validazione buffer_size
        if config.get('buffer_size', 0) < 1000:
            config['buffer_size'] = 10000
            print("AVVISO: buffer_size troppo piccolo, impostato a 10000")
```

## 🔄 Preaddestramento e Training

### `synthetic_env.py`

Implementa l'ambiente sintetico per il preaddestramento:

```python
class SyntheticSnakeEnv(gym.Env):
    def __init__(self, grid_size=8):
        # Versione semplificata dell'ambiente
        # ... (implementazione simile a SnakeEnv ma con segnali di reward più densi)
        
    def _calculate_reward(self, ate_food, done, prev_distance, new_distance):
        # Funzione di reward ingegnerizzata per il preaddestramento
        if ate_food:
            return 10.0
        elif done:
            return -10.0
        elif new_distance < prev_distance:
            return 0.5  # Ricompensa parziale per avvicinarsi al cibo
        else:
            return -0.1  # Piccola penalità per allontanarsi
```

### `pretrain.py`

Implementa l'algoritmo di preaddestramento:

```python
def pretrain_agent(agent, exp_generator, steps, batch_size, update_interval=100):
    rewards = []
    losses = []
    
    for step in range(steps):
        # Genera batch di esperienze sintetiche
        experiences = exp_generator.generate_batch(batch_size)
        
        # Memorizza le esperienze
        for state, action, next_state, reward, done in experiences:
            agent.memory.push(state, action, next_state, reward, done)
        
        # Aggiorna il modello
        loss = agent.optimize_model()
        losses.append(loss)
        
        # Aggiorna la rete target periodicamente
        if step % update_interval == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            
    return rewards, losses
```

### `train.py`

Implementa l'algoritmo di training nell'ambiente reale:

```python
def train_agent(model_type="base", episodes=5000, grid_size=10):
    # Inizializza ambiente e agente
    env = SnakeEnv(grid_size=grid_size)
    state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]
    action_dim = env.action_space.n
    
    cfg = get_config(model_type)
    agent = DQNAgent(state_dim, action_dim, cfg)
    
    # Carica modello preaddestrato se disponibile
    # ... (codice per caricare il checkpoint)
    
    # Loop principale di training
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # Seleziona azione con epsilon decrescente
            epsilon = max(cfg['epsilon_end'], cfg['epsilon'] * (cfg['epsilon_decay'] ** episode))
            action = agent.select_action(state, epsilon)
            
            # Esegui azione nell'ambiente
            next_state, reward, done, info = env.step(action)
            
            # Memorizza esperienza e ottimizza
            agent.memory.push(state, action, next_state, reward, done)
            agent.optimize_model()
            
            state = next_state
            total_reward += reward
            
        # Aggiorna rete target periodicamente
        if episode % cfg['target_update'] == 0:
            agent.update_target_network()
            
        # Salva checkpoint periodicamente
        # ... (codice per salvare il checkpoint)
```

## 🤖 Autoplay

### `autoplay.py`

Implementa il controller per la modalità autoplay:

```python
class AutoplayController:
    def __init__(self, env, model_complexity="base", checkpoint_path=None):
        # Inizializza il controller
        self.env = env
        self.state = self.env.reset()
        self.agent = self._load_agent(model_complexity, checkpoint_path)
        
        # Statistiche
        self.games_played = 0
        self.scores = []
        self.steps = 0
        
    def _load_agent(self, model_complexity, checkpoint_path):
        # Carica l'agente da checkpoint o crea un nuovo agente
        state_dim = self.env.observation_space.shape[0] * self.env.observation_space.shape[1] * self.env.observation_space.shape[2]
        action_dim = self.env.action_space.n
        
        cfg = get_config(model_complexity)
        agent = DQNAgent(state_dim, action_dim, cfg)
        
        if checkpoint_path:
            # Carica l'agente da checkpoint
            agent.load(checkpoint_path)
            
        return agent
        
    def get_action(self):
        # Ottiene l'azione dall'agente con epsilon=0 (modalità inferenza)
        action = self.agent.select_action(self.state, epsilon=0)
        next_state, reward, done, info = self.env.step(action)
        
        self.state = next_state
        self.steps += 1
        
        if done:
            self.games_played += 1
            self.scores.append(info['score'])
            self.state = self.env.reset()
            
        return action, done, info
        
    def get_stats(self):
        # Restituisce statistiche sull'autoplay
        if not self.scores:
            return {"games_played": 0, "avg_score": 0, "max_score": 0}
            
        return {
            "games_played": self.games_played,
            "avg_score": sum(self.scores) / len(self.scores),
            "max_score": max(self.scores),
            "last_scores": self.scores[-10:]
        }
```

## 🧩 Main (Punto di Ingresso)

### `main.py`

Punto di ingresso principale che coordina i vari componenti:

```python
def main():
    # Parsing degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description="Snake Game con DQN")
    parser.add_argument("--mode", choices=["manual", "train", "pretrain", "autoplay"], default="manual")
    parser.add_argument("--model", choices=["base", "avanzato", "complesso", "perfetto"], default="base")
    # ... (altri argomenti)
    args = parser.parse_args()
    
    if args.mode == "manual":
        # Avvia il gioco in modalità manuale
        game = SnakeGame(grid_size=args.grid_size)
        ui = SnakeUI(game)
        ui.run()
        
    elif args.mode == "train":
        # Avvia il training
        train_agent(model_type=args.model, episodes=args.episodes, grid_size=args.grid_size)
        
    elif args.mode == "pretrain":
        # Avvia il preaddestramento
        pretrain(model_type=args.model, steps=args.steps)
        
    elif args.mode == "autoplay":
        # Avvia la modalità autoplay
        env = SnakeEnv(grid_size=args.grid_size)
        controller = AutoplayController(env, model_complexity=args.model, checkpoint_path=args.checkpoint)
        ui = SnakeUI(env.game, autoplay=controller)
        ui.run()
```

## 🧪 Riferimenti Teorici

### Deep Q-Learning (DQN)

L'algoritmo DQN è stato introdotto da DeepMind nel 2013 e combina Q-Learning con reti neurali profonde. Le innovazioni chiave includono:

1. **Approssimazione della funzione Q**: La funzione Q è approssimata da una rete neurale profonda:
   ```
   Q(s, a; θ) ≈ Q*(s, a)
   ```
   dove θ sono i parametri della rete.

2. **Aggiornamento dei parametri**: I parametri vengono aggiornati per minimizzare la differenza tra il valore Q predetto e il valore target:
   ```
   L(θ) = E[(r + γ·max_a' Q(s', a'; θ') - Q(s, a; θ))²]
   ```
   dove θ' sono i parametri della rete target.

3. **Experience Replay**: Le esperienze (s, a, r, s') vengono memorizzate in un buffer e campionate casualmente per l'aggiornamento, riducendo la correlazione tra campioni consecutivi.

4. **Target Network**: Una rete separata con parametri θ' utilizzata per calcolare i valori target, aggiornata periodicamente per stabilizzare il training.

### Dueling DQN

L'architettura Dueling DQN separa la stima del valore di stato V(s) dalla stima dell'advantage A(s, a) delle azioni:

```
Q(s, a) = V(s) + (A(s, a) - 1/|A| * Σ_a' A(s, a'))
```

Questa separazione aiuta a:
- Valutare gli stati senza dover valutare l'effetto di ogni azione
- Identificare quali azioni sono migliori di altre

### Policy ε-greedy

La policy ε-greedy bilancia esplorazione e sfruttamento:
- Con probabilità ε: scegli un'azione casuale (esplorazione)
- Con probabilità 1-ε: scegli l'azione con il massimo Q-value (sfruttamento)

Durante il training, ε viene solitamente ridotto gradualmente per favorire lo sfruttamento delle azioni ottimali apprese.

## 🔄 Pattern Architetturali e Best Practices

Il sistema implementa diversi pattern di design per garantire modularità, manutenibilità ed estensibilità:

### 1. Dependency Injection

Il pattern Dependency Injection è utilizzato per gestire le dipendenze tra componenti senza creare accoppiamenti rigidi:

```python
# Esempio di Dependency Injection nel main file
def main():
    # Configurazione e creazione delle dipendenze
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Creazione dei componenti core
    config = ModelConfig.get_config(args.model, {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate
    })
    
    # Creazione dell'ambiente e dell'agente con DI
    env = create_environment(args.grid_size, args.reward_shaping)
    agent = create_agent(env.observation_space, env.action_space, config)
    
    # Injection nelle diverse modalità
    if args.mode == 'train':
        trainer = Trainer(env, agent, args.episodes)
        trainer.train()
    elif args.mode == 'autoplay':
        player = AutoplayController(env, agent)
        player.run()
    # ...
```

### 2. Factory Method

Il pattern Factory Method è utilizzato per la creazione di modelli con configurazione dinamica:

```python
# Già visto in ModelFactory
@classmethod
def create(cls, name, input_dim, output_dim, **kwargs):
    if name not in cls._registry:
        raise ValueError(f"Modello '{name}' non registrato")
        
    return cls._registry[name](input_dim, output_dim, **kwargs)
```

### 3. Strategy Pattern

Il pattern Strategy è utilizzato per implementare algoritmi intercambiabili:

```python
class RewardStrategy(ABC):
    """Strategia astratta per il calcolo delle reward."""
    
    @abstractmethod
    def calculate_reward(self, state, action, next_state, done, info):
        """Calcola la reward per una transizione."""
        pass
        
class BasicRewardStrategy(RewardStrategy):
    """Strategia base con reward sparse."""
    
    def calculate_reward(self, state, action, next_state, done, info):
        if info.get('ate_food', False):
            return 1.0
        elif done:
            return -1.0
        return -0.01

class ShapedRewardStrategy(RewardStrategy):
    """Strategia con reward engineering."""
    
    def calculate_reward(self, state, action, next_state, done, info):
        # Reward base
        reward = 0
        if info.get('ate_food', False):
            reward += 10.0
        elif done:
            reward -= 10.0
            
        # Reward per avvicinamento al cibo
        prev_distance = info.get('prev_distance_to_food', 0)
        curr_distance = info.get('curr_distance_to_food', 0)
        if curr_distance < prev_distance:
            reward += 0.1
            
        return reward
```

### 4. Observer Pattern

Il pattern Observer è utilizzato per notificare eventi di gioco:

```python
# Già visto in SnakeGame con register_observer e notify_observers

class GameObserver(ABC):
    """Interfaccia per observer di eventi di gioco."""
    
    @abstractmethod
    def on_game_event(self, event_type, data):
        """Gestisce un evento di gioco."""
        pass
        
class MetricsCollector(GameObserver):
    """Observer che raccoglie metriche di gioco."""
    
    def __init__(self):
        self.steps = 0
        self.foods_eaten = 0
        self.collisions = 0
        
    def on_game_event(self, event_type, data):
        if event_type == "step":
            self.steps += 1
        elif event_type == "food_eaten":
            self.foods_eaten += 1
        elif event_type == "collision":
            self.collisions += 1
```

## 🧪 Tecniche di Ottimizzazione

Il sistema implementa diverse tecniche di ottimizzazione per massimizzare le performance:

### 1. Vectorized Operations

Le operazioni vettoriali sono utilizzate per massimizzare la performance su CPU:

```python
def process_batch(states, actions, rewards, next_states, dones):
    """Elabora un batch di transizioni con operazioni vettoriali."""
    # Anziché elaborare ogni transizione singolarmente,
    # utilizziamo operazioni vettoriali numpy/torch
    
    # Esempio: calcola distanze con operazione vettoriale
    food_positions = extract_food_positions(states)
    head_positions = extract_head_positions(states)
    
    # Calcola distanze Manhattan in un'unica operazione
    distances = np.abs(head_positions - food_positions).sum(axis=1)
    
    # Calcola reward vettorizzate
    vectorized_rewards = rewards + 0.1 * (distances < 5)
    
    return vectorized_rewards
```

### 2. GPU Acceleration

Ottimizzazioni specifiche per sfruttare la GPU:

```python
def optimize_for_gpu(model, device):
    """Ottimizza un modello per l'esecuzione su GPU."""
    # Fusione delle operazioni per ridurre overhead di kernel launch
    model = torch.jit.script(model)
    
    # Utilizzo di memoria condivisa per batch processing
    if hasattr(model, 'share_memory'):
        model.share_memory()
        
    # Imposta stream priorità per operazioni critiche
    if device.type == 'cuda':
        torch.cuda.set_stream_priority(torch.cuda.current_stream(device), priority=1)
        
    return model
```

### 3. Memory Optimization

Tecniche per ridurre l'utilizzo di memoria:

```python
def optimize_memory_usage(batch_size, observation_shape, action_dim, buffer_size):
    """Configura strutture dati memory-efficient."""
    # Usa numpy arrays con tipo dati ottimizzato
    if max(observation_shape) < 256:
        obs_dtype = np.uint8  # Per osservazioni con valori piccoli
    else:
        obs_dtype = np.float16  # Half precision per osservazioni più grandi
        
    # Allocazione efficiente di buffer circolari
    observations = np.zeros((buffer_size,) + observation_shape, dtype=obs_dtype)
    actions = np.zeros(buffer_size, dtype=np.int8 if action_dim < 128 else np.int16)
    rewards = np.zeros(buffer_size, dtype=np.float16)  # Half precision per rewards
    dones = np.zeros(buffer_size, dtype=np.bool_)  # Boolean per flags done
    
    return observations, actions, rewards, dones
```

### 4. Parallel Training

Implementazione di training parallelo per sfruttare multi-core CPU:

```python
def train_parallel(num_processes, env_creator, agent_creator, episodes_per_process):
    """Esegue training parallelo su multiple CPU cores."""
    import multiprocessing as mp
    from multiprocessing import Queue
    
    # Coda per raccogliere risultati
    result_queue = Queue()
    
    def worker(proc_id, queue):
        # Crea ambiente e agente per questo processo
        env = env_creator()
        agent = agent_creator()
        
        # Esegui training locale
        total_reward = 0
        for episode in range(episodes_per_process):
            episode_reward = train_episode(env, agent)
            total_reward += episode_reward
            
        # Salva il modello e invia risultati
        checkpoint_path = f"checkpoint_{proc_id}.pt"
        agent.save(checkpoint_path)
        queue.put((proc_id, total_reward, checkpoint_path))
    
    # Crea e avvia processi
    processes = []
    for i in range(num_processes):
        p = mp.Process(target=worker, args=(i, result_queue))
        p.start()
        processes.append(p)
        
    # Raccogli risultati
    results = []
    for _ in range(num_processes):
        results.append(result_queue.get())
        
    # Attendi completamento
    for p in processes:
        p.join()
        
    # Elabora risultati, ad es. fai model averaging
    return results
```

## 📝 Best Practices Implementate

### 1. Gestione Errori Robusta

```python
def load_checkpoint(path, device=None):
    """Carica un checkpoint con gestione robusta degli errori."""
    try:
        if not os.path.exists(path):
            logging.error(f"Checkpoint non trovato: {path}")
            return None
            
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        checkpoint = torch.load(path, map_location=device)
        logging.info(f"Checkpoint caricato con successo: {path}")
        
        return checkpoint
    except Exception as e:
        logging.exception(f"Errore nel caricamento del checkpoint: {e}")
        return None
```

### 2. Logging Strutturato

```python
def setup_logging(level=logging.INFO, log_file=None):
    """Configura logging strutturato per il sistema."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurazione root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Handler per console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # Handler per file se specificato
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
    return logger
```

### 3. Controllo di Versione per Modelli

```python
class ModelVersion:
    """Gestisce versioni e compatibilità dei modelli."""
    
    CURRENT_VERSION = "1.2.0"
    
    @staticmethod
    def is_compatible(saved_version):
        """Verifica se una versione salvata è compatibile con quella corrente."""
        if saved_version is None:
            return False
            
        major, minor, _ = ModelVersion.CURRENT_VERSION.split('.')
        s_major, s_minor, _ = saved_version.split('.')
        
        # Verifica compatibilità
        if int(s_major) != int(major):
            return False  # Incompatibilità di versione major
            
        if int(s_minor) < int(minor):
            logging.warning(f"Versione minore precedente: {saved_version} vs {ModelVersion.CURRENT_VERSION}")
            
        return True
```

---

Questa guida fornisce una panoramica tecnica approfondita dell'implementazione del sistema. Per dettagli specifici sull'utilizzo pratico, consultare le guide [Installazione](INSTALLATION.md), [Training](TRAINING.md) e [Autoplay](AUTOPLAY.md). 