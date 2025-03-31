# 🎓 Protocollo di Training Avanzato

Questa documentazione tecnica fornisce le specifiche complete dei processi di addestramento dell'agente DQN, dettagliando le metodologie di ottimizzazione, configurazioni parametriche e tecniche di accelerazione hardwareaware implementate.

## 📑 Architettura della Pipeline di Training

Il sistema implementa un processo di addestramento multi-fase con transfer learning progettato per ottimizzare efficienza computazionale e performance finali:

```
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│                           │     │                           │     │                           │
│  1. Initialization &      │────►│  2. Synthetic Pre-Training │────►│  3. Target Environment    │
│     Hardware Profiling    │     │     with Dense Rewards     │     │     Training              │
│                           │     │                           │     │                           │
└───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
                                                                                  │
                                                                                  ▼
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│                           │     │                           │     │                           │
│  6. Production Deployment │◄────│  5. Performance           │◄────│  4. Hyperparameter        │
│     & Inference           │     │     Evaluation            │     │     Optimization          │
│                           │     │                           │     │                           │
└───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
```

### 1. Initialization & Hardware Profiling

Prima fase del training pipeline che ottimizza le configurazioni in base all'hardware disponibile:

- **System Detection**: Identificazione automatica di:
  - Architettura CPU (x86-64, ARM64) e numero di core
  - Dispositivi CUDA/ROCm disponibili, compute capability e memoria
  - Memoria di sistema disponibile per replay buffer e batch processing
  - Acceleratori specializzati (TPU, ASIC)

- **Configuration Selection**: 
  - Selezione automatica dell'architettura neurale ottimale
  - Dimensionamento dinamico del replay buffer
  - Configurazione dei parametri di batch based su bandwidth memory
  - Distribuzione dei carichi di lavoro tra dispositivi

### 2. Pretraining Sintetico con Reward Engineering

Il pretraining sintetico utilizza un ambiente controllato con funzioni di reward engineered per accelerare l'apprendimento delle policy fondamentali:

- **Ambiente Sintetico Parametrizzato**:
  - Dimensione griglia ridotta (8x8, 10x10) per accelerare le iterazioni
  - Generazione statistica di stati iniziali che massimizzano la diversità
  - Reward shaping avanzato con segnali ad alta densità
  - Curriculum learning progressivo con difficoltà crescente

- **Reward Engineering Differenziato**:
```python
def synthetic_reward(self, state, action, next_state, done, ate_food):
    # Ricompensa primaria per obiettivi fondamentali
    if ate_food:
        r_primary = self.config.FOOD_REWARD  # 10.0 
    elif done:
        r_primary = self.config.COLLISION_PENALTY  # -10.0
    else:
        r_primary = self.config.STEP_PENALTY  # -0.05 (small penalty to encourage efficiency)
    
    # Reward component for distance improvement
    prev_distance = self._manhattan_distance(state)
    new_distance = self._manhattan_distance(next_state)
    r_distance = self.config.DISTANCE_FACTOR * (prev_distance - new_distance)
    
    # Reward component for avoiding traps (dead-ends)
    r_trap_avoidance = self._calculate_trap_avoidance(next_state) if not ate_food else 0
    
    # Reward for spatial distribution (exploration)
    r_exploration = self._calculate_exploration_bonus(next_state)
    
    # Weighted aggregation of components
    final_reward = (
        r_primary + 
        r_distance * self.config.DISTANCE_WEIGHT + 
        r_trap_avoidance * self.config.TRAP_AVOIDANCE_WEIGHT +
        r_exploration * self.config.EXPLORATION_WEIGHT
    )
    
    return final_reward
```

- **Transfer Learning**: 
  - Inizializzazione delle reti con pesi preaddestrati generici
  - Fine-tuning progressivo delle feature layers
  - Knowledge distillation da modelli di dimensioni maggiori

## 🧪 Implementazione del Deep Q-Learning

Il sistema implementa una versione avanzata dell'algoritmo Deep Q-Learning con numerose ottimizzazioni per stabilità e convergenza.

### Architettura Neurale per DQN

Per ogni livello di complessità modelliamo un'architettura neurale specifica:

#### 1. Modello Base (3,074 parametri)
```python
class BaseDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Rete semplice ma efficiente
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(), 
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
```

#### 2. Modello Avanzato (12,322 parametri)
```python
class AdvancedDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Rete più profonda con dropout per regolarizzazione
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.network(x)
```

#### 3. Modello Complesso (41,986 parametri)
```python
class ComplexDuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Feature extractor comune
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # Stream per il valore di stato
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Stream per l'advantage delle azioni
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Aggregazione con sottrazione della media per stabilità numerica
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
```

#### 4. Modello Perfetto (172,546 parametri)
```python
class PerfectDuelingAttentionDQN(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size=20):
        super().__init__()
        
        # Embedding spaziale per input processing avanzato
        self.spatial_embed = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU()
        )
        
        # Attention mechanism per focus sulle aree rilevanti
        self.attention = SpatialAttentionModule(512, grid_size)
        
        # Feature processing
        self.feature_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        
        # Value stream con residual connections
        self.value_stream = nn.Sequential(
            ResidualBlock(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream con residual connections
        self.advantage_stream = nn.Sequential(
            ResidualBlock(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # Processing con attention
        spatial_features = self.spatial_embed(x)
        attended = self.attention(spatial_features)
        features = self.feature_layer(attended)
        
        # Dueling architecture
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q-value computation con normalizzazione robusta
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
```

### Algoritmo DQN Avanzato

L'implementazione core del training DQN:

```python
def optimize_model(self):
    # Skip optimization if buffer isn't sufficiently filled
    if len(self.memory) < self.batch_size:
        return 0.0
        
    # Sample from prioritized replay memory
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
    
    # Mixed precision training if supported
    with torch.cuda.amp.autocast() if self.use_amp else contextlib.nullcontext():
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values using Double DQN
        if self.use_double_dqn:
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
        else:
            next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        
        # Compute target Q values
        target_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # Compute loss with importance sampling weights for prioritized replay
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
    # Optimize model
    self.optimizer.zero_grad()
    
    if self.use_amp:
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
    else:
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
    # Update priorities in replay buffer if using prioritized replay
    if self.use_prioritized:
        new_priorities = torch.abs(td_errors).detach().cpu().numpy() + self.prioritized_eps
        self.memory.update_priorities(indices, new_priorities)
        
    return loss.item()
```

## 🚄 Modalità di Training Rapido

Il sistema fornisce modalità ottimizzate per training accelerato e test rapidi:

### Modalità Demo

La modalità demo è pensata per dimostrare rapidamente il funzionamento del sistema di training e autoplay:

```bash
# Avvia la modalità demo
python main.py --demo
```

Questa modalità:
- Addestra un modello base per 300 episodi
- Utilizza una griglia di dimensioni ridotte (10x10)
- Al termine dell'addestramento mostra automaticamente l'agente in azione

È ideale per:
- Dimostrazioni rapide
- Test iniziali
- Verificare il funzionamento dell'intero sistema

### Modalità Train-and-Play

Per addestrare un modello e vederlo subito in azione:

```bash
# Addestra e poi osserva l'agente
python main.py --mode train-and-play --model base --episodes 500
```

Questa modalità:
- Addestra il modello per il numero specificato di episodi
- Salva automaticamente il modello in `training/checkpoints/dqn_MODEL_latest.pt`
- Avvia immediatamente la modalità autoplay con il modello appena addestrato

Puoi personalizzare ulteriormente questa modalità:
```bash
# Addestramento più intensivo seguito da autoplay
python main.py --mode train-and-play --model avanzato --episodes 1000 --grid-size 15 --speed 15
```

## 🧠 Concetti Fondamentali

### Deep Q-Learning (DQN)

DQN è un algoritmo di reinforcement learning che utilizza reti neurali per approssimare la funzione Q (che stima il valore atteso di un'azione in un certo stato). Caratteristiche chiave:

- **Q-Learning**: Algoritmo che apprende la funzione di valore ottimale Q*(s,a) secondo l'equazione di Bellman:
  ```
  Q*(s,a) = 𝔼[r + γ·max_a' Q*(s',a')]
  ```
  Dove r è la ricompensa immediata, γ è il fattore di sconto, s' lo stato futuro, e a' l'azione futura.

- **Approssimazione con Reti Neurali**: Utilizza reti profonde per generalizzare su stati mai visti, permettendo di gestire spazi di stato enormi

- **Experience Replay**: Memorizza le transizioni (s, a, r, s', done) in un buffer e le campiona casualmente per l'aggiornamento, riducendo la correlazione tra campioni consecutivi e migliorando la stabilità dell'apprendimento

- **Target Network**: Utilizza una rete separata con parametri θ' per calcolare i valori target, aggiornata periodicamente per stabilizzare il training:
  ```
  L(θ) = 𝔼[(r + γ·max_a' Q(s',a';θ') - Q(s,a;θ))²]
  ```

### Dueling DQN

Architettura avanzata che separa la stima del valore dello stato dalla stima dell'advantage (quanto un'azione sia migliore delle altre):

- **Value Stream**: Stima quanto è buono uno stato V(s) indipendentemente dall'azione scelta

- **Advantage Stream**: Stima quanto è migliore un'azione rispetto alle altre A(s,a) in un dato stato

- **Combinazione**: Unisce le due stime per calcolare i Q-values secondo la formula:
  ```
  Q(s,a) = V(s) + (A(s,a) - 1/|A|·∑_a' A(s,a'))
  ```
  Questa separazione consente all'agente di valutare stati senza dover valutare ogni singola azione, migliorando l'efficienza dell'apprendimento in spazi di azione limitati.

## 🔄 Fase 1: Preaddestramento Sintetico

### Scopo

Il preaddestramento accelera la convergenza del modello insegnando concetti base come:
- Evitare collisioni con i muri
- Muoversi verso il cibo
- Evitare di tornare indietro

Questo è particolarmente importante poiché permette al modello di superare la fase iniziale di esplorazione casuale, che può essere estremamente inefficiente.

### Ambiente Sintetico

L'ambiente sintetico è ottimizzato per:
- Generare esperienze velocemente
- Creare situazioni di apprendimento specifiche
- Fornire segnali di ricompensa più densi

```python
def _calculate_reward(self, has_eaten, has_collided, prev_dist, new_dist):
    """
    Calcola ricompense ingegnerizzate per guidare meglio l'apprendimento.
    """
    if has_eaten:
        return 10.0  # Ricompensa significativa per mangiare cibo
    elif has_collided:
        return -10.0  # Penalità significativa per collisioni
    elif new_dist < prev_dist:
        return 0.5  # Ricompensa parziale per avvicinarsi al cibo
    elif new_dist > prev_dist:
        return -0.2  # Piccola penalità per allontanarsi
    else:
        return -0.05  # Piccolissima penalità per incoraggiare movimento efficiente
```

### Come eseguire il preaddestramento

```bash
# Preaddestramento con modello base (CPU)
python -m pretraining.pretrain --model base --steps 500000

# Preaddestramento con modello complesso (GPU consigliata)
python -m pretraining.pretrain --model complesso --steps 1000000 --workers 4
```

### Parametri Configurabili

| Parametro         | Descrizione                                       | Valore Default | Impatto sul Training |
|-------------------|---------------------------------------------------|----------------|----------------------|
| `--model`         | Complessità del modello (base/avanzato/complesso/perfetto) | base    | Determina l'architettura della rete neurale e la capacità di rappresentazione |
| `--steps`         | Numero di passi di preaddestramento               | 500000        | Maggiori step = training più lungo ma potenzialmente più stabile |
| `--batch_size`    | Dimensione del batch per l'aggiornamento          | 64            | Batches più grandi = aggiornamenti più stabili ma richiedono più memoria |
| `--workers`       | Numero di worker per training parallelo           | auto           | Più workers = training più veloce se l'hardware lo supporta |
| `--seed`          | Seed per la riproducibilità                       | 42            | Stesso seed = stessi risultati, utile per debugging |
| `--checkpoint`    | Percorso del checkpoint da cui riprendere         | Nessuno       | Permette di continuare un training interrotto |
| `--learning_rate` | Velocità di apprendimento della rete              | 1e-3          | Controlla la dimensione degli aggiornamenti dei pesi |
| `--gamma`         | Fattore di sconto per ricompense future           | 0.99          | Valori più alti = maggiore importanza alle ricompense future |
| `--target_update` | Frequenza di aggiornamento della rete target      | 1000          | Valori più bassi = target più aggiornati ma training meno stabile |
| `--eval_interval` | Intervallo di valutazione (in steps)              | 10000         | Quanto spesso valutare le prestazioni |
| `--save_interval` | Intervallo di salvataggio (in steps)              | 50000         | Quanto spesso salvare i checkpoint |

### Configurazione ottimale per hardware specifico

| Hardware | Modello | Steps | Batch Size | Workers | Tempo stimato |
|----------|---------|-------|------------|---------|---------------|
| CPU standard | base | 300.000 | 64 | 1-2 | ~2 ore |
| Multi-core CPU | avanzato | 500.000 | 128 | 4 | ~3 ore |
| GPU singola | complesso | 1.000.000 | 256 | 2 | ~2 ore |
| Multi-GPU | perfetto | 2.000.000 | 512 | 8 | ~3 ore |

### Monitoraggio del Preaddestramento

Durante il preaddestramento, il sistema genera:
- **Log TensorBoard**: Visualizzazione in tempo reale delle metriche
- **Checkpoint periodici**: Salvataggio dello stato dell'agente
- **Grafici di apprendimento**: Ricompense, loss e altri indicatori

Per visualizzare i log TensorBoard:
```bash
tensorboard --logdir=logs/pretraining
```

Questo permette di monitorare metriche fondamentali come:
- **Loss della rete**: Indica la convergenza dell'apprendimento
- **Q-values medi**: Come l'agente valuta gli stati
- **Reward medio di valutazione**: Prestazioni effettive dell'agente
- **Distribuzione dei gradienti**: Per identificare problemi di instabilità

## 🔄 Fase 2: Training Reale

### Scopo

Il training reale perfeziona la policy appresa durante il preaddestramento, adattandola all'ambiente completo del gioco Snake. Mentre il preaddestramento fornisce una base solida, il training reale è essenziale per sviluppare strategie avanzate.

### Caratteristiche del Training Reale

- **Ricompense più sparse**: Solo per mangiare cibo e sopravvivere
- **Episodi più lunghi**: Il serpente può crescere significativamente
- **Strategia a lungo termine**: L'agente deve imparare a evitare situazioni di blocco
- **Ambiente più complesso**: Gestione dello spazio man mano che il serpente cresce

### Come eseguire il training reale

```bash
# Training con modello base (CPU)
python -m training.train --model base --episodes 5000

# Training con modello perfetto (GPU richiesta)
python -m training.train --model perfetto --episodes 10000 --grid_size 20

# Modalità semplificata: training seguito da autoplay
python main.py --mode train-and-play --model base --episodes 500 --grid-size 10

# Modalità demo: training rapido e autoplay
python main.py --demo
```

### Parametri Configurabili

| Parametro          | Descrizione                                      | Valore Default | Impatto sul Training |
|--------------------|--------------------------------------------------|----------------|----------------------|
| `--model`          | Complessità del modello                          | base           | Determina capacità, velocità e requisiti hardware |
| `--episodes`       | Numero di episodi di training                    | 5000           | Durata totale del training |
| `--grid_size`      | Dimensione della griglia di gioco                | 10             | Complessità dell'ambiente; griglia più grande = più difficile |
| `--workers`        | Numero di worker per training parallelo          | auto           | Accelerazione con hardware multi-core/GPU |
| `--pretrained`     | Checkpoint del modello preaddestrato             | auto           | Punto di partenza; migliore preaddestramento = convergenza più rapida |
| `--epsilon`        | Valore iniziale di epsilon per esplorazione      | 1.0            | Quanto è esplorativo l'agente all'inizio |
| `--epsilon_end`    | Valore finale di epsilon                         | 0.01           | Livello minimo di esplorazione mantenuto |
| `--epsilon_decay`  | Velocità di decadimento di epsilon               | 0.995          | Più alto = decadimento più lento dell'esplorazione |
| `--learning_rate`  | Velocità di apprendimento della rete             | 1e-4           | Dimensione degli aggiornamenti; valori più bassi = training più stabile |
| `--gamma`          | Fattore di sconto                                | 0.99           | Importanza delle ricompense future vs immediate |
| `--target_update`  | Frequenza aggiornamento rete target (in episodi) | 10             | Stabilità del training; valori più alti = più stabile ma convergenza più lenta |
| `--batch_size`     | Dimensione del batch per l'aggiornamento         | 128            | Compromesso tra efficienza e uso memoria |
| `--buffer_size`    | Dimensione del replay buffer                     | 100000         | Memoria delle esperienze; più grande = più diversità ma più RAM |
| `--eval_interval`  | Intervallo di valutazione (in episodi)           | 100            | Frequenza di valutazione delle prestazioni |
| `--save_interval`  | Intervallo di salvataggio (in episodi)           | 500            | Frequenza di salvataggio dei checkpoint |
| `--double_dqn`     | Abilita Double DQN                               | True           | Riduce la sovrastima dei Q-values |
| `--prioritized`    | Abilita Prioritized Experience Replay            | False          | Focus su esperienze più informative |

### Tecniche avanzate implementabili

1. **Double DQN**: Riduce la sovrastima dei Q-values usando la rete policy per selezionare l'azione e la rete target per stimarne il valore:

```python
# Implementazione Double DQN
next_state_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
next_state_values = self.target_net(next_states).gather(1, next_state_actions)
```

2. **Prioritized Experience Replay**: Campiona esperienze con priorità basata sull'errore TD:

```python
# Calcolo della priorità per Prioritized Experience Replay
td_error = abs(target_q_value - current_q_value).detach().cpu().numpy()
priority = (td_error + epsilon) ** alpha  # epsilon previene priorità zero
```

3. **Curriculum Learning**: Aumenta gradualmente la difficoltà durante il training:

```python
# Implementazione di curriculum learning
def adaptive_grid_size(episode, max_episodes):
    """Aumenta gradualmente la dimensione della griglia."""
    min_size = 8
    max_size = 20
    return min(
        min_size + int((max_size - min_size) * (episode / (max_episodes * 0.7))),
        max_size
    )
```

### Ottimizzazione Hardware

Il sistema ottimizza automaticamente l'utilizzo delle risorse:

- **CPU multi-core**: Distribuisce il training su più worker
- **CUDA/GPU**: Utilizza accelerazione hardware quando disponibile
- **Memoria**: Adatta le dimensioni del replay buffer alla RAM disponibile

#### Multi-GPU Training

Per sfruttare pienamente più GPU, implementiamo sia DataParallel che DistributedDataParallel:

```python
# Uso di DataParallel per training su singola macchina multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Utilizzo di {torch.cuda.device_count()} GPU")
    agent.policy_net = nn.DataParallel(agent.policy_net)
    agent.target_net = nn.DataParallel(agent.target_net)
```

#### Mixed Precision Training

Per GPU con supporto Tensor Cores (NVIDIA Volta+), implementiamo mixed precision training:

```python
# Abilita mixed precision training
if use_mixed_precision and torch.cuda.is_available():
    from torch.cuda.amp import autocast, GradScaler
    scaler = GradScaler()
```    

## 📊 Sessioni di Training Tipiche

### Training Veloce (per dimostrazione)

```bash
python main.py --demo
```

Per un training veloce di un modello base (circa 5 minuti su CPU standard), che permette di:
- Vedere rapidamente il processo di training
- Ottenere un modello funzionante per dimostrazioni
- Valutare la configurazione del sistema

### Training Standard (risultati decenti)

```bash
python main.py --mode train-and-play --model avanzato --episodes 1000 --grid-size 15
```

Per un training medio di un modello avanzato (circa 20-30 minuti su CPU multi-core), che fornisce:
- Un agente competente con strategie basilari
- Capacità di evitare ostacoli e trovare cibo efficacemente
- Performance stabili su griglie di medie dimensioni

### Training Profondo (risultati ottimali)

```bash
python main.py --mode train --model complesso --episodes 5000
```

Per un training approfondito (2-4 ore con GPU), che produce:
- Agente con strategie avanzate
- Capacità di gestire situazioni complesse
- Pianificazione a lungo termine efficace

## 📈 Monitoraggio e Analisi del Training

Durante il training, puoi monitorare vari parametri:

```bash
# Monitora il training con TensorBoard
tensorboard --logdir=training/logs
```

I grafici generati alla fine del training mostrano:
- **Curva di apprendimento**: Ricompensa media per episodio
- **Q-values**: Evoluzione della stima di valore
- **Epsilon**: Decadimento del fattore di esplorazione
- **Loss**: Andamento della funzione di perdita

### Analisi Post-Training

Dopo il training, valuta le prestazioni del modello:

```bash
# Valuta il modello su 100 episodi
python main.py --mode autoplay --model base --checkpoint training/checkpoints/dqn_base_latest.pt
```

Puoi anche riavviare l'addestramento da un checkpoint esistente:

```bash
# Riprendi l'addestramento da un checkpoint
python main.py --mode train --model base --checkpoint training/checkpoints/dqn_base_latest.pt --episodes 500
```

## 🏆 Risultati Attesi

| Complessità Modello | Episodi | Punteggio Medio Atteso | Tempo di Training* |
|---------------------|---------|------------------------|-------------------|
| Base (Demo)         | 300     | 15-30                  | 5-10 minuti       |
| Base                | 5000    | 30-50                  | 30-60 minuti      |
| Avanzato            | 7000    | 50-100                 | 1-2 ore           |
| Complesso           | 10000   | 100-200                | 3-5 ore           |
| Perfetto            | 20000   | 200-400                | 8-12 ore          |

*I tempi di training variano significativamente in base all'hardware disponibile.

---

Dopo aver completato il training, potrai utilizzare il modello addestrato in modalità autoplay. Consulta la [Guida alla Modalità Autoplay](AUTOPLAY.md) per ulteriori dettagli. 