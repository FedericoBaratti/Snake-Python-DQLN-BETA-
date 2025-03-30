# 🏗️ Architettura del Sistema - Specifiche Tecniche

Questo documento tecnico descrive l'architettura software completa del sistema Snake con UI e Deep Q-Learning, con focus sui pattern di design, componenti algoritmici e ottimizzazioni runtime implementate.

## 📊 Modello Architetturale

Il sistema è implementato secondo un'architettura a eventi basata su microservizi, con disaccoppiamento totale tra componenti attraverso interfacce SOLID:

```
                                 ┌───────────────────────┐
                                 │                       │
                                 │   UI Engine (Pygame)  │◄───────────┐
                                 │  Rendering Pipeline   │            │
                                 │                       │            │
                                 └────────────┬──────────┘            │
                                              │                       │
                                              ▼                       │
┌───────────────────────┐          ┌─────────────────────┐    ┌──────┴──────────┐
│                       │          │                     │    │                  │
│  Synthetic Training   │◄─────────┤   Core Engine      ├────►   Inference      │
│  Data Generation      │          │   (Gymnasium)      │    │   Controller     │
│                       │          │                     │    │                  │
└───────────┬───────────┘          └─────────┬───────────┘    └──────┬──────────┘
            │                                 │                       │
            │                                 ▼                       │
            │                      ┌─────────────────────┐            │
            │                      │                     │            │
            └─────────────────────►│  Neural Engine      │◄───────────┘
                                   │  (PyTorch Backend)  │
                                   │                     │
                                   └─────────────────────┘

                                   │                     │
                                   ▼                     ▼
                        ┌─────────────────┐   ┌─────────────────┐
                        │                 │   │                 │
                        │  CPU Execution  │   │  GPU/TPU        │
                        │  Vectorization  │   │  Acceleration   │
                        │                 │   │                 │
                        └─────────────────┘   └─────────────────┘
```

## 🧩 Componenti Core

Il sistema è composto da sei componenti primari, ciascuno progettato secondo i principi SOLID con dipendenze unidirezionali e interfacce ben definite:

### 1. Core Engine (`backend/`)

Implementa il motore di simulazione parametrizzabile e l'ambiente di apprendimento:

- **Snake Kernel** (`snake_game.py`): Implementa un sistema di simulazione asincrono ottimizzato per operazioni vettoriali con una macchina a stati finiti (FSM) per la gestione del ciclo di vita del gioco.

- **Gymnasium Environment** (`environment.py`): Fornisce un'interfaccia standardizzata conforme alle specifiche Gymnasium 0.29.1 con supporto per observation encoding parametrizzabile e meccanismi di reward engineering avanzati.

- **System Utilities** (`utils.py`): Implementa componenti di sistema critici, tra cui detection hardware dynamica con profiling delle risorse disponibili, gestione intelligente della memoria e serializzazione ottimizzata.

### 2. UI Engine (`frontend/`)

Framework grafico per la visualizzazione e l'interazione:

- **UI Orchestrator** (`ui.py`): Gestisce il ciclo di rendering e implementa una pipeline di visualizzazione hardware-accelerata con supporto per:
  - Rendering vettoriale ad alta precisione
  - Interfaccia multi-layered con compositing parallelo
  - Sistema di animazione parametrizzabile
  - Visualizzazione in tempo reale delle metriche di gioco ed agent

- **Event System** (`input_handler.py`): Implementa un sistema di eventi asincrono con:
  - Buffering degli input con priority queue
  - Gestione avanzata del parallelismo per input handling non bloccante
  - State machine per il mapping contestuale dei comandi

- **Rendering Engine** (`renderer.py`): Fornisce primitives grafiche ottimizzate con:
  - Hardware-acceleration trasparente su piattaforme supportate
  - Pipeline di rendering configurabile con supporto per shaders custom
  - Buffer management avanzato con double-buffering automatico

### 3. Neural Engine (`dqn_agent/`)

Implementa il framework di Deep Reinforcement Learning con architetture neurali avanzate:

- **Model Factory** (`models.py`): Definisce le architetture neurali con implementazioni parametriche di:
  - Standard Deep Q-Network con layer fully-connected e attivazioni ReLU/LeakyReLU
  - Dueling Deep Q-Network con separazione Value/Advantage e aggregazione statisticamente corretta
  - Attention-augmented DQN con meccanismi di self-attention per catturare dipendenze spaziali
  - Configurazioni di normalizzazione (Batch Norm, Layer Norm, Group Norm) per stabilizzazione del training

- **Reinforcement Core** (`dqn_agent.py`): Implementa l'algoritmo DQN avanzato con:
  - Experience Replay buffer con stratificazione temporale e prioritizzazione TD-error
  - Target Network con soft/hard update parametrizzabile e Polyak averaging
  - Algoritmi di exploration avanzati (epsilon-greedy, Boltzmann exploration, UCB)
  - Double Q-Learning per mitigazione del bias di sovrastima
  - N-step Returns per bilanciamento tra bias e varianza nella propagazione del reward

- **Configuration Manager** (`config.py`): Framework di gestione delle configurazioni con:
  - Definizione modulare degli iperparametri con validazione dinamica
  - Mapping automatico delle configurazioni allo hardware disponibile
  - Sistema di versioning per garantire compatibilità dei checkpoint
  - Auto-tuning delle configurazioni basato sul profiling delle performance

### 4. Synthetic Training System (`pretraining/`)

Implementa un sistema di preaddestramento accelerato per convergenza rapida:

- **Synthetic Environment** (`synthetic_env.py`): Ambiente di simulazione ottimizzato con:
  - Generazione statistica di stati iniziali con distribuzione bilanciata
  - Funzione di reward ingegnerizzata con segnali densi per guidare l'apprendimento iniziale
  - Astrazione delle dinamiche complesse per focalizzare l'apprendimento sui pattern fondamentali
  - Generazione di scenari edge-case per migliorare la robustezza dell'agente

- **Pretraining Pipeline** (`pretrain.py`): Sistema di addestramento accelerato con:
  - Parallelizzazione massiva della generazione di esperienze sintetiche
  - Curriculum learning automatico con progressione dinamica della difficoltà
  - Knowledge distillation da modelli preaddestrati per transfer learning efficiente
  - Checkpointing incrementale con priorità sulle configurazioni performanti

### 5. Distributed Training Engine (`training/`)

Sistema avanzato di orchestrazione del training con ottimizzazione multi-device:

- **Training Orchestrator** (`train.py`): Pipeline di training completa con:
  - Distribuzione automatica del carico su CPU/GPU/TPU disponibili
  - Implementazione di tecniche avanzate come Prioritized Experience Replay, Double DQN, Dueling DQN
  - Integrazione con TensorBoard per monitoraggio real-time delle metriche di performance
  - Strategie di sampling dinamiche per ottimizzare l'utilizzo della memoria di replay
  - Sistema di early stopping basato su metriche di performance con plateau detection

- **Checkpoint System**: Meccanismo di persistence sofisticato con:
  - Serializzazione efficiente dei modelli con supporto per compressione
  - Metadata storage per tracciabilità completa della configurazione
  - Ripristino del training da qualsiasi checkpoint con ricalibrazione degli iperparametri
  - Versioning automatico per garantire compatibilità forward/backward

### 6. Inference System (`autoplay/`)

Runtime di esecuzione ottimizzato per l'inferenza:

- **Autoplay Controller** (`autoplay.py`): Gestore dell'esecuzione automatica con:
  - Policy deterministica ottimizzata per inferenza rapida
  - Registrazione dettagliata delle metriche di performance
  - Visualizzazione delle decision boundary e confidence level
  - Sistema di analisi statistica delle performance con aggregazione multi-run

## 🔄 Flusso di Esecuzione

### Modalità Manuale

1. L'utente avvia il gioco in modalità manuale
2. L'interfaccia utente inizializza il gioco e visualizza la griglia
3. L'input handler cattura gli input dell'utente e li traduce in azioni
4. Il core game aggiorna lo stato in base alle azioni
5. L'UI renderizza il nuovo stato

### Modalità Training

1. L'ambiente viene inizializzato
2. L'agente DQN viene creato con configurazione specificata
3. Per ogni episodio:
   - L'ambiente viene resettato
   - L'agente esegue azioni e riceve ricompense
   - Le esperienze vengono memorizzate nel replay buffer
   - La rete DQN viene aggiornata in base alle esperienze
4. I checkpoint vengono salvati periodicamente
5. Le metriche di performance vengono registrate

### Modalità Autoplay

1. L'utente avvia il gioco in modalità autoplay
2. Il sistema carica un agente preaddestrato
3. L'autoplay controller interfaccia l'agente con l'ambiente
4. L'agente seleziona le azioni in base alla policy appresa
5. L'UI visualizza il gioco in tempo reale

## 🧠 Architettura DQN

L'agente DQN implementa diverse tecniche avanzate:

### Architettura Dueling DQN

```
                          ┌────────────────┐
                          │                │
                          │   Input Layer  │
                          │                │
                          └────────┬───────┘
                                   │
                           ┌───────┴───────┐
                           │               │
                           │  Hidden Layers│
                           │               │
                           └───────┬───────┘
                                   │
                      ┌────────────┴────────────┐
                      │                         │
          ┌───────────▼──────────┐   ┌──────────▼───────────┐
          │                      │   │                      │
          │    Value Stream      │   │   Advantage Stream   │
          │                      │   │                      │
          └───────────┬──────────┘   └──────────┬───────────┘
                      │                         │
                      │                         │
          ┌───────────▼──────────────────────▼──────────────┐
          │                                                 │
          │          Q-Value Combination Layer              │
          │                                                 │
          └─────────────────────┬─────────────────────────┘
                                │
                      ┌─────────▼──────────┐
                      │                    │
                      │     Output Layer   │
                      │                    │
                      └────────────────────┘
```

### Experience Replay

Il sistema utilizza un buffer di esperienze per memorizzare le transizioni (stato, azione, nuovo stato, ricompensa, flag di terminazione) e consentire l'apprendimento off-policy.

```
┌──────────────────────────────────────────┐
│           Experience Buffer              │
├──────────┬────────┬──────────┬───────┬───┤
│  Stato   │ Azione │   Stato  │Ricom- │Done│
│  attuale │        │  nuovo   │pensa  │    │
├──────────┼────────┼──────────┼───────┼───┤
│    S₁    │   A₁   │    S₂    │   R₁  │ F₁ │
├──────────┼────────┼──────────┼───────┼───┤
│    S₂    │   A₂   │    S₃    │   R₂  │ F₂ │
├──────────┼────────┼──────────┼───────┼───┤
│    ...   │   ...  │    ...   │   ... │... │
└──────────┴────────┴──────────┴───────┴───┘
```

## 🔌 Interazione tra Componenti

### Comunicazione UI-Core

L'interfaccia utente comunica con il core del gioco attraverso un'architettura event-driven. Gli eventi di input vengono tradotti in azioni che modificano lo stato del gioco, che viene poi renderizzato.

### Interazione DQN-Ambiente

L'agente DQN interagisce con l'ambiente attraverso l'interfaccia Gymnasium:

1. L'agente osserva lo stato corrente dell'ambiente
2. Seleziona un'azione in base alla policy corrente
3. L'ambiente esegue l'azione e restituisce il nuovo stato, la ricompensa e un flag di terminazione
4. L'agente memorizza la transizione e aggiorna la policy

## 📊 Gestione Risorse Hardware

Il sistema è progettato per ottimizzare automaticamente l'utilizzo delle risorse hardware:

- **Rilevamento Hardware**: All'avvio, il sistema rileva CPU, GPU e memoria disponibile
- **Scaling Automatico**: Il training viene adattato in base alle risorse disponibili
- **Multi-processing**: Utilizza tutti i core CPU disponibili per operazioni parallele
- **CUDA Optimization**: Sfrutta le GPU NVIDIA per accelerare il training

## 🔍 Architettura di Monitoraggio

Il sistema include strumenti per monitorare le prestazioni dell'agente e la qualità del training:

- **TensorBoard Integration**: Logging in tempo reale di metriche come ricompense, lunghezza episodi e loss
- **Checkpoint System**: Salvataggio periodico dello stato dell'agente e ripristino del training
- **Evaluation Loop**: Valutazione periodica delle prestazioni dell'agente su episodi di test

## 🧩 Estensibilità

L'architettura è progettata per essere facilmente estensibile:

- **Nuovi Modelli**: Puoi aggiungere nuove architetture di rete in `models.py`
- **Algoritmi Alternativi**: L'implementazione modulare consente di sostituire DQN con altri algoritmi come PPO o A2C
- **Ambienti Personalizzati**: Puoi creare nuovi ambienti che seguono l'interfaccia Gymnasium

## 📝 Note sullo Sviluppo

Per contribuire allo sviluppo, tieni presente:

- **Principi SOLID**: L'architettura segue i principi SOLID per garantire manutenibilità
- **Testing Automatizzato**: Le componenti principali hanno test unitari
- **Documentazione del Codice**: Docstring dettagliate e commenti esplicativi

---

Questo documento fornisce una panoramica dell'architettura del sistema. Per dettagli specifici sull'implementazione, consulta la [Guida al Codice](CODE_GUIDE.md) e i commenti nel codice sorgente. 

## 🔄 Pipeline di Esecuzione

### Pipeline di Addestramento Completa

La pipeline di addestramento ottimizzata prevede i seguenti stage:

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │     │                   │
│ 1. Hardware       │────►│ 2. Synthetic      │────►│ 3. Full           │────►│ 4. Evaluation     │
│    Profiling      │     │    Pretraining    │     │    Training       │     │    & Analysis     │
│                   │     │                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘     └───────────────────┘
                                    │                         │                         │
                                    ▼                         ▼                         ▼
                          ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
                          │                   │     │                   │     │                   │
                          │ ● Experience      │     │ ● Experience      │     │ ● Performance     │
                          │   Generation      │     │   Collection      │     │   Metrics         │
                          │ ● Policy Init     │     │ ● Policy Update   │     │ ● Generalization  │
                          │ ● Base Learning   │     │ ● Fine-tuning     │     │   Analysis        │
                          │                   │     │                   │     │                   │
                          └───────────────────┘     └───────────────────┘     └───────────────────┘
```

1. **Hardware Profiling**: Rilevamento delle risorse disponibili e selezione automatica della configurazione ottimale.
2. **Synthetic Pretraining**: Training accelerato su ambiente sintetico per apprendere i comportamenti fondamentali.
3. **Full Training**: Training sull'ambiente completo con tuning fine degli iperparametri.
4. **Evaluation & Analysis**: Valutazione statistica delle performance e generazione di report dettagliati.

### Pipeline di Inferenza

```
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ 1. Model          │────►│ 2. Environment    │────►│ 3. Action         │
│    Loading        │     │    Observation    │     │    Selection      │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
│                   │     │                   │     │                   │
│ ● Model           │     │ ● State           │     │ ● Q-Value         │
│   Deserialization │     │   Encoding        │     │   Computation     │
│ ● Hardware        │     │ ● Feature         │     │ ● Deterministic   │
│   Optimization    │     │   Extraction      │     │   Policy          │
│                   │     │                   │     │                   │
└───────────────────┘     └───────────────────┘     └───────────────────┘
                                                            │
                                                            ▼
                                                  ┌───────────────────┐
                                                  │                   │
                                                  │ 4. Environment    │
                                                  │    Update         │
                                                  │                   │
                                                  └───────────────────┘
                                                            │
                                                            ▼
                                                  ┌───────────────────┐
                                                  │                   │
                                                  │ ● State           │
                                                  │   Transition      │
                                                  │ ● Reward          │
                                                  │   Computation     │
                                                  │                   │
                                                  └───────────────────┘
``` 

## 🧠 Fondamenti Algoritmici DQN

Il sistema implementa diverse varianti avanzate dell'algoritmo Deep Q-Network (DQN), con estensioni che migliorano significativamente le performance di apprendimento.

### Formulazione Matematica Base

L'algoritmo DQN si basa sull'approssimazione della funzione di valore ottimale Q*(s,a) attraverso reti neurali parametrizzate:

```
Q(s, a; θ) ≈ Q*(s, a)
```

dove Q* rappresenta la funzione di valore ottimale che mappa coppie stato-azione ai loro valori attesi, e θ sono i parametri della rete neurale.

L'obiettivo dell'apprendimento è minimizzare la discrepanza tra i valori Q predetti e i target calcolati secondo l'equazione di Bellman:

```
L(θ) = 𝔼[(r + γ·max_a' Q(s', a'; θ') - Q(s, a; θ))²]
```

dove:
- r: ricompensa immediata
- γ: fattore di sconto temporale (0 < γ ≤ 1)
- max_a' Q(s', a'; θ'): massimo valore Q stimato per lo stato futuro s'
- θ': parametri della rete target, periodicamente sincronizzati con θ

### Architettura Dueling DQN

La nostra implementazione avanzata utilizza l'architettura Dueling DQN, che decompone la funzione Q in due componenti:

```
Q(s, a; θ, α, β) = V(s; θ, β) + (A(s, a; θ, α) - 1/|A|·∑_a' A(s, a'; θ, α))
```

dove:
- V(s; θ, β): rappresenta il valore dello stato s indipendentemente dall'azione scelta
- A(s, a; θ, α): rappresenta l'advantage dell'azione a rispetto alle altre azioni in s
- Il termine di sottrazione implementa un aggregatore statisticamente robusto

Questa decomposizione permette all'agente di valutare gli stati senza dover stimare l'effetto di ogni azione, migliorando significativamente l'efficienza dell'apprendimento.

### Double DQN

Per contrastare il bias di sovrastima tipico dell'algoritmo Q-learning, implementiamo Double DQN:

```
y_t = r + γ·Q(s', argmax_a' Q(s', a'; θ); θ')
```

Questo approccio disaccoppia la selezione dell'azione dalla sua valutazione:
- La rete di policy (θ) seleziona l'azione con argmax_a'
- La rete target (θ') ne valuta il valore Q

### Prioritized Experience Replay

Implementiamo un buffer di replay prioritarizzato che campiona transizioni con probabilità proporzionale al loro TD-error:

```
p(i) = (|δ_i| + ε)^α / ∑_j (|δ_j| + ε)^α
```

dove:
- δ_i = r + γ·max_a' Q(s', a'; θ') - Q(s, a; θ): TD-error per la transizione i
- ε: piccola costante per evitare probabilità zero
- α: parametro di prioritizzazione (α = 0 corrisponde a sampling uniforme)

Per compensare il bias introdotto dal sampling non uniforme, utilizziamo importance sampling con weight:

```
w_i = (N·p(i))^(-β)
```

dove β è anch'esso annealed durante il training da un valore iniziale fino a 1.

## 📊 Ottimizzazione Hardware-Aware

Il sistema implementa un'architettura di esecuzione adattiva che si configura automaticamente in base all'hardware disponibile.

### CPU Vectorization

Su sistemi CPU-only, implementiamo:

- **SIMD Operations**: Utilizzo estensivo di NumPy con operazioni vettoriali ottimizzate
- **Multi-processing**: Distribuzione del carico su tutti i core disponibili utilizzando moduli come `multiprocessing` e `Ray`
- **Batch Processing**: Aggregazione delle operazioni in batch per massimizzare throughput e minimizzare overhead

### GPU Acceleration

Su sistemi con GPU CUDA o ROCm:

- **Automatic Mixed Precision**: Training in precisione FP16/BF16 con dynamic loss scaling
- **Asynchronous Data Transfer**: Pipeline ottimizzata per concurrency tra CPU e GPU
- **Custom CUDA Kernels**: Implementazione di kernels specializzati per operazioni critiche
- **Tensor Core Utilization**: Ottimizzazione per hardware specifico (NVIDIA Tensor Cores)

### Distributed Training

Per training su cluster multi-node:

- **Parameter Server Architecture**: Sincronizzazione efficiente dei parametri tra worker
- **Gradient Accumulation**: Gestione ottimale della memoria per modelli di grandi dimensioni
- **Sharded Data Parallelism**: Distribuzione del processing e della memoria su più device

## 🔌 Interfacce e Estensibilità

Il sistema è progettato per essere estensibile attraverso interfacce ben definite:

### Estensione Algoritmi

Per implementare nuovi algoritmi di RL, è sufficiente:

```python
class CustomRLAlgorithm(BaseRLAlgorithm):
    def __init__(self, config):
        super().__init__(config)
        # Inizializzazione specifica
        
    def select_action(self, state, epsilon=0.0):
        # Implementazione della policy
        
    def update(self, batch):
        # Implementazione dell'algoritmo di apprendimento
        
    def save(self, path):
        # Serializzazione del modello
```

### Architetture Neurali Custom

Nuove architetture possono essere integrate estendendo il model factory:

```python
class CustomNeuralArchitecture(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        # Definizione della rete
        
    def forward(self, x):
        # Forward pass
        
# Registrazione dell'architettura
register_architecture("custom", CustomNeuralArchitecture)
```

### Ambienti Personalizzati

Nuovi ambienti possono essere integrati implementando l'interfaccia Gymnasium:

```python
class CustomEnvironment(gym.Env):
    def __init__(self, config):
        super().__init__()
        self.observation_space = spaces.Box(...)
        self.action_space = spaces.Discrete(...)
        
    def reset(self):
        # Reset environment
        
    def step(self, action):
        # Execute action and return (next_state, reward, done, info)
```

## 📝 Principi di Progettazione

Il sistema è stato progettato seguendo principi avanzati di software engineering:

- **SOLID Principles**: Single Responsibility, Open-Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- **Design Patterns**: Factory Method, Strategy, Observer, Command, Adapter, Facade
- **Clean Architecture**: Separazione delle concerns con dipendenze unidirezionali
- **Test-Driven Development**: Test unitari, funzionali e di integrazione con coverage elevato
- **Continuous Integration**: Pipeline CI/CD per garantire affidabilità del codice

---

Questo documento fornisce una panoramica dell'architettura del sistema. Per implementazioni specifiche, consultare la [Guida al Codice](CODE_GUIDE.md) e i commenti nel codice sorgente. 