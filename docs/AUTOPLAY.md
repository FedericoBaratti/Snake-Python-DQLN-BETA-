# 🤖 Runtime di Inferenza e Deployment in Produzione

Questo documento tecnico fornisce le specifiche complete per il deployment in produzione, il runtime di inferenza e l'analisi delle performance degli agenti DQN addestrati.

## 📋 Architettura del Sistema di Inferenza

Il sistema di inferenza è composto da componenti specializzati progettati per massimizzare l'efficienza durante l'esecuzione degli agenti addestrati:

```
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│                           │     │                           │     │                           │
│    Model Serialization    │────►│  Inference Optimization   │────►│  Runtime Environment      │
│    & Deserialization      │     │  & Acceleration           │     │  Integration              │
│                           │     │                           │     │                           │
└───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
                                                                                  │
                                                                                  ▼
┌───────────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│                           │     │                           │     │                           │
│   Performance Analysis    │◄────│    Decision Boundary      │◄────│   Execution Pipeline      │
│   & Benchmarking          │     │    Visualization          │     │   & Monitoring            │
│                           │     │                           │     │                           │
└───────────────────────────┘     └───────────────────────────┘     └───────────────────────────┘
```

Ogni componente è ottimizzato per execution efficiency e low-latency decision making.

## 🚀 Specifiche del Runtime di Inferenza

### Quick Deployment Guide

Per avviare immediatamente l'inferenza con un agente preaddestrato:

```bash
# Inferenza con agente base (adatto per CPU-only)
python main.py --mode autoplay --model base

# Inferenza con agente complesso e visualizzazione avanzata
python main.py --mode autoplay --model complesso --visualization full --grid-size 20

# Inferenza con agente perfetto ottimizzato per GPU
python main.py --mode autoplay --model perfetto --device cuda --optimization full
```

### Acceleration & Optimization Parameters

Il sistema supporta diverse ottimizzazioni per l'inferenza:

| Parametro | Descrizione | Valori | Default |
|-----------|-------------|--------|---------|
| `--device` | Dispositivo di computing per l'inferenza | cpu, cuda, xla | auto |
| `--optimization` | Livello di ottimizzazione per l'inferenza | none, basic, medium, full | medium |
| `--batch-inference` | Abilita inferenza batch per simulazioni multiple | true, false | false |
| `--quantization` | Tipo di quantizzazione per modelli | none, dynamic, static | none |
| `--jit-compile` | Compila il modello con TorchScript | true, false | true |
| `--visualization` | Modalità di visualizzazione | none, basic, decision, full | basic |
| `--fps` | Frame rate target per la visualizzazione | 1-120 | 30 |
| `--telemetry` | Livello di telemetria durante l'esecuzione | none, basic, full | basic |

### Model Loading & Optimization

Il sistema supporta tecniche avanzate di model loading e ottimizzazione per l'inferenza:

```python
def load_optimized_model(checkpoint_path, device, optimization_level):
    """Carica e ottimizza un modello per inferenza ad alte prestazioni."""
    # Carica il checkpoint con metadata
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Ricostruisci il modello dalla configurazione salvata
    model_config = checkpoint['config']
    state_dim = model_config['state_dim']
    action_dim = model_config['action_dim']
    
    # Crea l'architettura appropriata
    model = create_model_architecture(
        model_config['architecture'], 
        state_dim, 
        action_dim, 
        model_config['hidden_layers']
    )
    
    # Carica i pesi del modello
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Imposta in modalità inferenza
    
    # Applica ottimizzazioni in base al livello specificato
    if optimization_level >= OptimizationLevel.BASIC:
        # Fusion di operazioni per ridurre overhead
        model = optimize_for_inference(model)
        
    if optimization_level >= OptimizationLevel.MEDIUM:
        # Quantizzazione per ridurre memoria e aumentare throughput
        if device == 'cpu':
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
    
    if optimization_level >= OptimizationLevel.FULL:
        # Compilazione JIT per ottimizzazioni a livello di grafo
        dummy_input = torch.zeros((1, state_dim), device=device)
        model = torch.jit.trace(model, dummy_input)
        model = torch.jit.optimize_for_inference(model)
    
    return model, model_config
```

## 🏎️ Modalità Rapide di Deployment

Per verificare rapidamente i risultati dell'addestramento o eseguire dimostrazioni, il sistema offre pipeline accelerate:

### Modalità Demo

```bash
python main.py --demo
```

Questa pipeline esegue:

1. **Initialization Ottimizzata**:
   - Caricamento efficiente delle dipendenze con lazy loading
   - Configurazione automatica in base all'hardware rilevato
   - Preallocation delle risorse di memoria critica

2. **Fast Training**:
   - Training accelerato con 300 episodi su modello base
   - Segnali di reward engineered per convergenza rapida
   - Checkpointing progressivo con ripristino del miglior modello

3. **Optimized Deployment**:
   - Model pruning per ridurre dimensioni e latency
   - Post-training quantization per aumentare throughput
   - Inizializzazione diretta del runtime di inferenza

4. **Performance Visualization**:
   - Rendering hardware-accelerato per visualizzazione fluida
   - Overlay in tempo reale delle statistiche di performance
   - Replay configurabile delle sessioni di gioco più significative

### Train-and-Play Platform

```bash
# Configurazione bilanciata per training e deployment immediato
python main.py --mode train-and-play --model avanzato --episodes 2000

# Configurazione avanzata con metriche estese
python main.py --mode train-and-play --model complesso --episodes 5000 --telemetry full
```

Questa modalità implementa un workflow end-to-end che:

1. **Esegue il training completo** con la configurazione specificata
2. **Ottimizza automaticamente il modello** per l'inferenza
3. **Inizializza il runtime di autoplay** con il modello appena addestrato
4. **Genera report dettagliati** sulle performance

## ⚙️ Configurazione Avanzata del Runtime

Il sistema di autoplay supporta configurazioni granulari per ottimizzare diversi aspetti dell'esperienza.

### Matrix of Runtime Parameters

| Parametro | Descrizione | Valori | Default | Impatto |
|-----------|-------------|--------|---------|---------|
| `--model` | Complessità dell'architettura | base, avanzato, complesso, perfetto | base | Determina la capacità di pianificazione |
| `--checkpoint` | Percorso specifico del modello | path | auto | Permette di selezionare modelli specifici |
| `--grid-size` | Dimensione della griglia di gioco | 5-100 | 10 | Influisce sulla complessità dell'ambiente |
| `--speed` | Velocità di esecuzione (FPS) | 1-120 | 10 | Controllabilità vs fluidità |
| `--render-mode` | Modalità di rendering | pixel, vector, hybrid | vector | Performance vs qualità visiva |
| `--games` | Numero di partite consecutive | 1-∞ | ∞ | Utile per benchmark automatizzati |
| `--max-steps` | Steps massimi per partita | 10-100000 | 1000 | Previene loop infiniti | 
| `--seed` | Seed per inizializzazione | int | random | Riproducibilità dei risultati |
| `--state-representation` | Formato dell'input al modello | grid, features, vision | grid | Diversi encoding dello stato |
| `--parallel-envs` | Ambienti paralleli per inferenza | 1-64 | 1 | Throughput vs latenza |

### Customized Execution Profiles

Il sistema supporta profili di esecuzione personalizzati per ottimizzare diversi scenari d'uso:

```python
# Profili predefiniti per casi d'uso specifici
EXECUTION_PROFILES = {
    "benchmark": {
        "visualization": "none",
        "games": 100,
        "parallel_envs": 16,
        "telemetry": "full",
        "save_stats": True,
        "optimization": "full"
    },
    "demo": {
        "visualization": "full",
        "render_mode": "vector",
        "speed": 8,
        "telemetry": "basic",
        "optimization": "medium"
    },
    "analysis": {
        "visualization": "decision",
        "speed": 5,
        "telemetry": "full",
        "record_trajectory": True,
        "save_stats": True
    },
    "performance": {
        "visualization": "basic",
        "optimization": "full",
        "jit_compile": True,
        "quantization": "dynamic",
        "parallel_envs": 4
    }
}

# Utilizzo dei profili
python main.py --mode autoplay --model complesso --profile benchmark
```

## 🔍 Real-time Inference Monitoring

Durante l'esecuzione dell'autoplay, il sistema fornisce monitoraggio avanzato delle performance in tempo reale:

### Telemetry Core

```python
class InferenceTelemetry:
    """Sistema di telemetria avanzato per il monitoraggio delle performance di inferenza."""
    
    def __init__(self, level="basic"):
        self.level = level
        self.inference_times = deque(maxlen=1000)  # Ultimi 1000 tempi di inferenza
        self.rewards = []  # Ricompense per episodio
        self.scores = []   # Punteggi finali per episodio
        self.steps = []    # Passi per episodio
        self.q_values = [] # Statistiche dei Q-values
        self.actions = []  # Azioni selezionate
        
        # Metriche in tempo reale
        self.current_episode = 0
        self.current_step = 0
        self.current_score = 0
        self.current_reward = 0
        self.episode_start_time = None
        self.total_start_time = time.time()
        
        # Metriche avanzate
        if level == "full":
            self.action_heatmap = np.zeros((4,))  # Distribuzione delle azioni
            self.position_heatmap = None  # Inizializzato dopo conoscere grid_size
            self.decision_confidence = []  # Differenza tra Q-value massimo e medio
            self.exploration_metrics = []  # Entropia della distribuzione dei Q-values
        
    def start_episode(self):
        """Inizia il tracciamento di un nuovo episodio."""
        self.current_episode += 1
        self.current_step = 0
        self.current_score = 0
        self.current_reward = 0
        self.episode_start_time = time.time()
        
    def record_step(self, state, action, reward, next_state, done, q_values=None, inference_time=None):
        """Registra informazioni su un singolo step."""
        self.current_step += 1
        self.current_reward += reward
        
        if inference_time:
            self.inference_times.append(inference_time)
        
        # Registra Q-values se disponibili e telemetria è full
        if self.level == "full" and q_values is not None:
            self.q_values.append(q_values)
            self.actions.append(action)
            
            # Calcola confidenza della decisione
            q_max = np.max(q_values)
            q_mean = np.mean(q_values)
            q_std = np.std(q_values)
            self.decision_confidence.append(q_max - q_mean)
            
            # Calcola entropia
            q_softmax = softmax(q_values)
            entropy = -np.sum(q_softmax * np.log(q_softmax + 1e-10))
            self.exploration_metrics.append(entropy)
            
            # Aggiorna heatmaps
            self.action_heatmap[action] += 1
            if self.position_heatmap is not None:
                head_pos = get_snake_head(state)
                self.position_heatmap[head_pos[0], head_pos[1]] += 1
        
        # Se l'episodio è terminato, registra le statistiche finali
        if done:
            self.scores.append(self.current_score)
            self.rewards.append(self.current_reward)
            self.steps.append(self.current_step)
            episode_time = time.time() - self.episode_start_time
            
            # Log delle statistiche dell'episodio
            if self.level != "none":
                logger.info(f"Episodio {self.current_episode}: Score={self.current_score}, "
                           f"Steps={self.current_step}, Time={episode_time:.2f}s")
    
    def get_stats(self):
        """Restituisce statistiche complete dell'esecuzione."""
        stats = {
            "episodes_completed": self.current_episode,
            "total_steps": sum(self.steps),
            "avg_score": np.mean(self.scores) if self.scores else 0,
            "max_score": max(self.scores) if self.scores else 0,
            "avg_reward": np.mean(self.rewards) if self.rewards else 0,
            "avg_steps": np.mean(self.steps) if self.steps else 0,
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0,
            "total_runtime": time.time() - self.total_start_time,
        }
        
        if self.level == "full":
            # Statistiche avanzate
            stats.update({
                "action_distribution": self.action_heatmap / (np.sum(self.action_heatmap) + 1e-10),
                "avg_decision_confidence": np.mean(self.decision_confidence) if self.decision_confidence else 0,
                "avg_exploration": np.mean(self.exploration_metrics) if self.exploration_metrics else 0,
                "score_percentiles": {
                    "25th": np.percentile(self.scores, 25) if len(self.scores) >= 4 else 0,
                    "50th": np.percentile(self.scores, 50) if len(self.scores) >= 2 else 0,
                    "75th": np.percentile(self.scores, 75) if len(self.scores) >= 4 else 0,
                    "95th": np.percentile(self.scores, 95) if len(self.scores) >= 20 else 0,
                }
            })
        
        return stats
```

### Real-time Visualization

Durante l'esecuzione in modalità `--visualization full`, il sistema renderizza:

1. **Stato del gioco**: Visualizzazione ad alta fedeltà degli elementi di gioco
2. **Q-Value Heatmap**: Rappresentazione visiva dei valori Q per ogni azione possibile
3. **Confidence Metrics**: Indicatori di quanto l'agente è sicuro delle sue decisioni
4. **Trajectory Analysis**: Tracciamento del percorso dell'agente con highlight delle decisioni chiave
5. **Performance Dashboard**: Metriche real-time delle prestazioni dell'agente

## 🕹️ Human-Agent Interaction

Il sistema supporta interazione avanzata durante l'autoplay:

| Tasto | Azione | Descrizione Tecnica |
|-------|--------|---------------------|
| P | Pause/Resume | Sospende l'esecuzione mantenendo lo stato interno dell'agente |
| + | Increase Speed | Incrementa il target framerate con step di 5 FPS |
| - | Decrease Speed | Decrementa il target framerate con step di 5 FPS |
| R | Reset | Reinizializza l'ambiente preservando lo stato dell'agente |
| S | Show Statistics | Genera report statistico completo dell'esecuzione corrente |
| V | Toggle Visualization | Cicla tra le diverse modalità di visualizzazione |
| Q | Quit | Termina l'esecuzione salvando checkpoint e statistiche |
| T | Take Control | Transizione dinamica tra controllo AI e umano |
| D | Debug View | Attiva/disattiva visualizzazione debug avanzata |
| N | Network View | Visualizza rappresentazione grafica della rete neurale |
| B | Benchmark Mode | Esegue benchmark di performance su 100 episodi |
| E | Export Data | Esporta metriche e traiettorie in formato serializzato |

### Take Control Mode

La modalità "Take Control" permette il passaggio fluido tra controllo IA e umano:

```python
def handle_take_control(self):
    """Gestisce la transizione tra controllo AI e umano."""
    self.human_control = not self.human_control
    
    if self.human_control:
        # Preserva lo stato corrente dell'agente
        self._saved_agent_state = {
            "epsilon": self.agent.epsilon,
            "state": self.current_state.copy()
        }
        logger.info("Controllo passato all'utente. L'IA è in modalità osservazione.")
        
        # Attiva raccolta dati per imitation learning (opzionale)
        if self.config.collect_human_demonstrations:
            self.demonstration_buffer = []
            self.collecting_demonstrations = True
    else:
        # Ripristina controllo all'agente
        logger.info("Controllo ripristinato all'IA.")
        
        # Termina raccolta dati e opzionalmente esegui fine-tuning
        if self.collecting_demonstrations and self.demonstration_buffer:
            logger.info(f"Raccolte {len(self.demonstration_buffer)} dimostrazioni umane.")
            if self.config.finetune_from_demonstrations and len(self.demonstration_buffer) > 10:
                self._finetune_from_demonstrations()
            self.collecting_demonstrations = False
```

## 📊 Analisi Comparativa dei Modelli

Il sistema supporta benchmark comparativi tra diverse architetture per valutarne l'efficacia e l'efficienza:

```bash
# Esegui benchmark comparativo tra tutti i modelli disponibili
python -m autoplay.compare --models base avanzato complesso perfetto --episodes 50 --grid-size 20
```

### Metriche di Confronto

| Modello   | Score Medio | Max Score | Passi/Episodio | Efficienza | Inferenza(ms) |
|-----------|-------------|-----------|----------------|------------|---------------|
| Base      | 35-70       | 75-100    | 300-600        | 0.10-0.15  | 0.2-0.5       |
| Avanzato  | 70-120      | 150-180   | 550-950        | 0.15-0.22  | 0.5-1.0       |
| Complesso | 120-200     | 220-320   | 900-1500       | 0.22-0.30  | 1.0-2.0       |
| Perfetto  | 200-350+    | 350-500+  | 1500-3000+     | 0.30-0.40+ | 2.0-4.0       |

*L'efficienza è calcolata come score/passi ed esprime quanto efficacemente l'agente raccoglie cibo.
*Il tempo di inferenza è misurato su CPU standard; su GPU è tipicamente 5-10x inferiore.

### Caratteristiche Qualitative dei Modelli

#### Base
- Implementa strategie elementari di ricerca del cibo
- Evita ostacoli immediati ma ha visione limitata
- Occasionalmente entra in loop o si intrappola

#### Avanzato
- Mostra pianificazione a breve termine
- Gestisce sezioni del corpo come ostacoli dinamici
- Evita la maggior parte delle trappole semplici

#### Complesso
- Dimostra pianificazione spaziale avanzata
- Utilizza strategie di gestione dello spazio
- Naviga efficacemente in situazioni complesse

#### Perfetto
- Implementa strategie quasi ottimali
- Manifesta comportamenti emergenti di auto-preservazione
- Massimizza l'utilizzo dello spazio con percorsi ottimizzati

## 🛠️ Troubleshooting e Performance Optimization

### Common Issues & Solutions

#### Problema: L'agente si blocca in pattern ciclici

**Cause potenziali**:
- Policy subottimale per situazioni specifiche
- Exploration insufficiente durante il training
- Overfitting su stati specifici

**Soluzioni**:
```bash
# Utilizzo di un modello più complesso
python main.py --mode autoplay --model complesso

# Introduzione di rumore stocastico durante l'inferenza
python main.py --mode autoplay --model avanzato --epsilon 0.05

# Riaddestramento con maggiore diversità di stati
python main.py --mode train --model avanzato --episodes 5000 --augmentation true
```

#### Problema: Performance inconsistenti tra esecuzioni

**Cause potenziali**:
- Alta varianza nella policy appresa
- Initialization stocastica dell'ambiente
- Instabilità nella rete neurale

**Soluzioni**:
```bash
# Fissaggio del seed per riproducibilità
python main.py --mode autoplay --seed 42

# Utilizzo della policy deterministica
python main.py --mode autoplay --deterministic true

# Ensemble di modelli per decisioni più robuste
python main.py --mode autoplay --ensemble 3
```

### Performance Optimization

Per ottimizzare le performance di inferenza:

```bash
# Ottimizzazione massima per CPU
python main.py --mode autoplay --optimization full --device cpu --quantization dynamic

# Ottimizzazione per GPU
python main.py --mode autoplay --optimization full --device cuda --batch-inference true

# Bilanciamento performance/visualizzazione
python main.py --mode autoplay --optimization medium --visualization basic --fps 30
```

## 🔬 Use Cases Avanzati

### Decision-Making Analysis

```bash
# Analisi dettagliata del processo decisionale
python main.py --mode autoplay --visualization decision --telemetry full --record-trajectory
```

Questa configurazione:
1. Visualizza in tempo reale i Q-values per ogni azione
2. Genera heatmap di attivazione neurale
3. Registra la traiettoria completa per analisi post-execution
4. Calcola metriche di entropia decisionale

### Resilience Testing

```bash
# Stress test dell'agente in condizioni avverse
python main.py --mode autoplay --adversarial true --obstacle-frequency 0.3
```

Questa modalità:
1. Introduce dinamicamente ostacoli aggiuntivi
2. Modifica la posizione del cibo in modo avverso
3. Testa la robustezza della policy a perturbazioni
4. Valuta la capacità di recupero da situazioni difficili

### Continuous Learning

```bash
# Apprendimento continuo dall'esperienza
python main.py --mode autoplay --continuous-learning true --update-frequency 100
```

Questa configurazione:
1. Continua ad aggiornare il modello durante l'inferenza
2. Migliora la policy basandosi sulle nuove esperienze
3. Adatta l'agente a scenari non visti durante il training
4. Implementa meccanismi di lifelong learning

---

Per una comprensione approfondita dell'architettura sottostante e dei meccanismi di implementazione, consultare la [Guida al Codice](CODE_GUIDE.md). 