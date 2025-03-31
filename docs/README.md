# 🐍 Snake con UI e Deep Q-Learning (DQN)

Benvenuti nell'ecosistema **Snake con UI e Deep Q-Learning**, un framework avanzato per la sperimentazione di algoritmi di Reinforcement Learning applicati al classico gioco Snake. Questo progetto implementa un ambiente completamente parametrizzabile con architetture DQN ottimizzate e un'interfaccia grafica ad alte prestazioni.

## 📋 Panoramica Architetturale

Il framework è costruito su un'architettura a microservizi con disaccoppiamento completo tra componenti:

1. 🎮 **Frontend modulare** con rendering ottimizzato e pipeline di visualizzazione accelerata
2. 🤖 **Backend multi-threaded** con simulazione fisica paralelizzabile e ottimizzata per calcolo vettoriale
3. 🧠 **Runtime DQN completamente parametrizzabile** con supporto per architetture neurali avanzate
4. 🔄 **Sistema di preaddestramento sintetico** con generazione euristica di scenari ottimali
5. 🛠️ **Training engine distribuito** con supporto per multi-GPU e accelerazione hardware specializzata
6. 📊 **Telemetria avanzata** con visualizzazione in tempo reale delle metriche di apprendimento

## 🚀 Architettura Tecnica

- **Interfaccia grafica vettoriale** con rendering hardware-accelerato tramite Pygame
- **Kernel di simulazione ottimizzato** con operazioni a bassa latenza
- **Framework DQN avanzato** con implementazioni di:
  - Dueling DQN con separazione valore-vantaggio
  - Double DQN per eliminazione bias di sovrastima
  - Prioritized Experience Replay con sampling bilanciato
  - N-step returns per propagazione efficiente del segnale di reward
- **Preaddestramento eurisitico** con sintesi di mappe di reward ad alta densità
- **Ottimizzazione hardware-aware** con decomposizione automatica per CPU multi-core, GPU e TPU
- **Sistema di checkpoint avanzato** con ripristino del training da stati arbitrari
- **Monitoring real-time** con integrazione TensorBoard e serializzazione delle metriche

## 🔧 Requisiti Tecnici

- **Runtime**: Python 3.11.9+ con supporto JIT
- **Compute**: Architettura x86-64 o ARM64 con estensioni SIMD
- **Memoria**: 4GB baseline, 8GB+ per modelli avanzati
- **GPU**: CUDA 11.8+ o ROCm 5.0+ (opzionale ma fortemente consigliato)

## 📊 Specifiche dei Modelli DQN

Il framework implementa una suite di architetture neurali progressive, ciascuna ottimizzata per diversi scenari di utilizzo:

| Architettura | Topologia | Parametri | Prestazioni | Requisiti Computazionali |
|--------------|-----------|-----------|-------------|--------------------------|
| Base         | FC[64-32] con ReLU | 3,074 | Strategia elementare, cattura 15-30 unità | CPU single-core, ~50MB VRAM |
| Avanzato     | FC[128-64-32] con ReLU e Dropout | 12,322 | Pianificazione a breve termine, cattura 30-100 unità | 2+ core CPU, 1GB+ VRAM |
| Complesso    | Dueling[256-128-64-32] con Batch Norm | 41,986 | Pianificazione spaziale avanzata, cattura 100-200 unità | 4+ core CPU, 2GB+ VRAM |
| Perfetto     | Dueling+Attention[512-256-128-64] con Layer Norm | 172,546 | Pianificazione quasi-ottimale, cattura 200-400+ unità | 8+ core CPU, 4GB+ VRAM o GPU dedicata |

## 🎯 Capacità del Sistema

1. **Simulazione ad alta fedeltà** del classico gioco Snake con fisica deterministicamente riproducibile
2. **Training supervisionato** con architetture neurali avanzate scalabili fino a centinaia di migliaia di parametri
3. **Ottimizzazione fine-grained** con modulazione automatica degli iperparametri durante il training
4. **Orchestrazione intelligente delle risorse** con bilanciamento dinamico tra CPU e GPU
5. **Visualizzazione avanzata delle decision boundary** durante l'esecuzione dell'agente

## 📌 Getting Started

Per un deployment e configurazione ottimali, consultare:

- [Guida all'Installazione](INSTALLATION.md) - Configurazione dell'ambiente di runtime e deployment delle dipendenze
- [Architettura del Sistema](ARCHITECTURE.md) - Specifiche architetturali complete e diagrammi dei componenti
- [Manuale di Training](TRAINING.md) - Protocolli di addestramento e ottimizzazione degli iperparametri
- [Documentazione Autoplay](AUTOPLAY.md) - Configurazione dell'agente in modalità inferenza
- [Specifiche del Codice](CODE_GUIDE.md) - Documentazione API e specifiche di implementazione

## 🔍 Struttura del Repository

```
snake-rl-project/
├── frontend/              # Interfaccia grafica vettoriale accelerata
│   ├── ui.py              # Pipeline di rendering
│   ├── input_handler.py   # Gestione eventi asincrona
│   └── renderer.py        # Engine grafico hardware-accelerato
├── backend/               # Core engine e runtime di simulazione
│   ├── snake_game.py      # Kernel di simulazione vettoriale 
│   ├── environment.py     # Interfaccia Gymnasium con observation encoding
│   └── utils.py           # Utilità di sistema e rilevamento hardware
├── dqn_agent/             # Framework DQN avanzato
│   ├── models.py          # Implementazioni neurali parametriche
│   ├── dqn_agent.py       # Algoritmi di apprendimento ottimizzati
│   └── config.py          # Configurazioni architetturali
├── pretraining/           # Sistema di preaddestramento euristico
│   ├── synthetic_env.py   # Generatore di scenari sintetici
│   └── pretrain.py        # Pipeline di addestramento accelerato
├── training/              # Engine di training distribuito
│   ├── train.py           # Orchestratore di addestramento
│   └── checkpoints/       # Repository di modelli serializzati
├── autoplay/              # Runtime di inferenza
│   └── autoplay.py        # Controller di esecuzione autonoma
├── docs/                  # Documentazione tecnica completa
├── requirements.txt       # Specifiche delle dipendenze
└── main.py                # Entrypoint parametrico
```

## 🎮 Utilizzo

```bash
# Installazione ottimizzata con compilazione dei moduli nativi
pip install -r requirements.txt

# Modalità manuale con interfaccia vettoriale
python main.py --mode manual --grid-size 20 --render-mode vector

# Inferenza con modello avanzato preaddestrato e visualizzazione delle decision boundary
python main.py --mode autoplay --model avanzato --checkpoint training/checkpoints/dqn_avanzato_v3.2.pt --visualization full

# Training distribuito con ottimizzazione automatica degli iperparametri
python main.py --mode train --model complesso --episodes 10000 --grid-size 20 --workers auto --auto-tune

# Preaddestramento euristico seguito da training completo e deployment in inferenza
python main.py --mode train-and-play --model perfetto --pretrain-steps 2000000 --episodes 5000 --grid-size 30
```

## 📈 Ottimizzazione Hardware-Aware

Il framework implementa ottimizzazione intelligente a runtime:

- **CPU Only**: Vectorized NumPy operations con parallelizzazione multi-processo
- **CUDA Acceleration**: Kernel PyTorch ottimizzati con trasferimento zero-copy
- **Multi-GPU**: Distribuzione automatica con torch.nn.DataParallel per single-node e DistributedDataParallel per multi-node
- **TPU Support**: Compatibilità sperimentale con PyTorch/XLA per esecuzione su hardware specializzato
- **Mixed Precision**: Training automatico in FP16/BF16 su hardware compatibile con opportuno dynamic loss scaling

## 🧪 Stack Tecnologico

- **Python 3.11.9**: Runtime principale con ottimizzazione JIT
- **PyTorch 2.2.0**: Framework deep learning con supporto CUDA e ROCm
- **Pygame 2.5.2**: Engine grafico con accelerazione hardware
- **Gymnasium 0.29.1**: Framework standardizzato per ambienti di reinforcement learning
- **NumPy 1.26.0**: Computazione numerica vettoriale ad alta performance
- **Ray 2.9.0**: Framework distribuito per training parallelo
- **TensorBoard 2.15.1**: Visualizzazione real-time delle metriche di training

## 📄 Licenza

Questo progetto è distribuito sotto licenza MIT. Per dettagli completi, consultare il file LICENSE.

## 🤝 Contributi

I contributi sono benvenuti secondo il protocollo GitFlow. Prima di contribuire, consultare [CONTRIBUTING.md](CONTRIBUTING.md) per le linee guida tecniche.

## 📞 Contatti

Per segnalazioni tecniche o richieste di feature, utilizzare il sistema di issue tracking o contattare gli autori tramite i canali ufficiali.

---

Grazie per l'interesse nel nostro framework di reinforcement learning! 🚀 

# 📋 Documentazione Snake con DQN - Versione 2.0

## 🆕 Aggiornamenti Principali della Versione 2.0

Questa versione introduce significativi miglioramenti all'interfaccia utente e nuove funzionalità per l'interazione con i modelli preaddestrati:

1. **Interfaccia di Selezione Modelli**
   - Finestra modale per visualizzare tutti i modelli disponibili
   - Navigazione tramite tastiera e selezione intuitiva
   - Caricamento dinamico del modello selezionato senza riavvio

2. **Caricamento Dinamico dei Modelli**
   - Rilevamento automatico della complessità del modello dal nome del file
   - Reinizializzazione del controller autoplay in tempo reale
   - Sistema di fallback per gestione errori durante il caricamento

3. **UI Migliorata**
   - Overlay semi-trasparente per migliore leggibilità
   - Feedback visivo sulla selezione corrente
   - Visualizzazione ottimizzata con scroll per liste lunghe

4. **Nuovi Comandi**
   - Tasto `M` per aprire/chiudere la finestra di selezione modelli
   - Navigazione con frecce ↑/↓ nella lista modelli
   - Tasto `Invio` per caricare il modello selezionato

5. **Opzioni da Linea di Comando**
   - Nuovo parametro `--select-model` per avviare direttamente con la finestra di selezione
   - Integrazione completa con i parametri esistenti

## 📄 File Modificati

- `frontend/ui.py` - Aggiunta interfaccia di selezione modelli
- `main.py` - Aggiunto supporto per il parametro `--select-model`
- `MANUALE_UTENTE.md` - Aggiornato con istruzioni dettagliate
- `README.md` - Aggiornato con informazioni sulla nuova versione
- `docs/AUTOPLAY.md` - Aggiunta documentazione tecnica per la selezione modelli

## 🖥️ Nuove Funzionalità Implementate

### 1. Scansione Automatica dei Checkpoint

```python
def refresh_checkpoint_list(self):
    """Aggiorna la lista dei checkpoint disponibili nel sistema."""
    checkpoint_dir = os.path.join("training", "checkpoints")
    if os.path.exists(checkpoint_dir):
        # Trova tutti i file .pt (checkpoint PyTorch)
        self.available_checkpoints = sorted(
            [f for f in glob.glob(os.path.join(checkpoint_dir, "*.pt")) 
             if "_memory" not in f],  # Esclude i file di memoria
            key=os.path.getmtime,  # Ordina per data di modifica
            reverse=True  # Più recenti prima
        )
```

### 2. Interfaccia di Selezione Visuale

```python
def draw_model_selector(self):
    """Disegna la finestra di selezione del modello."""
    if not self.show_model_selector:
        return
    
    # Dimensioni della finestra di selezione
    selector_width = min(self.window_width - 100, 600)
    selector_height = min(self.window_height - 100, 400)
    
    # Posizione centrale della finestra
    selector_x = (self.window_width - selector_width) // 2
    selector_y = (self.window_height - selector_height) // 2
    
    # Disegna lo sfondo semi-trasparente
    overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 180))  # Nero semi-trasparente
    self.window.blit(overlay, (0, 0))
    
    # [...] Rendering del contenuto
```

### 3. Caricamento Dinamico del Modello

```python
def load_selected_model(self):
    """Carica il modello selezionato nell'autoplay controller."""
    if not self.autoplay_controller:
        return False
        
    checkpoint_path = self.get_current_model_path()
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Ottieni la complessità del modello dal nome del file
        model_complexity = "base"  # Default
        for complexity in ["base", "avanzato", "complesso", "perfetto"]:
            if complexity in os.path.basename(checkpoint_path):
                model_complexity = complexity
                break
        
        try:
            # Reinizializza il controller autoplay con il nuovo modello
            # [...]
            return True
        except Exception as e:
            print(f"Errore nel caricamento del modello: {e}")
            return False
    return False
```

### 4. Avvio con Selezione del Modello

```python
# In main.py
if args.mode == 'autoplay' or args.mode == 'train-and-play':
    # [...] Modalità autoplay
    ui = GameUI(game=game, speed=args.speed, autoplay_controller=controller)
    
    # Se richiesto, mostra subito il selettore di modelli
    if hasattr(args, 'select_model') and args.select_model:
        print("Attivazione finestra di selezione modello...")
        ui.toggle_model_selector()
        
    ui.run()
```

## 🔍 Utilizzo Consigliato

### Avvio con Selezione del Modello
```bash
python main.py --mode autoplay --select-model
```

### Caricamento di Modello Specifico
```bash
python main.py --mode autoplay --model avanzato --checkpoint training/checkpoints/dqn_avanzato_final.pt
```

### Addestramento e Selezione
```bash
# Prima addestra un modello
python main.py --mode train --model base --episodes 1000

# Poi avvia con selezione modelli
python main.py --mode autoplay --select-model
```

## 📈 Prossimi Sviluppi

- Supporto per categorie di modelli nell'interfaccia di selezione
- Visualizzazione delle prestazioni previste per ogni modello
- Possibilità di confrontare più modelli simultaneamente
- Interfaccia di debug per analizzare le decisioni del modello in tempo reale

---

Per ulteriori dettagli, consultare il [Manuale Utente](../MANUALE_UTENTE.md) aggiornato.

© 2023 - Baratti Federico - Versione 2.0 