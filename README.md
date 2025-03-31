# 🐍 Snake con UI e Deep Q-Learning (DQN) - Versione 2.0

Benvenuti nell'ecosistema **Snake con UI e Deep Q-Learning**, un framework avanzato per la sperimentazione di algoritmi di Reinforcement Learning applicati al classico gioco Snake. Questo progetto implementa un ambiente completamente parametrizzabile con architetture DQN ottimizzate e un'interfaccia grafica ad alte prestazioni.

## 🆕 Novità della Versione 2.0

Questa versione introduce importanti miglioramenti all'interfaccia utente e nuove funzionalità:

- **Selezione visuale dei modelli**: Interfaccia intuitiva per caricare dinamicamente modelli preaddestrati
- **Cambio modello in runtime**: Possibilità di alternare tra diversi modelli senza riavviare l'applicazione
- **UI migliorata**: Design più moderno e responsive con migliore feedback visivo
- **Modalità di avvio rapido**: Parametro `--select-model` per avviare direttamente con la selezione del modello
- **Documentazione aggiornata**: Manuale utente completo con dettagli su tutte le nuove funzionalità

## 📋 Panoramica Architetturale

Il framework è costruito su un'architettura a microservizi con disaccoppiamento completo tra componenti:

1. 🎮 **Frontend modulare** con rendering ottimizzato e pipeline di visualizzazione accelerata
2. 🤖 **Backend multi-threaded** con simulazione fisica paralelizzabile e ottimizzata per calcolo vettoriale
3. 🧠 **Runtime DQN completamente parametrizzabile** con supporto per architetture neurali avanzate
4. 🔄 **Sistema di preaddestramento sintetico** con generazione euristica di scenari ottimali
5. 🛠️ **Training engine distribuito** con supporto per multi-GPU e accelerazione hardware specializzata
6. 📊 **Telemetria avanzata** con visualizzazione in tempo reale delle metriche di apprendimento
7. 🖱️ **Interfaccia di selezione modelli** con supporto per caricamento dinamico e confronto in tempo reale

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
6. **Selezione dinamica del modello** con caricamento a runtime di checkpoint preaddestrati

## 📌 Getting Started

Per un deployment e configurazione ottimali, consultare:

- [Guida all'Installazione](INSTALLATION.md) - Configurazione dell'ambiente di runtime e deployment delle dipendenze
- [Architettura del Sistema](ARCHITECTURE.md) - Specifiche architetturali complete e diagrammi dei componenti
- [Manuale di Training](TRAINING.md) - Protocolli di addestramento e ottimizzazione degli iperparametri
- [Documentazione Autoplay](AUTOPLAY.md) - Configurazione dell'agente in modalità inferenza
- [Specifiche del Codice](CODE_GUIDE.md) - Documentazione API e specifiche di implementazione
- [Manuale Utente](MANUALE_UTENTE.md) - Guida completa all'utilizzo del software e tutte le funzionalità

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
python main.py --mode manual --grid-size 20

# Inferenza con modello avanzato preaddestrato
python main.py --mode autoplay --model avanzato --checkpoint training/checkpoints/dqn_avanzato_latest.pt

# Avvio con selezione visuale del modello
python main.py --mode autoplay --select-model

# Training distribuito con modello complesso
python main.py --mode train --model complesso --episodes 10000 --grid-size 20

# Preaddestramento euristico seguito da training completo e deployment in inferenza
python main.py --mode train-and-play --model perfetto --episodes 5000 --grid-size 30

# Modalità demo per una dimostrazione rapida
python main.py --demo
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

I contributi sono benvenuti secondo il protocollo GitFlow. Prima di contribuire, consultare [MANUALE_UTENTE.md](MANUALE_UTENTE.md) per le linee guida tecniche.

## 📞 Contatti

Per segnalazioni tecniche o richieste di feature, utilizzare il sistema di issue tracking o contattare gli autori tramite i canali ufficiali.

## 📝 Changelog

### Versione 2.0
- Aggiunta interfaccia di selezione visuale dei modelli
- Implementato sistema di caricamento dinamico dei checkpoint
- Migliorata l'interfaccia utente con feedback visivo avanzato
- Aggiunto parametro di avvio rapido `--select-model`
- Aggiornata documentazione e manuale utente

### Versione 1.0
- Rilascio iniziale con funzionalità di base
- Implementazione del gioco Snake con controlli manuali
- Agente DQN con supporto per modalità autoplay
- Sistema di addestramento con checkpoint
- Interfaccia grafica in Pygame

---

Grazie per l'interesse nel nostro framework di reinforcement learning! 🚀 
