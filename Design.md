# 🐍🎮 **Design definitivo e dettagliato del Progetto: Snake con UI e Deep Q-Learning (DQN)**

### **Versione Python:** **3.11.9**
### **Tecnologie utilizzate:** Python, Pygame, PyTorch, OpenAI Gym

---

## 📌 **Obiettivi del progetto:**

1. Realizzare il classico gioco **Snake** con una UI moderna (Pygame).
2. Implementare una modalità **autoplay** basata su un agente **Deep Q-Learning** (DQN).
3. Includere un **preaddestramento sintetico** per velocizzare la convergenza del modello.
4. Garantire che il training sfrutti al massimo tutte le risorse hardware disponibili (CPU, GPU, TPU).
5. Offrire modularità nella complessità del modello (base, avanzato, complesso, perfetto).

---

## 📁 **Struttura dettagliata delle cartelle e file del progetto:**
```
snake-rl-project/
├── frontend/
│   └── ui.py                     # UI grafica (Pygame)
├── backend/
│   ├── snake_game.py             # Logica core del gioco
│   ├── environment.py            # Ambiente custom (Gym API)
│   └── utils.py                  # Funzioni utilità (stato, reward)
├── dqn_agent/
│   ├── dqn_agent.py              # Gestione DQN e memoria replay
│   ├── models.py                 # Architetture modulari DQN
│   └── config.py                 # Gestione configurazione modelli
├── pretraining/
│   ├── synthetic_env.py          # Ambiente sintetico per pretraining
│   └── pretrain.py               # Script pre-training sintetico
├── training/
│   ├── train.py                  # Training reale ottimizzato
│   └── checkpoints/              # Salvataggio modelli
├── autoplay/
│   └── autoplay.py               # Autoplay integrato con UI
├── tests/
│   ├── test_snake_game.py        # Test unitari per SnakeGame
│   ├── test_snake_interactive_v2.py # Test interattivo avanzato
│   ├── test_snake_simple.py      # Test interattivo semplice
│   ├── test_snake_auto.py        # Test automatico senza GUI
│   └── TESTS.md                  # Documentazione sui test
├── docs/
│   ├── README.md                 # Panoramica del progetto
│   ├── INSTALLATION.md           # Guida all'installazione
│   ├── ARCHITECTURE.md           # Spiegazione architetturale dettagliata
│   ├── TRAINING.md               # Istruzioni training e risorse
│   ├── AUTOPLAY.md               # Guida modalità autoplay
│   ├── CODE_GUIDE.md             # Guida dettagliata codice sorgente
│   └── TESTS.md                  # Documentazione dettagliata dei test
├── requirements.txt              # Dipendenze del progetto
└── main.py                       # Punto di avvio principale
```

---

## 🖥️ **Frontend (UI con Pygame)**

### Funzionalità principali (`ui.py`):

- Visualizzazione grafica:
  - Griglia di gioco
  - Serpente, cibo, punteggio, record
- Interazioni utente:
  - Controllo manuale (tastiera: WASD/frecce)
  - Switch modalità autonoma/manuale
- Collegamento backend:
  - Invoca `step(action)`, `reset()`, `render()`

**Flusso UI ↔ Backend:**
```
UI (azione utente o autoplay) → Backend (esecuzione azione) → UI (aggiornamento visivo)
```

---

## ⚙️ **Backend (Logica di gioco e Ambiente Gym)**

### 📌 **snake_game.py**

Classe `SnakeGame`:
- Metodi fondamentali:
  - `reset()`: riavvio gioco
  - `step(action)`: aggiorna stato e calcola reward
  - `is_collision(position)`: collisioni muro/corpo
  - `get_state()`: stato numerico gioco
  - `get_score()`: punteggio corrente

### 📌 **environment.py**

Classe Gym `SnakeEnv(gym.Env)`:
- Compatibile con API Gym:
  - `step(action)`, `reset()`, `render()`, `close()`
  - Comunica con SnakeGame per aggiornare lo stato.

---

## 🧠 **DQN Agent (Agente Deep Q-Learning)**

### 📌 **dqn_agent.py**

Classe `DQNAgent`:
- Replay buffer (memoria esperienze passate)
- Policy ε-greedy
- Metodi training (`optimize_model()`), salvataggio e caricamento checkpoint.

### 📌 **models.py**

Strutture reti neurali modulari, selezionabili da config:

| Livello    | Architettura rete neurale (PyTorch)                          | Parametri |
|------------|---------------------------------------------------------------|-----------|
| Base       | FC(64) → ReLU → FC(32) → ReLU → FC(4)                         | ~3k       |
| Avanzato   | FC(128) → ReLU → FC(64) → ReLU → FC(32) → ReLU → FC(4)        | ~12k      |
| Complesso  | FC(256) → ReLU → FC(128) → ReLU → FC(64) → ReLU → FC(32) → FC(4)| ~40k      |
| Perfetto   | FC(512) → ReLU → FC(256) → ReLU → FC(128) → ReLU → FC(64) → FC(4)| ~170k     |

### 📌 **config.py**
- Configurazione modulare livelli modello, parametri training, risorse hardware.

---

## 🚀 **Preaddestramento sintetico**

### 📌 **synthetic_env.py**
- Ambiente semplificato per produrre rapidamente esperienze iniziali utili.
- Riduzione complessità (es. griglia più piccola, velocità superiore).

### 📌 **pretrain.py**
- Script automatizzato per pre-allenare velocemente il modello DQN.
- Salva checkpoint per proseguimento training reale.

---

## 💻 **Training reale ottimizzato (train.py)**

- Carica modello preaddestrato.
- Allenamento intensivo sfruttando:
  - CPU multi-core (multiprocessing/threading)
  - GPU multiple (PyTorch DataParallel/DistributedDataParallel)
  - TPU (opzionale, PyTorch XLA se disponibile)
- Logging dettagliato performance (loss, reward, Q-value).
- Salvataggio checkpoint periodici.

**Ottimizzazione hardware automatica:**
```
if GPU disponibile → uso GPU
elif TPU disponibile → uso TPU
else → CPU multi-core (massimo carico distribuito)
```

---

## 🤖 **Modalità Autoplay**

### 📌 **autoplay.py**
- Carica automaticamente modelli addestrati.
- Invoca backend con DQN agente per predizione azioni.
- Visualizza agente autonomo tramite UI.

---

## 🧪 **Testing completo (`tests/`)**

### 📌 **Tipologie di test implementate**

1. **Test Unitari (`test_snake_game.py`)**
   - Verifica funzionamento di tutte le componenti del gioco
   - Test inizializzazione, movimento, collisioni, punteggio
   - Validazione comportamento completo del gioco

2. **Test Interattivo Avanzato (`test_snake_interactive_v2.py`)**
   - GUI completa per testing manuale e automatico
   - Modalità auto con intelligenza semplice
   - Regolazione velocità, visualizzazione statistiche
   - Controlli avanzati di pausa/reset/replay

3. **Test Interattivo Semplice (`test_snake_simple.py`)**
   - Versione leggera per test rapidi
   - Interfaccia minimalista con controlli essenziali

4. **Test Automatico (`test_snake_auto.py`)**
   - Esecuzione automatica senza UI
   - Algoritmo euristico semplice per valutare gameplay
   - Metriche di performance e statistiche multiple

### 📌 **Documentazione dei test**

- **`TESTS.md`**: Documentazione dettagliata sull'uso dei test
- Guida per esecuzione e personalizzazione
- Spiegazione casi d'uso e configurazioni

**Esecuzione test unitari:**
```bash
python -m tests.test_snake_game
```

**Esecuzione test interattivi:**
```bash
python -m tests.test_snake_interactive_v2
```

---

## 📚 **Documentazione obbligatoria dettagliata (docs/)**

- `README.md`: Introduzione generale progetto, configurazione e livelli di complessità.
- `INSTALLATION.md`: Guida all'installazione passo-passo.
- `ARCHITECTURE.md`: Spiegazione approfondita architettura software, rete DQN.
- `TRAINING.md`: Istruzioni dettagliate per training ottimizzato con hardware disponibile.
- `AUTOPLAY.md`: Guida modalità autonoma.
- `CODE_GUIDE.md`: Spiegazione dettagliata e guida completa ai file sorgente.
- `TESTS.md`: Documentazione dettagliata sui test disponibili e loro utilizzo.

---

## ⚙️ **File requirements.txt (librerie obbligatorie):**
```
pygame
torch
gymnasium
numpy
matplotlib
```

---

## 🚨 **Conclusione:**
Questo design definitivo è super-esplicativo, completo e pronto per essere implementato integralmente.  
Garantisce chiarezza architetturale, modularità, ottimizzazione hardware e qualità del codice, con una documentazione esaustiva per un'esperienza completa, professionale e pronta all'uso.