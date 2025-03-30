# 🔧 Specifiche di Deployment e Configurazione del Sistema

Questa documentazione tecnica fornisce le procedure dettagliate per il deployment, la configurazione e l'ottimizzazione dell'ambiente di runtime per il framework Snake con UI e Deep Q-Learning.

## 📋 Requisiti di Sistema

### Specifiche Hardware

- **CPU**: 
  - **Minimo**: Dual-core x86-64 o ARM64 @ 2.0GHz+
  - **Raccomandato**: Quad-core+ x86-64 @ 3.0GHz+ con supporto AVX2
  - **Ottimale**: Octa-core+ x86-64 @ 3.5GHz+ con supporto AVX512

- **Memoria**:
  - **Minimo**: 4GB RAM (DDR3-1600+ o equivalente)
  - **Raccomandato**: 8GB+ RAM (DDR4-2400+ o equivalente)
  - **Ottimale per Training**: 16GB+ RAM (DDR4-3200+ o equivalente)

- **Storage**:
  - **Minimo**: 500MB spazio disponibile su storage standard
  - **Raccomandato**: 2GB+ su SSD (lettura 500MB/s+)
  - **Ottimale**: 5GB+ su NVMe SSD (lettura 2GB/s+)

- **GPU** (opzionale ma fortemente consigliato per training avanzato):
  - **Entry-level**: NVIDIA GTX 1050+ / AMD RX 550+ (2GB+ VRAM)
  - **Mid-range**: NVIDIA GTX 1660+ / AMD RX 5500+ (4GB+ VRAM)
  - **High-end**: NVIDIA RTX 2060+ / AMD RX 5700+ (6GB+ VRAM)
  - **Workstation**: NVIDIA RTX 3070+ / AMD RX 6800+ (8GB+ VRAM)

### Specifiche Software

- **Sistema Operativo**:
  - **Windows**: Windows 10 1903+ / Windows 11 (build 21H2+)
  - **macOS**: Catalina 10.15+ / Apple Silicon con Rosetta 2
  - **Linux**: Kernel 5.4+ con glibc 2.31+ (Ubuntu 20.04+, Debian 11+, Fedora 34+)

- **Python**:
  - **Versione**: Python 3.11.9 o superiore
  - **Componentistica**: Tkinter supportato, OpenSSL 1.1.1+
  - **Compilazione**: Build con ottimizzazioni PGO (consigliato)

- **Librerie di sistema**:
  - **Windows**: VCRuntime 2015+ (14.X+), DirectX 11+
  - **Linux**: X11 o Wayland, Mesa 20.0+, OpenGL 3.3+
  - **macOS**: Metal 2.0+, Cocoa framework

## 🛠️ Procedura di Installazione

### 1. Setup dell'Ambiente di Sviluppo

#### 1.1 Clonazione del Repository

```bash
# Clona il repository con opzione per submoduli
git clone --recurse-submodules https://github.com/tuorepository/snake-dqn.git

# Navigazione nella directory del progetto
cd snake-dqn

# Verifica dell'integrità (opzionale ma consigliato)
git lfs pull
```

#### 1.2 Configurazione dell'Ambiente Virtuale

La segmentazione dell'ambiente tramite virtualizzazione è fortemente consigliata per isolare le dipendenze e garantire compatibilità.

##### Per Windows:

```bash
# Creazione dell'ambiente con venv
python -m venv venv --prompt snake-dqn

# Attivazione dell'ambiente
venv\Scripts\activate

# Verifica dell'isolamento
where python  # Dovrebbe puntare all'ambiente virtuale
```

##### Per macOS e Linux:

```bash
# Creazione dell'ambiente con venv
python3 -m venv venv --prompt snake-dqn

# Attivazione dell'ambiente
source venv/bin/activate

# Verifica dell'isolamento
which python  # Dovrebbe puntare all'ambiente virtuale
```

### 2. Installazione delle Dipendenze con Ottimizzazione

#### 2.1 Configurazione dei Canali di Distribuzione

```bash
# Configurazione di pip per utilizzare mirror veloci e indici ottimizzati
pip config set global.index-url https://pypi.org/simple
pip config set global.extra-index-url https://download.pytorch.org/whl/cu118
pip config set global.trusted-host pypi.org
```

#### 2.2 Installazione Ottimizzata delle Dipendenze

```bash
# Aggiornamento del package manager
pip install --upgrade pip setuptools wheel

# Installazione con ottimizzazioni hardware-aware
pip install -r requirements.txt --no-cache-dir --compile
```

#### 2.3 Verifica dell'Installazione

```bash
# Validazione dell'ambiente e delle dipendenze
python -m snake_dqn.validate_install

# Verifica della disponibilità di accelerazione hardware
python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available(), '- Dispositivi:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
```

### 3. Configurazione per Performance Ottimali

#### 3.1 Ottimizzazione CUDA (per GPU NVIDIA)

```bash
# Verifica della configurazione CUDA
python -m snake_dqn.tools.cuda_check

# Impostazione della cache CUDA per ottimizzare la compilazione JIT
# Windows
setx CUDA_CACHE_MAXSIZE 2147483648

# Linux/macOS
echo 'export CUDA_CACHE_MAXSIZE=2147483648' >> ~/.bashrc
source ~/.bashrc
```

#### 3.2 Ottimizzazione CPU

```bash
# Attivazione della libreria MKL per operazioni ottimizzate su CPU Intel
# Windows
setx MKL_NUM_THREADS 4
setx OMP_NUM_THREADS 4

# Linux/macOS
echo 'export MKL_NUM_THREADS=4' >> ~/.bashrc
echo 'export OMP_NUM_THREADS=4' >> ~/.bashrc
source ~/.bashrc
```

#### 3.3 Configurazione PyTorch

```python
# Script di esempio per ottimizzare PyTorch
import torch
import os

# Configurazione threading ottimale per il sistema
num_cores = os.cpu_count()
torch.set_num_threads(num_cores)

# Abilita autotune per convolutional layers
torch.backends.cudnn.benchmark = True

# Configura precision per hardware specifico
if torch.cuda.is_available():
    # Verifica supporto per Tensor Cores
    capability = torch.cuda.get_device_capability()
    if capability[0] >= 7:  # Volta+ (7.0 = Volta, 7.5 = Turing, 8.0 = Ampere)
        print("Tensor Cores supportati - Abilitazione ottimizzazioni avanzate")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
else:
    print("Configurazione ottimizzata per CPU")
    # Abilita AVX/AVX2 ottimizzazioni
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
```

### 4. Verifica dell'Installazione Completa

#### 4.1 Test Base del Sistema

```bash
# Avvio in modalità manuale per verificare l'UI
python main.py --mode manual

# Test rapido in modalità headless per verificare il backend
python -m snake_dqn.test.core_test --headless
```

#### 4.2 Verifica dell'Accelerazione Hardware

```bash
# Benchmark delle performance di training
python -m snake_dqn.benchmark.training_bench --duration 30

# Benchmark delle performance di inferenza
python -m snake_dqn.benchmark.inference_bench --model base --iterations 1000
```

#### 4.3 Test della Modalità Demo

```bash
# Esecuzione della demo con benchmark automatico
python main.py --demo --benchmark
```

## 🔍 Configurazione Avanzata

### 1. Multi-GPU Setup

Per sistemi con più GPU, è possibile configurare esplicitamente l'utilizzo delle risorse:

```bash
# Specifica quali GPU utilizzare (indici 0-indexed)
# Windows
setx CUDA_VISIBLE_DEVICES "0,1"

# Linux/macOS
export CUDA_VISIBLE_DEVICES="0,1"
```

### 2. Ottimizzazione Memory Footprint

Configurazioni per ottimizzare l'utilizzo di memoria:

```bash
# Limita il consumo di memoria GPU
# Windows
setx PYTORCH_CUDA_ALLOC_CONF "max_split_size_mb:128"

# Linux/macOS
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
```

### 3. Configurazione TPU (per Google Colab o Cloud TPU)

Per l'utilizzo di Tensor Processing Units:

```python
# Abilitazione PyTorch XLA per TPU
import os
os.environ['XRT_TPU_CONFIG'] = 'tpu_worker;0;localhost:51011'

import torch
import torch_xla
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"XLA Device: {device}")
```

## 🛑 Troubleshooting Avanzato

### 1. Problemi di Compatibilità CUDA

#### Sintomo: `CUDA error: no kernel image is available for execution on the device`

Questo errore indica incompatibilità tra versione CUDA e compute capability della GPU.

**Soluzione**:
```bash
# Reinstallazione di PyTorch con versione CUDA compatibile
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verifica compatibilità
python -c "import torch; print('CUDA version:', torch.version.cuda); print('Compute capability:', torch.cuda.get_device_capability())"
```

### 2. Problemi di Memoria GPU

#### Sintomo: `RuntimeError: CUDA out of memory`

**Soluzione**:
```bash
# Riduzione batch size e ottimizzazione memoria
python main.py --mode train --model base --batch-size 32 --buffer-size 5000 --optimize-memory

# In alternativa, abilitare gestione intelligente della memoria
python main.py --mode train --model base --gradient-checkpointing --amp
```

### 3. Prestazioni CPU Subottimali

#### Sintomo: Training molto lento su CPU

**Soluzione**:
```bash
# Abilita ottimizzazioni CPU-specifiche
python main.py --mode train --model base --optimize-for cpu --num-workers auto --vectorize

# Verifica ottimizzazioni
python -m snake_dqn.tools.cpu_check
```

### 4. Problemi di Visualizzazione

#### Sintomo: Rendering lento o crash nell'interfaccia

**Soluzione**:
```bash
# Utilizzo di backend di rendering software
python main.py --mode manual --render-backend software

# Diagnosi problemi grafici
python -m snake_dqn.tools.graphics_check
```

## 🔄 Aggiornamento del Sistema

Per aggiornare il framework all'ultima versione stabile:

```bash
# Pull dell'ultima versione dal repository
git pull origin main

# Aggiornamento delle dipendenze
pip install -r requirements.txt --upgrade

# Pulizia cache e file temporanei
python -m snake_dqn.tools.clean_cache
```

## 📊 Validazione dell'Ambiente

Il framework include un tool di validazione completa per verificare la corretta configurazione dell'ambiente:

```bash
# Esecuzione validazione completa
python -m snake_dqn.validate --all

# Validazione componenti specifici
python -m snake_dqn.validate --gpu --cpu --memory --display
```

Output di un sistema correttamente configurato:
```
✅ Python: 3.11.9 (compatibile)
✅ Sistema Operativo: Windows 10 (build 21H2+)
✅ CPU: AMD Ryzen 7 5800X (8 core, AVX2 supportato)
✅ Memoria: 32GB disponibili (sufficiente)
✅ GPU: NVIDIA RTX 3070 (8GB VRAM, compute capability 8.6)
✅ CUDA: 11.8 (compatibile con PyTorch 2.2.0)
✅ Display: 2560x1440 @ 144Hz (compatibile)
✅ Librerie richieste: tutte installate correttamente
✅ Benchmark veloce: prestazioni nei parametri attesi

💯 Ambiente ottimale! Il sistema è configurato per massime prestazioni.
```

---

In caso di problemi durante l'installazione o la configurazione, consultare la documentazione online o aprire un issue sul repository del progetto. 