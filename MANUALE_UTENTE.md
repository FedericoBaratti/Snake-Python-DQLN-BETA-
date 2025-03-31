# 📘 Manuale Utente - Snake con UI e Deep Q-Learning

## 📋 Indice
1. [Introduzione](#introduzione)
2. [Installazione](#installazione)
3. [Avvio del Gioco](#avvio-del-gioco)
4. [Modalità di Gioco](#modalità-di-gioco)
5. [Comandi di Gioco](#comandi-di-gioco)
6. [Selezione del Modello](#selezione-del-modello)
7. [Addestramento del Modello](#addestramento-del-modello)
8. [Configurazione Avanzata](#configurazione-avanzata)
9. [Risoluzione Problemi](#risoluzione-problemi)
10. [Funzionalità Aggiuntive](#funzionalità-aggiuntive)
11. [Architettura del Software](#architettura-del-software)
12. [Test del Sistema](#test-del-sistema)
13. [Utilizzo per Ricerca e Sviluppo](#utilizzo-per-ricerca-e-sviluppo)
14. [Note Finali](#note-finali)

## 🎮 Introduzione

Snake con UI e Deep Q-Learning è un'implementazione avanzata del classico gioco Snake che integra tecniche di Reinforcement Learning per addestrare un agente in grado di giocare autonomamente. Il progetto combina un'interfaccia grafica intuitiva con algoritmi di Deep Q-Network (DQN) per creare un ambiente interattivo dove puoi giocare manualmente o lasciare che l'intelligenza artificiale giochi per te.

Versione 2.0 aggiunge un'interfaccia migliorata con supporto per la selezione visuale dei modelli e altre funzionalità avanzate.

## 💾 Installazione

### Requisiti di Sistema
- Python 3.7 o superiore
- Memoria RAM: 4GB minimo (8GB raccomandata)
- GPU: opzionale ma raccomandata per addestramento di modelli complessi

### Installazione Pacchetti
1. Clona o scarica il repository
2. Installa le dipendenze:
   ```
   pip install -r requirements.txt
   ```

I pacchetti principali richiesti includono:
- PyTorch (per il deep learning)
- Pygame (per l'interfaccia grafica)
- NumPy (per calcoli numerici)
- Gymnasium (ambiente compatibile con OpenAI Gym)

## 🚀 Avvio del Gioco

Il gioco può essere avviato utilizzando il file principale `main.py` con varie opzioni:

### Avvio Base
```
python main.py
```
Questo avvia il gioco in modalità manuale con le impostazioni predefinite.

### Parametri di Avvio
- `--mode`: Seleziona la modalità di gioco
  - `manual`: Controllo manuale (predefinito)
  - `autoplay`: Il gioco è controllato da un agente DQN addestrato
  - `train`: Addestra un nuovo modello DQN
  - `train-and-play`: Addestra un modello e poi mostra l'autoplay
- `--model`: Seleziona la complessità del modello DQN
  - `base`: Modello semplice (3k parametri)
  - `avanzato`: Modello più complesso (12k parametri)
  - `complesso`: Modello avanzato (40k parametri)
  - `perfetto`: Modello altamente ottimizzato (170k parametri)
- `--checkpoint`: Percorso al file di checkpoint del modello (opzionale)
- `--grid-size`: Dimensione della griglia di gioco (predefinito: 20)
- `--speed`: Velocità iniziale del gioco (1-20, predefinito: 10)
- `--episodes`: Numero di episodi per l'addestramento
- `--demo`: Attiva la modalità demo (addestramento rapido)
- `--select-model`: Avvia direttamente con la finestra di selezione del modello

### Esempi di Utilizzo
```
# Gioca in modalità manuale su una griglia 15x15
python main.py --grid-size 15

# Avvia la modalità autoplay con un modello avanzato preaddestrato
python main.py --mode autoplay --model avanzato --checkpoint training/checkpoints/dqn_avanzato_latest.pt

# Avvia la modalità autoplay e mostra subito la finestra di selezione del modello
python main.py --mode autoplay --select-model

# Addestra un nuovo modello base e poi gioca in autoplay
python main.py --mode train-and-play --model base --grid-size 10 --episodes 1000

# Avvia la modalità demo (addestramento rapido)
python main.py --demo
```

## 🎮 Modalità di Gioco

### Modalità Manuale
In questa modalità, controlli il serpente utilizzando i tasti freccia o WASD. L'obiettivo è mangiare il cibo senza colpire il tuo corpo o i bordi della griglia.

Caratteristiche:
- Control completo del movimento del serpente
- Visualizzazione del punteggio in tempo reale
- Memorizzazione del punteggio più alto

### Modalità Autoplay
In questa modalità, il serpente è controllato da un agente DQN precedentemente addestrato. Puoi osservare come l'agente gioca da solo.

Caratteristiche:
- Visualizzazione delle azioni intraprese dall'agente
- Statistiche sul decision-making dell'agente
- Possibilità di alternare tra controllo manuale e automatico con il tasto T

### Modalità Addestramento
Questa modalità avvia il processo di addestramento di un nuovo agente DQN. Durante l'addestramento, vedrai statistiche e metriche di apprendimento.

Caratteristiche:
- Visualizzazione delle metriche di addestramento in tempo reale
- Salvataggio automatico di checkpoint periodici
- Possibilità di interrompere e riprendere l'addestramento

### Modalità Addestramento e Gioco
Simile alla modalità di addestramento, ma al termine passa automaticamente alla modalità autoplay per mostrare le prestazioni dell'agente addestrato.

## ⌨️ Comandi di Gioco

### Controlli Base
- **Frecce** o **WASD**: Muovi il serpente
- **Spazio**: Pausa/Riprendi gioco
- **R**: Riavvia il gioco
- **ESC**: Esci dal gioco

### Controlli Aggiuntivi
- **T**: Attiva/disattiva modalità autoplay (se disponibile)
- **M**: Apri/chiudi la finestra di selezione del modello
- **+/-**: Aumenta/diminuisci la velocità del gioco
- **1-4**: Imposta il livello di difficoltà (1=facile, 4=difficile)

### Controlli Finestra Selezione Modello
- **↑/↓**: Naviga tra i modelli disponibili
- **Invio**: Seleziona e carica il modello evidenziato
- **ESC**: Chiudi la finestra di selezione senza caricare

### Interfaccia Utente
L'interfaccia di gioco è composta da:
- **Griglia di gioco**: Area principale dove si svolge il gioco
- **Sidebar**: Mostra punteggio, record, modalità corrente e altre informazioni
- **Informazioni di stato**: Indicano se il gioco è in pausa, in modalità autoplay, ecc.
- **Finestra di selezione modello**: Interfaccia per selezionare e caricare modelli preaddestrati

## 🤖 Selezione del Modello

La versione 2.0 introduce un'interfaccia visuale per la selezione dei modelli preaddestrati, che permette di:

1. Visualizzare tutti i checkpoint disponibili
2. Caricare dinamicamente un modello senza riavviare il gioco
3. Alternare tra diversi modelli per confrontarne le prestazioni

### Utilizzo della Finestra di Selezione

Per aprire la finestra di selezione del modello:
- Premi il tasto **M** durante il gioco
- Oppure avvia il gioco con il parametro `--select-model`

Nella finestra di selezione:
1. Usa i tasti **↑/↓** per navigare tra i modelli disponibili
2. Premi **Invio** per selezionare e caricare il modello evidenziato
3. Premi **ESC** per chiudere la finestra senza caricare un modello

Una volta caricato un modello, la modalità autoplay verrà attivata automaticamente e potrai osservare come il modello selezionato gioca al gioco.

### Formati dei Modelli

I modelli sono salvati nella directory `training/checkpoints/` con il seguente formato di nome:
- `dqn_base_latest.pt`: Ultimo modello base salvato
- `dqn_avanzato_final.pt`: Modello avanzato finale
- `dqn_complesso_ep1000.pt`: Modello complesso salvato dopo 1000 episodi
- `dqn_perfetto_ep5000.pt`: Modello perfetto salvato dopo 5000 episodi

La complessità del modello viene riconosciuta automaticamente dal nome del file.

## 🧠 Addestramento del Modello

### Addestramento Base
Per addestrare rapidamente un modello base:
```
python main.py --mode train --model base --grid-size 10 --episodes 300
```

Questo comando:
1. Crea e addestra un modello DQN base su una griglia 10x10
2. Esegue 300 episodi di addestramento (sufficiente per dimostrazioni)
3. Salva il modello in `training/checkpoints/dqn_base_latest.pt` e `training/checkpoints/dqn_base_final.pt`
4. Esegue una valutazione del modello periodicamente durante l'addestramento

Per addestrare e poi passare direttamente alla modalità autoplay:
```
python main.py --mode train-and-play --model base --grid-size 10 --episodes 300
```

### Addestramento Personalizzato
Per un addestramento più avanzato:
```
python main.py --mode train --model avanzato --grid-size 15 --episodes 5000
```

### Preaddestramento Sintetico
Il sistema supporta il preaddestramento sintetico per accelerare la convergenza:
```
python pretraining/pretrain.py --model avanzato
```

Questo genera esperienze sintetiche per dare all'agente una base di conoscenze prima dell'addestramento reale.

### Modalità Demo
Per una dimostrazione rapida del sistema di addestramento e autoplay:
```
python main.py --demo
```

Questa modalità utilizza tutti i parametri ottimizzati per un'esperienza dimostrativa veloce.

### Monitoraggio dell'Addestramento
Durante l'addestramento, vengono visualizzate le seguenti metriche:
- Episodio corrente e progresso totale
- Reward medio per episodio
- Epsilon (ε) attuale (tasso di esplorazione)
- Lunghezza media dell'episodio
- Tempo stimato di completamento

### Checkpoint
I modelli vengono salvati periodicamente in `training/checkpoints/`. Puoi utilizzare questi checkpoint per:
- Riprendere l'addestramento da un punto specifico
  ```
  python main.py --mode train --model base --checkpoint training/checkpoints/dqn_base_ep1000.pt --episodes 2000
  ```
- Utilizzare l'agente addestrato in modalità autoplay
  ```
  python main.py --mode autoplay --model base --checkpoint training/checkpoints/dqn_base_final.pt
  ```
- Confrontare le prestazioni di diversi modelli

## ⚙️ Configurazione Avanzata

### Tipi di Modelli
Il software supporta 4 livelli di complessità dei modelli:

1. **Base** (3k parametri)
   - Adatto per: Addestramento rapido, hardware limitato
   - Richiede: Qualsiasi CPU
   - Architettura: 2 hidden layers [64, 32]
   - Performance: Buona per griglie fino a 10x10

2. **Avanzato** (12k parametri)
   - Adatto per: Migliori performance, computer medi
   - Richiede: 2+ core CPU, 1GB+ VRAM (consigliata)
   - Architettura: 3 hidden layers [128, 64, 32]
   - Performance: Buona per griglie fino a 15x15

3. **Complesso** (40k parametri)
   - Adatto per: Eccellenti performance, computer potenti
   - Richiede: 4+ core CPU, 2GB+ VRAM
   - Architettura: 4 hidden layers [256, 128, 64, 32]
   - Performance: Ottima per griglie fino a 20x20

4. **Perfetto** (170k parametri)
   - Adatto per: Performance massima, hardware avanzato
   - Richiede: 8+ core CPU, 4GB+ VRAM
   - Architettura: 4 hidden layers [512, 256, 128, 64]
   - Performance: Eccellente per qualsiasi dimensione di griglia

### Architettura DQN
Il sistema implementa diverse tecniche avanzate di Deep Q-Learning:

- **Experience Replay**: Memorizza e riutilizza le esperienze passate
- **Dueling DQN**: Separa la stima del valore di stato e del vantaggio dell'azione
- **Target Network**: Stabilizza l'addestramento usando una rete target aggiornata periodicamente
- **Epsilon-greedy Policy**: Bilancia esplorazione e sfruttamento durante l'addestramento

### Rilevamento Hardware Automatico
Il software rileva automaticamente l'hardware disponibile e ottimizza le configurazioni di conseguenza:
- Numero di CPU core
- Disponibilità e numero di GPU
- Memoria di sistema

## 🔧 Risoluzione Problemi

### Problemi Comuni

#### Il gioco è lento
- Riduci la dimensione della griglia (`--grid-size`)
- Utilizza un modello più semplice (`--model base`)
- Assicurati che non ci siano altri programmi che consumano risorse in background

#### Errori durante l'addestramento
- Assicurati di avere abbastanza memoria disponibile
- Utilizza un modello meno complesso per l'hardware disponibile
- Riduci la dimensione del batch o della memoria di replay

#### Il modello non impara bene
- Aumenta il numero di episodi di addestramento
- Prova un modello più complesso
- Regola i parametri di addestramento come learning rate o gamma

#### Crash all'avvio dell'autoplay
- Verifica che il checkpoint specificato esista
- Assicurati che il modello e la dimensione della griglia corrispondano a quelli usati durante l'addestramento

### Supporto Hardware Specifico

#### CUDA e GPU
Il software supporta automaticamente l'accelerazione GPU tramite CUDA se disponibile. Per verificare la disponibilità:
```
python -c "import torch; print('CUDA disponibile:', torch.cuda.is_available())"
```

#### Ottimizzazione CPU
Su sistemi senza GPU, il software utilizza ottimizzazioni specifiche per CPU:
- Multi-threading per operazioni parallele
- Mini-batch più piccoli per ridurre l'utilizzo di memoria

## 🔍 Funzionalità Aggiuntive

### Selezione Dinamica del Modello
La nuova interfaccia di selezione del modello permette di:
- Visualizzare e selezionare qualsiasi modello disponibile nel sistema
- Cambiare modello durante il gioco senza riavviare l'applicazione
- Confrontare diversi modelli in tempo reale

### Modalità Demo
La modalità demo è pensata per una dimostrazione rapida delle capacità del software:
```
python main.py --demo
```

Questa modalità:
1. Imposta la dimensione della griglia a 10x10
2. Addestra un modello base per 300 episodi
3. Passa automaticamente alla modalità autoplay

### Visualizzazione delle Prestazioni
Durante la modalità autoplay, viene visualizzata una sidebar con:
- Punteggio attuale e record
- Informazioni sul modello utilizzato
- Azioni intraprese dall'agente
- Statistiche sulle prestazioni

### Velocità di Gioco Dinamica
È possibile regolare la velocità del gioco in tempo reale utilizzando i tasti + e -. La velocità influisce solo sulla visualizzazione, non sull'addestramento o sulle decisioni dell'agente.

### Ambiente Compatibile con Gymnasium
Il sistema implementa un'interfaccia compatibile con Gymnasium (successore di OpenAI Gym), permettendo l'utilizzo di altri algoritmi di reinforcement learning con minime modifiche.

## 📊 Architettura del Software

Il sistema è organizzato secondo un'architettura modulare con diversi componenti:

1. **Core Game & Engine** (`backend/`)
   - Implementa la logica del gioco Snake
   - Fornisce un ambiente compatibile con Gymnasium

2. **Interfaccia Utente** (`frontend/`)
   - Gestisce la visualizzazione e l'interazione con l'utente
   - Implementata con Pygame

3. **DQN Agent** (`dqn_agent/`)
   - Implementa l'agente basato su Deep Q-Learning
   - Definisce le architetture di rete neurale utilizzate

4. **Preaddestramento Sintetico** (`pretraining/`)
   - Implementa la fase di preaddestramento dell'agente

5. **Training Reale** (`training/`)
   - Gestisce il training dell'agente sull'ambiente completo

6. **Autoplay Controller** (`autoplay/`)
   - Coordina l'interazione tra l'agente addestrato e l'ambiente di gioco

7. **Test Suite** (`tests/`)
   - Implementa test unitari e interattivi per il sistema
   - Offre ambienti di test automatizzati e manuali

Per maggiori dettagli sull'architettura, consulta il file [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🧪 Test del Sistema

Il sistema include una suite completa di test per verificare il corretto funzionamento del gioco Snake e delle sue componenti. Questi test sono disponibili nella cartella `tests/` e possono essere utilizzati sia per lo sviluppo che per verificare l'installazione.

### Tipi di Test Disponibili

#### 1. Test Unitari (`test_snake_game.py`)
Test completi che verificano tutte le funzionalità del gioco Snake utilizzando il framework unittest di Python.

**Esecuzione:**
```bash
python -m tests.test_snake_game
```

Per eseguire un test specifico:
```bash
python -m tests.test_snake_game TestSnakeGame.test_movement
```

#### 2. Test Interattivo Avanzato (`test_snake_interactive_v2.py`)
Interfaccia grafica avanzata che permette di testare manualmente il gioco o utilizzare una modalità automatica con un algoritmo semplice.

**Caratteristiche:**
- Controlli con frecce direzionali
- Modalità automatica attivabile con 'A'
- Regolazione velocità con '+/-'
- Pausa/Riavvio con 'P'/'R'

**Esecuzione:**
```bash
python -m tests.test_snake_interactive_v2
```

#### 3. Test Interattivo Semplice (`test_snake_simple.py`)
Versione semplificata dell'interfaccia grafica per test rapidi e leggeri.

**Esecuzione:**
```bash
python -m tests.test_snake_simple
```

#### 4. Test Automatico (`test_snake_auto.py`)
Test che esegue automaticamente il gioco senza interfaccia grafica, utilizzando un algoritmo euristico.

**Esecuzione:**
```bash
python -m tests.test_snake_auto
```

Per eseguire più test consecutivi e vedere statistiche:
```bash
python -m tests.test_snake_auto 5  # Esegue 5 test
```

### Utilizzo dei Test per il Debugging

I test interattivi sono particolarmente utili per:
- Verificare il corretto funzionamento grafico del gioco
- Testare le meccaniche di movimento e collisione
- Verificare l'integrazione tra backend e frontend

I test automatici sono ideali per:
- Verificare rapidamente le performance del gioco
- Identificare potenziali problemi di stabilità
- Generare dati statistici sul funzionamento

### Documentazione Dettagliata dei Test

Per una documentazione completa sui test disponibili, consultare il file [TESTS.md](docs/TESTS.md), che include:
- Spiegazione dettagliata di ogni test
- Guida all'esecuzione e personalizzazione
- Casi d'uso comuni e configurazioni avanzate
- Suggerimenti per la risoluzione di problemi

## 🚀 Utilizzo per Ricerca e Sviluppo

### Estendere il Modello
Il sistema è progettato per essere facilmente estensibile:

1. **Nuovi Modelli**: 
   - Aggiungi nuove architetture di rete in `dqn_agent/models.py`
   - Registra il nuovo modello in `dqn_agent/config.py`

2. **Algoritmi Alternativi**:
   - Implementa nuovi algoritmi di RL in una nuova directory
   - Aggiungi interfacce compatibili con l'ambiente esistente

3. **Ambienti Personalizzati**:
   - Crea nuovi ambienti che seguono l'interfaccia Gymnasium
   - Modifica i parametri di ricompensa per sperimentare

### Metrics e Logging
Il sistema include strumenti per monitorare le prestazioni:
- Logging automatico delle metriche di training
- Salvataggio delle sessioni per analisi successive
- Valutazione automatica del modello

---

## 📝 Note Finali

Questo software è stato progettato come strumento educativo per imparare i principi del Reinforcement Learning attraverso un'applicazione pratica. Il codice è organizzato in modo modulare per facilitare l'apprendimento e la sperimentazione.

La versione 2.0 introduce significativi miglioramenti all'interfaccia utente e nuove funzionalità per una migliore esperienza di utilizzo.

Per maggiori informazioni o supporto, consulta la documentazione di ogni modulo o visita il repository del progetto. 