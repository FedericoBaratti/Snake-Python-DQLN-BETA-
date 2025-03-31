# Documentazione dei Test del Progetto Snake

Questo documento descrive dettagliatamente la cartella `tests` del progetto Snake-Python-DQLN, la sua organizzazione, i tipi di test disponibili e le diverse configurazioni per eseguire e personalizzare i test.

## Panoramica della Cartella Tests

La cartella `tests` contiene diversi script che consentono di testare il funzionamento del gioco Snake implementato in `backend/snake_game.py`. Questi test spaziano da test unitari automatizzati a interfacce grafiche interattive.

### Contenuto della Cartella

- `__init__.py` - File di inizializzazione che rende la cartella un modulo Python
- `test_snake_game.py` - Test unitari completi per verificare tutte le funzionalità
- `test_snake_interactive_v2.py` - GUI avanzata per test interattivi e modalità automatica 
- `test_snake_simple.py` - GUI semplificata per test interattivi di base
- `test_snake_auto.py` - Test automatizzati senza interfaccia grafica
- `TESTS.md` - Documentazione di base dei test

## Tipi di Test

### 1. Test Unitari (`test_snake_game.py`)

Questi test verificano sistematicamente ogni componente e funzionalità del gioco Snake tramite il framework unittest di Python.

#### Componenti Testati
- Inizializzazione del gioco
- Reset del gioco
- Posizionamento del cibo
- Generazione e normalizzazione dello stato
- Rotazione e movimento nelle diverse direzioni
- Rilevamento delle collisioni
- Comportamento del serpente quando mangia il cibo
- Terminazione corretta del gioco
- Funzioni di supporto (get_score, get_snake_positions, ecc.)

#### Esecuzione
```bash
python -m tests.test_snake_game
```

#### Configurazioni Avanzate
Per eseguire test specifici:
```bash
python -m tests.test_snake_game TestSnakeGame.test_movement
```

Per generare report dettagliati:
```bash
python -m unittest discover -s tests -p "test_snake_game.py" -v
```

### 2. Test Interattivo Avanzato (`test_snake_interactive_v2.py`)

Una GUI avanzata che permette di testare manualmente il gioco o utilizzare una modalità automatica di gioco.

#### Caratteristiche
- Controllo manuale con le frecce direzionali
- Modalità automatica intelligente
- Regolazione della velocità di gioco
- Visualizzazione dettagliata di statistiche
- Pausa e riavvio del gioco
- Gestione automatica del game over

#### Controlli
- **Frecce**: Movimento del serpente
  - Freccia Destra: Prosegui dritto
  - Freccia Giù: Gira a destra
  - Freccia Sinistra: Gira a sinistra
  - Freccia Su: Vai nella direzione opposta
- **P**: Pausa/Riprendi gioco
- **R**: Reset del gioco
- **A**: Attiva/Disattiva modalità automatica
- **+/-**: Aumenta/Diminuisci velocità
- **ESC**: Esci dal gioco

#### Esecuzione
```bash
python -m tests.test_snake_interactive_v2
```

#### Personalizzazione
È possibile modificare le dimensioni della griglia e delle celle:
```python
game_gui = SnakeGameGUI(grid_size=20, cell_size=25)
```

### 3. Test Interattivo Semplice (`test_snake_simple.py`)

Una versione semplificata dell'interfaccia grafica per test più leggeri e rapidi.

#### Caratteristiche
- Controllo manuale con le frecce direzionali
- Interfaccia minimalista
- Visualizzazione del punteggio
- Riavvio semplice

#### Controlli
- **Frecce**: Controllano la direzione del serpente
- **R**: Riavvia il gioco
- **ESC**: Esci dal gioco

#### Esecuzione
```bash
python -m tests.test_snake_simple
```

### 4. Test Automatico (`test_snake_auto.py`)

Test che esegue automaticamente il gioco senza interfaccia grafica, utilizzando una logica di movimento euristica.

#### Caratteristiche
- Esecuzione di uno o più test consecutivi
- Visualizzazione dettagliata delle azioni
- Statistiche di performance
- Configurazione del numero di test

#### Algoritmi di Movimento
Il test utilizza un algoritmo di movimento che combina:
- Logica euristica per muoversi verso il cibo (70% delle volte)
- Movimento casuale per evitare loop (30% delle volte)

#### Esecuzione
Per un singolo test:
```bash
python -m tests.test_snake_auto
```

Per eseguire più test consecutivi:
```bash
python -m tests.test_snake_auto 5  # Esegue 5 test
```

#### Personalizzazione
È possibile modificare i parametri del test nel codice:
- Dimensione della griglia (`grid_size`)
- Numero massimo di passi (`max_steps`)
- Proporzione tra movimento euristico e casuale
- Velocità di esecuzione (modificando il `time.sleep`)

## Casi d'Uso Comuni

### Sviluppo e Debugging
1. Utilizzare `test_snake_simple.py` per verificare rapidamente il funzionamento del gioco
2. Utilizzare `test_snake_interactive_v2.py` per testare in modo approfondito

### Test Automatizzati per CI/CD
1. Eseguire `test_snake_game.py` per verificare che tutte le funzionalità siano integre
2. Eseguire `test_snake_auto.py` con parametri specifici per misurare performance

### Addestramento e Validazione di Agenti IA
1. Utilizzare `test_snake_interactive_v2.py` in modalità automatica per verificare il comportamento semplice
2. Confrontare con altri modelli di IA implementati nel progetto

## Integrazione con Altre Parti del Progetto

La cartella `tests` fornisce un'interfaccia per testare principalmente le funzionalità del modulo `backend/snake_game.py`, che è alla base del gioco Snake. Questi test sono complementari al sistema di addestramento del modello DQN presente in altre parti del progetto, come le cartelle `training`, `pretraining` e `autoplay`.

## Risoluzione Problemi Comuni

### Errori di Pygame
Se riscontri problemi con i test interattivi, verifica che Pygame sia installato correttamente:
```bash
pip install pygame
```

### Prestazioni Basse
Per migliorare le prestazioni:
- Ridurre la dimensione della griglia nei test automatici
- Aumentare la velocità di esecuzione (FPS) nei test interattivi
- Utilizzare `test_snake_simple.py` invece di `test_snake_interactive_v2.py`

### Crash durante i Test
I test più comuni problemi sono:
- Collisioni impreviste in modalità automatica
- Problemi di risorse del sistema con Pygame
- Errori logici nel movimento del serpente

## Estensione e Personalizzazione

Per aggiungere nuovi test:
1. Creare un nuovo file nella cartella `tests`
2. Importare il modulo `backend.snake_game`
3. Implementare la logica di test
4. Aggiornare questo documento per includere il nuovo test

Per modificare la logica automatica:
1. Modificare la funzione `auto_play()` in `test_snake_interactive_v2.py`
2. O modificare la logica di selezione dell'azione in `test_snake_auto.py` 