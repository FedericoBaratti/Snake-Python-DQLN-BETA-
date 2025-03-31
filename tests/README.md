# Test del Gioco Snake

Questa cartella contiene vari script per testare il gioco Snake implementato nel modulo `backend/snake_game.py`.

## File di Test

1. **test_snake_game.py**
   - Test unitari che verificano tutte le funzionalità del gioco
   - Verifica inizializzazione, movimento, collisioni, punteggio, ecc.
   - Esecuzione: `python -m tests.test_snake_game`

2. **test_snake_interactive_v2.py**
   - Interfaccia grafica avanzata per testare il gioco manualmente
   - Include modalità automatica e controlli della velocità
   - Esecuzione: `python -m tests.test_snake_interactive_v2`

3. **test_snake_simple.py**
   - Interfaccia grafica semplificata per test manuali
   - Più stabile e leggera
   - Esecuzione: `python -m tests.test_snake_simple`

4. **test_snake_auto.py**
   - Test automatico senza interfaccia grafica
   - Esegue il gioco con un algoritmo greedy semplice
   - Mostra statistiche e metriche di performance
   - Esecuzione: `python -m tests.test_snake_auto [num_tests]`

## Utilizzo

Per eseguire i test unitari:
```bash
python -m tests.test_snake_game
```

Per eseguire il test interattivo avanzato:
```bash
python -m tests.test_snake_interactive_v2
```

Per eseguire il test interattivo semplice:
```bash
python -m tests.test_snake_simple
```

Per eseguire il test automatico:
```bash
python -m tests.test_snake_auto
```

Per eseguire più test automatici:
```bash
python -m tests.test_snake_auto 5  # Esegue 5 test
```

## Note

L'interfaccia grafica richiede che pygame sia installato. Se incontri problemi con i test interattivi, prova il test automatico che non richiede una GUI. 