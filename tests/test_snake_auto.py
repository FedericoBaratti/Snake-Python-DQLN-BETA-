#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Automatico del Gioco Snake
===============================
Script per testare automaticamente il gioco Snake senza interfaccia grafica

Autore: Claude AI
Versione: 1.0
"""

import time
import sys
import random
from backend.snake_game import SnakeGame, Direction

def test_auto():
    """Esegue un test automatico del gioco Snake."""
    # Crea un'istanza del gioco
    game = SnakeGame(grid_size=10)  # Griglia più piccola per test più rapidi
    
    print("Test automatico del gioco Snake")
    print("-" * 40)
    print(f"Dimensione griglia: {game.grid_size}x{game.grid_size}")
    print(f"Posizione iniziale: {game.snake[0]}")
    print(f"Direzione iniziale: {game.direction}")
    print(f"Posizione cibo: {game.food}")
    print("-" * 40)
    
    # Azioni disponibili
    actions = [0, 1, 2]  # Evita l'inversione di marcia (3) che potrebbe causare collisioni immediate
    
    # Massimo numero di passi
    max_steps = 100
    steps = 0
    
    # Esegui il gioco fino al game over o al limite di passi
    while not game.done and steps < max_steps:
        # Ottieni informazioni sul gioco
        head_x, head_y = game.snake[0]
        food_x, food_y = game.food
        
        # Calcola la distanza dal cibo
        distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Scegli un'azione con una semplice euristica
        if random.random() < 0.7:  # 70% delle volte, cerca di avvicinarsi al cibo
            # Calcola la direzione verso il cibo
            if game.direction == Direction.RIGHT:
                if food_y < head_y:
                    action = 2  # Gira a sinistra (verso l'alto)
                elif food_y > head_y:
                    action = 1  # Gira a destra (verso il basso)
                else:
                    action = 0  # Continua dritto
            elif game.direction == Direction.LEFT:
                if food_y < head_y:
                    action = 1  # Gira a destra (verso l'alto)
                elif food_y > head_y:
                    action = 2  # Gira a sinistra (verso il basso)
                else:
                    action = 0  # Continua dritto
            elif game.direction == Direction.UP:
                if food_x < head_x:
                    action = 2  # Gira a sinistra (verso sinistra)
                elif food_x > head_x:
                    action = 1  # Gira a destra (verso destra)
                else:
                    action = 0  # Continua dritto
            elif game.direction == Direction.DOWN:
                if food_x < head_x:
                    action = 1  # Gira a destra (verso sinistra)
                elif food_x > head_x:
                    action = 2  # Gira a sinistra (verso destra)
                else:
                    action = 0  # Continua dritto
        else:
            # 30% delle volte, movimento casuale per evitare loop
            action = random.choice(actions)
        
        # Esegui l'azione
        state, reward, done, info = game.step(action)
        
        # Stampa informazioni sul passo corrente
        action_names = ["Avanti", "Destra", "Sinistra", "Indietro"]
        print(f"Passo {steps+1}:")
        print(f"  Azione: {action_names[action]}")
        print(f"  Direzione: {game.direction}")
        print(f"  Posizione testa: {game.snake[0]}")
        print(f"  Lunghezza serpente: {len(game.snake)}")
        print(f"  Posizione cibo: {game.food}")
        print(f"  Distanza dal cibo: {distance}")
        print(f"  Punteggio: {game.score}")
        print(f"  Reward: {reward}")
        print(f"  Game over: {done}")
        print()
        
        steps += 1
        time.sleep(0.1)  # Rallenta l'esecuzione per leggere meglio l'output
    
    # Stampa risultati finali
    print("=" * 40)
    print("Risultati del test:")
    print(f"Passi totali: {steps}")
    print(f"Punteggio finale: {game.score}")
    print(f"Game over: {game.done}")
    if game.done:
        print("Motivo: " + ("Collisione con muro/corpo" if game.is_collision(game.snake[0]) else "Troppi passi senza mangiare"))
    print("=" * 40)
    
    return game.score

def test_multiple(num_tests=5):
    """Esegue più test e calcola statistiche."""
    print(f"Esecuzione di {num_tests} test automatici...\n")
    
    scores = []
    start_time = time.time()
    
    for i in range(num_tests):
        print(f"\nTEST #{i+1}")
        print("=" * 40)
        score = test_auto()
        scores.append(score)
    
    end_time = time.time()
    
    print("\nRISULTATI FINALI")
    print("=" * 40)
    print(f"Numero di test: {num_tests}")
    print(f"Tempo totale: {end_time - start_time:.2f} secondi")
    print(f"Punteggio medio: {sum(scores) / num_tests:.2f}")
    print(f"Punteggio massimo: {max(scores)}")
    print(f"Punteggio minimo: {min(scores)}")
    print("=" * 40)

if __name__ == "__main__":
    try:
        # Numero di test da eseguire
        num_tests = 1
        
        # Se specificato un argomento, usa quello come numero di test
        if len(sys.argv) > 1:
            try:
                num_tests = int(sys.argv[1])
            except ValueError:
                print(f"Argomento non valido. Utilizzo il valore predefinito: {num_tests}")
        
        # Esegui i test
        if num_tests == 1:
            test_auto()
        else:
            test_multiple(num_tests)
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc() 