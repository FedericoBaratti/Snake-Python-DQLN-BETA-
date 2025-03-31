#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del modulo Snake Game
==========================
Script per testare tutte le funzionalità del modulo snake_game.py

Autore: Claude AI
Versione: 1.0
"""

import unittest
import numpy as np
from backend.snake_game import SnakeGame, Direction

class TestSnakeGame(unittest.TestCase):
    """Classe per testare tutte le funzionalità del gioco Snake."""

    def setUp(self):
        """Inizializza un'istanza del gioco per ogni test."""
        self.grid_size = 20
        self.game = SnakeGame(grid_size=self.grid_size)

    def test_initialization(self):
        """Verifica che il gioco venga inizializzato correttamente."""
        # Verifica dimensione griglia
        self.assertEqual(self.game.grid_size, self.grid_size)
        # Verifica che il serpente abbia lunghezza 1
        self.assertEqual(len(self.game.snake), 1)
        # Verifica che il serpente sia posizionato al centro
        center = self.grid_size // 2
        self.assertEqual(self.game.snake[0], (center, center))
        # Verifica che il punteggio sia zero
        self.assertEqual(self.game.score, 0)
        # Verifica che il gioco non sia terminato
        self.assertFalse(self.game.done)

    def test_reset(self):
        """Verifica che il reset del gioco funzioni correttamente."""
        # Facciamo alcuni movimenti per cambiare lo stato del gioco
        self.game.step(0)
        self.game.step(1)
        self.game.score = 5
        
        # Reset del gioco
        self.game.reset()
        
        # Verifica che il punteggio sia tornato a zero
        self.assertEqual(self.game.score, 0)
        # Verifica che il serpente abbia di nuovo lunghezza 1
        self.assertEqual(len(self.game.snake), 1)
        # Verifica che il gioco non sia terminato
        self.assertFalse(self.game.done)

    def test_place_food(self):
        """Verifica che il cibo venga posizionato correttamente."""
        # Verifica che il cibo sia posizionato all'interno della griglia
        food_x, food_y = self.game.food
        self.assertGreaterEqual(food_x, 0)
        self.assertLess(food_x, self.grid_size)
        self.assertGreaterEqual(food_y, 0)
        self.assertLess(food_y, self.grid_size)
        
        # Verifica che il cibo non sia posizionato sul serpente
        self.assertNotIn(self.game.food, self.game.snake)

    def test_get_state(self):
        """Verifica che lo stato del gioco venga generato correttamente."""
        state = self.game.get_state()
        
        # Verifica dimensioni dello stato
        self.assertEqual(state.shape, (self.grid_size, self.grid_size))
        
        # Verifica posizione della testa (valore 2.0)
        head_x, head_y = self.game.snake[0]
        self.assertEqual(state[head_y, head_x], 2.0)
        
        # Verifica posizione del cibo (valore 3.0)
        food_x, food_y = self.game.food
        self.assertEqual(state[food_y, food_x], 3.0)

    def test_get_normalized_state(self):
        """Verifica che lo stato normalizzato venga generato correttamente."""
        norm_state = self.game.get_normalized_state()
        
        # Verifica che lo stato normalizzato abbia la dimensione corretta
        # 4 (pericoli) + 4 (direzione) + 2 (posizione cibo) = 10
        self.assertEqual(norm_state.shape, (10,))

    def test_turn_directions(self):
        """Verifica che le funzioni di rotazione funzionino correttamente."""
        # Verifichiamo le rotazioni in tutte le direzioni
        self.assertEqual(self.game._turn_right(Direction.RIGHT), Direction.DOWN)
        self.assertEqual(self.game._turn_right(Direction.DOWN), Direction.LEFT)
        self.assertEqual(self.game._turn_right(Direction.LEFT), Direction.UP)
        self.assertEqual(self.game._turn_right(Direction.UP), Direction.RIGHT)
        
        self.assertEqual(self.game._turn_left(Direction.RIGHT), Direction.UP)
        self.assertEqual(self.game._turn_left(Direction.UP), Direction.LEFT)
        self.assertEqual(self.game._turn_left(Direction.LEFT), Direction.DOWN)
        self.assertEqual(self.game._turn_left(Direction.DOWN), Direction.RIGHT)

    def test_collision_detection(self):
        """Verifica che il rilevamento delle collisioni funzioni correttamente."""
        # Collisione con i muri
        self.assertTrue(self.game.is_collision((-1, 0)))  # Fuori a sinistra
        self.assertTrue(self.game.is_collision((self.grid_size, 0)))  # Fuori a destra
        self.assertTrue(self.game.is_collision((0, -1)))  # Fuori in alto
        self.assertTrue(self.game.is_collision((0, self.grid_size)))  # Fuori in basso
        
        # Posizione valida (non collisione)
        center = self.grid_size // 2
        self.assertFalse(self.game.is_collision((center, center+1)))

    def test_movement(self):
        """Verifica che il movimento del serpente funzioni correttamente."""
        # Salviamo la posizione iniziale
        initial_position = self.game.snake[0]
        
        # Impostiamo una direzione nota per il test
        self.game.direction = Direction.RIGHT
        
        # Movimento in avanti (action 0)
        self.game.step(0)
        new_position = self.game.snake[0]
        expected_position = (initial_position[0] + 1, initial_position[1])
        self.assertEqual(new_position, expected_position)
        
        # Giriamo a destra (action 1) - dovrebbe andare verso il basso
        self.game.step(1)
        new_position = self.game.snake[0]
        expected_position = (expected_position[0], expected_position[1] + 1)
        self.assertEqual(new_position, expected_position)
        
        # Giriamo a sinistra (action 2) - dovrebbe tornare a destra
        self.game.step(2)
        new_position = self.game.snake[0]
        expected_position = (expected_position[0] + 1, expected_position[1])
        self.assertEqual(new_position, expected_position)
        
        # Inversione di marcia (action 3) - dovrebbe andare a sinistra
        self.game.step(3)
        new_position = self.game.snake[0]
        expected_position = (expected_position[0] - 1, expected_position[1])
        self.assertEqual(new_position, expected_position)

    def test_eating_food(self):
        """Verifica che mangiare il cibo funzioni correttamente."""
        # Posiziona il serpente e il cibo in posizioni note
        head_x, head_y = self.game.snake[0]
        
        # Posiziona il cibo davanti alla testa del serpente
        if self.game.direction == Direction.RIGHT:
            self.game.food = (head_x + 1, head_y)
        elif self.game.direction == Direction.LEFT:
            self.game.food = (head_x - 1, head_y)
        elif self.game.direction == Direction.DOWN:
            self.game.food = (head_x, head_y + 1)
        elif self.game.direction == Direction.UP:
            self.game.food = (head_x, head_y - 1)
        
        # Lunghezza iniziale del serpente
        initial_length = len(self.game.snake)
        
        # Muovi il serpente verso il cibo
        state, reward, done, info = self.game.step(0)
        
        # Verifica che il punteggio sia incrementato
        self.assertEqual(self.game.score, 1)
        
        # Verifica che il serpente sia cresciuto
        self.assertEqual(len(self.game.snake), initial_length + 1)
        
        # Verifica il reward ottenuto
        self.assertEqual(reward, 10.0)

    def test_game_over(self):
        """Verifica che il gioco termini correttamente in caso di collisione."""
        # Forziamo una collisione posizionando il serpente al bordo
        self.game.snake[0] = (0, 0)
        self.game.direction = Direction.LEFT
        
        # Muovi il serpente contro il muro
        state, reward, done, info = self.game.step(0)
        
        # Verifica che il gioco sia terminato
        self.assertTrue(done)
        
        # Verifica che il reward sia negativo
        self.assertEqual(reward, -10)

    def test_get_score(self):
        """Verifica che il metodo get_score funzioni correttamente."""
        self.assertEqual(self.game.get_score(), 0)
        
        # Modifichiamo manualmente il punteggio
        self.game.score = 5
        self.assertEqual(self.game.get_score(), 5)

    def test_get_snake_positions(self):
        """Verifica che il metodo get_snake_positions funzioni correttamente."""
        positions = self.game.get_snake_positions()
        self.assertEqual(positions, list(self.game.snake))

    def test_get_food_position(self):
        """Verifica che il metodo get_food_position funzioni correttamente."""
        food_position = self.game.get_food_position()
        self.assertEqual(food_position, self.game.food)

    def test_steps_without_food_limit(self):
        """Verifica che il gioco termini dopo troppi passi senza mangiare."""
        # Inizializziamo il numero di passi senza cibo
        self.game.steps_without_food = self.grid_size * 10  # Esattamente al limite
        
        # Posiziona il serpente lontano dai bordi per evitare altre collisioni
        center = self.grid_size // 2
        self.game.snake[0] = (center, center)
        
        # Impostiamo una direzione
        self.game.direction = Direction.RIGHT
        
        # Eseguiamo un passo
        state, reward, done, info = self.game.step(0)
        
        # Debug
        print(f"Done: {done}, Reward: {reward}, Steps without food: {self.game.steps_without_food}")
        
        # Verifica che il gioco sia terminato
        self.assertTrue(done)
        
        # Verifica che il reward sia negativo
        self.assertEqual(reward, -10)

if __name__ == '__main__':
    unittest.main() 