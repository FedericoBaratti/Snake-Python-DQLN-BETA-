#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Snake Game
================
Implementa la logica di base del gioco Snake.

Autore: Federico Baratti
Versione: 1.0
"""

import numpy as np
import random
from collections import deque
from enum import Enum

class Direction(Enum):
    """Enumerazione delle direzioni possibili per il serpente."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class SnakeGame:
    """
    Classe che implementa la logica del gioco Snake.
    
    Attributes:
        grid_size (int): Dimensione della griglia di gioco (quadrata)
        reset(): Reimposta il gioco
        step(action): Esegue un'azione e restituisce stato, reward, done, info
    """
    
    def __init__(self, grid_size=20):
        """
        Inizializza il gioco Snake.
        
        Args:
            grid_size (int): Dimensione della griglia di gioco (quadrata)
        """
        self.grid_size = grid_size
        self.reset()
    
    def reset(self):
        """
        Reimposta il gioco a uno stato iniziale.
        
        Returns:
            np.ndarray: Lo stato iniziale del gioco
        """
        # Posizione iniziale del serpente (centro della griglia)
        center = self.grid_size // 2
        self.snake = deque([(center, center)])
        
        # Direzione iniziale (casuale)
        self.direction = random.choice(list(Direction))
        
        # Genera la prima posizione del cibo
        self.place_food()
        
        # Azzera il punteggio
        self.score = 0
        
        # Passi senza mangiare cibo
        self.steps_without_food = 0
        
        # Flag che indica se il gioco è terminato
        self.done = False
        
        # Genera lo stato corrente
        return self.get_state()
    
    def place_food(self):
        """Posiziona il cibo in una posizione casuale non occupata dal serpente.
           Con una probabilità del 25%, posiziona il cibo vicino alla testa del serpente
           per facilitare l'apprendimento del modello."""
        head = self.snake[0]
        head_x, head_y = head
        
        # Con probabilità 25%, metti il cibo vicino alla testa per aiutare il modello
        if random.random() < 0.25:
            max_attempts = 15
            attempts = 0
            while attempts < max_attempts:
                # Genera un offset casuale tra -5 e 5 dalla testa
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                
                # Calcola la posizione candidata per il cibo
                food_x = max(0, min(self.grid_size - 1, head_x + offset_x))
                food_y = max(0, min(self.grid_size - 1, head_y + offset_y))
                
                self.food = (food_x, food_y)
                
                # Verifica che il cibo non sia posizionato sul serpente
                if self.food not in self.snake:
                    return
                attempts += 1
                
        # Altrimenti, usa il metodo standard per posizionare il cibo
        while True:
            self.food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            # Verifica che il cibo non sia posizionato sul serpente
            if self.food not in self.snake:
                break
    
    def get_state(self):
        """
        Genera una rappresentazione dello stato del gioco.
        
        Returns:
            np.ndarray: Array che rappresenta lo stato del gioco
        """
        # Crea una griglia vuota
        state = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        # Segna la posizione della testa del serpente
        head = self.snake[0]
        state[head[1], head[0]] = 2.0
        
        # Segna il corpo del serpente
        for segment in list(self.snake)[1:]:
            state[segment[1], segment[0]] = 1.0
        
        # Segna la posizione del cibo
        state[self.food[1], self.food[0]] = 3.0
        
        return state
    
    def get_normalized_state(self):
        """
        Genera una rappresentazione normalizzata dello stato per il DQN.
        
        Returns:
            np.ndarray: Array che rappresenta lo stato normalizzato del gioco
        """
        # Ottiene lo stato di base
        state = self.get_state()
        
        # Informazioni sulla direzione (one-hot encoding)
        direction_state = np.zeros(4, dtype=np.float32)
        direction_state[self.direction.value] = 1.0
        
        # Informazioni sulla testa, cibo e pericoli nelle 4 direzioni
        head = self.snake[0]
        head_x, head_y = head
        
        # Distanza dal cibo (normalizzata)
        food_x, food_y = self.food
        food_distance_x = (food_x - head_x) / self.grid_size
        food_distance_y = (food_y - head_y) / self.grid_size
        
        # Pericoli nelle 4 direzioni (muri o corpo del serpente)
        danger_straight = self.is_collision(self._get_new_head(self.direction))
        danger_right = self.is_collision(self._get_new_head(self._turn_right(self.direction)))
        danger_left = self.is_collision(self._get_new_head(self._turn_left(self.direction)))
        # Aggiungiamo il pericolo quando si gira indietro (180 gradi)
        back_direction = self._turn_right(self._turn_right(self.direction))
        danger_back = self.is_collision(self._get_new_head(back_direction))
        
        # Crea un vettore di features
        features = np.array([
            # Pericoli
            danger_straight,
            danger_right,
            danger_left,
            danger_back,
            # Direzione corrente
            *direction_state,
            # Posizione relativa del cibo
            food_distance_x,
            food_distance_y,
        ], dtype=np.float32)
        
        return features
    
    def _get_new_head(self, direction):
        """
        Calcola la nuova posizione della testa in base alla direzione.
        
        Args:
            direction (Direction): La direzione in cui muoversi
            
        Returns:
            tuple: Nuova posizione della testa (x, y)
        """
        head = self.snake[0]
        head_x, head_y = head
        
        if direction == Direction.RIGHT:
            return (head_x + 1, head_y)
        elif direction == Direction.LEFT:
            return (head_x - 1, head_y)
        elif direction == Direction.DOWN:
            return (head_x, head_y + 1)
        elif direction == Direction.UP:
            return (head_x, head_y - 1)
    
    def _turn_right(self, direction):
        """Calcola la direzione ruotando a destra."""
        return Direction((direction.value + 1) % 4)
    
    def _turn_left(self, direction):
        """Calcola la direzione ruotando a sinistra."""
        return Direction((direction.value - 1) % 4)
    
    def is_collision(self, position=None):
        """
        Verifica se c'è una collisione con i muri o con il corpo del serpente.
        
        Args:
            position (tuple, optional): Posizione da controllare. Se None, usa la testa corrente.
            
        Returns:
            bool: True se c'è una collisione, False altrimenti
        """
        if position is None:
            position = self.snake[0]
        
        x, y = position
        
        # Collisione con i muri
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        
        # Collisione con il corpo del serpente (esclusa la testa)
        if position in list(self.snake)[1:]:
            return True
        
        return False
    
    def step(self, action):
        """
        Esegue un'azione nel gioco.
        
        Args:
            action (int): Azione da eseguire:
                0 - Vai dritto
                1 - Gira a destra
                2 - Gira a sinistra
                3 - Vai indietro (gira di 180 gradi)
        
        Returns:
            tuple:
                np.ndarray: Nuovo stato
                float: Reward ottenuto
                bool: Flag che indica se il gioco è terminato
                dict: Informazioni aggiuntive
        """
        self.steps_without_food += 1
        
        # Se il gioco è già terminato, restituisci lo stato corrente
        if self.done:
            return self.get_state(), 0, True, {}
        
        # Determina la nuova direzione in base all'azione
        if action == 1:  # Gira a destra
            self.direction = self._turn_right(self.direction)
        elif action == 2:  # Gira a sinistra
            self.direction = self._turn_left(self.direction)
        elif action == 3:  # Gira di 180 gradi (indietro)
            self.direction = self._turn_right(self._turn_right(self.direction))
        # Se action == 0, continua dritto (nessun cambio di direzione)
        
        # Calcola la nuova posizione della testa
        new_head = self._get_new_head(self.direction)
        
        # Verifica se c'è una collisione
        reward = 0
        if self.is_collision(new_head):
            self.done = True
            reward = -10  # Penalità per la collisione
            return self.get_state(), reward, True, {"score": self.score}
        
        # Aggiungi la nuova testa
        self.snake.appendleft(new_head)
        
        # Verifica se il serpente ha mangiato il cibo
        if new_head == self.food:
            self.score += 1
            reward = 10.0  # Reward positivo per aver mangiato
            self.steps_without_food = 0
            self.place_food()
        else:
            # Rimuovi l'ultima parte del corpo se non ha mangiato
            self.snake.pop()
            
            # Piccolo reward negativo per ogni passo senza cibo per incoraggiare efficienza
            reward = -0.05
            
            # Calcola la distanza dal cibo
            head = self.snake[0]
            old_distance = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
            new_distance = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            
            # Reward più forte per avvicinarsi al cibo
            if new_distance < old_distance:
                reward += 0.2
            elif new_distance > old_distance:
                reward -= 0.1  # Penalità per allontanarsi dal cibo
            
            # Penalità se il serpente sta girando in tondo senza mangiare
            if self.steps_without_food > self.grid_size * 2:
                reward -= 0.3
        
        # Termina il gioco se è troppo lungo senza mangiare
        if self.steps_without_food > self.grid_size * 10:
            self.done = True
            reward = -10  # Penalità per aver girato in tondo
        
        return self.get_state(), reward, self.done, {"score": self.score}
    
    def get_score(self):
        """Restituisce il punteggio corrente."""
        return self.score
    
    def get_grid_size(self):
        """Restituisce la dimensione della griglia."""
        return self.grid_size
    
    def get_snake_positions(self):
        """Restituisce le posizioni correnti del serpente."""
        return list(self.snake)
    
    def get_food_position(self):
        """Restituisce la posizione corrente del cibo."""
        return self.food 