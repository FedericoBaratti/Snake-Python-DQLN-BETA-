#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Snake Game Core
=====================
Implementa la logica di base del gioco Snake con funzionalità migliorate.

Autore: Federico Baratti
Versione: 2.0
"""

import numpy as np
import random
from collections import deque
from enum import Enum
from typing import Tuple, List, Dict, Any, Optional, Deque

class Direction(Enum):
    """Enumerazione delle direzioni possibili per il serpente."""
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class RewardSystem:
    """
    Sistema di ricompense configurabile per il gioco Snake.
    
    Questa classe gestisce le ricompense assegnate in diverse situazioni di gioco,
    consentendo la personalizzazione delle strategie di reward per l'addestramento.
    """
    
    def __init__(self, config: Optional[Dict[str, float]] = None):
        """
        Inizializza il sistema di ricompense con valori predefiniti o personalizzati.
        
        Args:
            config (Dict[str, float], optional): Configurazione personalizzata delle ricompense
        """
        # Valori predefiniti
        self.default_config = {
            "reward_food": 10.0,           # Ricompensa per aver mangiato il cibo
            "reward_death": -10.0,         # Penalità per la morte
            "reward_step": -0.01,          # Piccola penalità per ogni passo (incoraggia l'efficienza)
            "reward_closer_to_food": 0.1,  # Piccola ricompensa per avvicinarsi al cibo
            "reward_farther_from_food": -0.1,  # Piccola penalità per allontanarsi dal cibo
            "reward_circular_movement": -0.5,  # Penalità per movimenti circolari ripetitivi
            "reward_efficient_path": 0.2,   # Ricompensa per seguire un percorso efficiente
            "reward_survival": 0.001,      # Piccola ricompensa per la sopravvivenza
            "reward_growth": 0.5           # Ricompensa per la crescita del serpente
        }
        
        # Aggiorna con la configurazione personalizzata se specificata
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Tracciamento delle posizioni precedenti per rilevare movimenti circolari
        self.position_history = []
        self.max_history_size = 20
    
    def get_food_reward(self) -> float:
        """Ottieni la ricompensa per aver mangiato il cibo."""
        return self.config["reward_food"]
    
    def get_death_reward(self) -> float:
        """Ottieni la penalità per la morte."""
        return self.config["reward_death"]
    
    def get_step_reward(self) -> float:
        """Ottieni la ricompensa/penalità per ogni passo."""
        return self.config["reward_step"]
    
    def get_distance_reward(self, old_distance: float, new_distance: float) -> float:
        """
        Calcola la ricompensa basata sul cambiamento di distanza dal cibo.
        
        Args:
            old_distance (float): Distanza precedente dal cibo
            new_distance (float): Nuova distanza dal cibo
            
        Returns:
            float: Ricompensa calcolata
        """
        if new_distance < old_distance:
            return self.config["reward_closer_to_food"]
        elif new_distance > old_distance:
            return self.config["reward_farther_from_food"]
        return 0.0
    
    def get_survival_reward(self, snake_length: int) -> float:
        """
        Calcola una piccola ricompensa per la sopravvivenza.
        
        Args:
            snake_length (int): Lunghezza attuale del serpente
            
        Returns:
            float: Ricompensa calcolata
        """
        # Più ricompensa per serpenti più lunghi (più difficile sopravvivere)
        survival_factor = min(1.0, snake_length / 20.0)
        return self.config["reward_survival"] * (1.0 + survival_factor)
    
    def check_circular_movement(self, new_position: Tuple[int, int]) -> float:
        """
        Verifica se il serpente sta facendo movimenti circolari ripetitivi.
        
        Args:
            new_position (Tuple[int, int]): Nuova posizione del serpente
            
        Returns:
            float: Penalità se viene rilevato un movimento circolare
        """
        # Aggiorna la cronologia delle posizioni
        self.position_history.append(new_position)
        if len(self.position_history) > self.max_history_size:
            self.position_history.pop(0)
            
        # Verifica movimenti circolari (posizione visitata più volte)
        if len(self.position_history) >= 8:
            position_counts = {}
            for pos in self.position_history:
                position_counts[pos] = position_counts.get(pos, 0) + 1
                
            # Se una posizione è stata visitata più di 2 volte in un breve periodo
            if max(position_counts.values()) > 2:
                return self.config["reward_circular_movement"]
                
        return 0.0
    
    def get_growth_reward(self, snake_length: int) -> float:
        """
        Calcola una ricompensa basata sulla crescita del serpente.
        
        Args:
            snake_length (int): Lunghezza attuale del serpente
            
        Returns:
            float: Ricompensa calcolata
        """
        # Ricompensa base moltiplicata per un fattore che aumenta con la lunghezza
        return self.config["reward_growth"] * min(1.0, snake_length / 10.0)
    
    def reset(self):
        """Reimposta lo stato interno del sistema di ricompense."""
        self.position_history = []

class SnakeGame:
    """
    Classe che implementa la logica del gioco Snake con funzionalità avanzate.
    
    Attributes:
        grid_size (int): Dimensione della griglia di gioco (quadrata)
        reset(): Reimposta il gioco
        step(action): Esegue un'azione e restituisce stato, reward, done, info
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Inizializza il gioco Snake con configurazione personalizzabile.
        
        Args:
            config (Dict[str, Any], optional): Configurazione del gioco
        """
        # Configurazione predefinita
        self.default_config = {
            "grid_size": 20,
            "max_steps_without_food": 100,
            "reward_system": None,  # Usa il sistema di ricompense predefinito
            "food_placement": "random",  # 'random' o 'smart'
            "initial_snake_length": 1,
            "allow_backward": False,  # Se True, permette di girare di 180 gradi
        }
        
        # Configurazione effettiva
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
            
        # Proprietà del gioco
        self.grid_size = self.config["grid_size"]
        self.max_steps_without_food = self.config["max_steps_without_food"]
        self.allow_backward = self.config["allow_backward"]
        
        # Sistema di ricompense
        self.reward_system = self.config["reward_system"] or RewardSystem()
        
        # Inizializza il gioco
        self.reset()
    
    def reset(self):
        """
        Reimposta il gioco a uno stato iniziale.
        
        Returns:
            np.ndarray: Lo stato iniziale del gioco
        """
        # Posizione iniziale del serpente (centro della griglia)
        center = self.grid_size // 2
        
        # Inizializza il serpente con la lunghezza specificata
        self.snake = deque([(center, center)])
        for _ in range(1, self.config["initial_snake_length"]):
            # Aggiungi segmenti in posizione iniziale (saranno mossi al primo step)
            self.snake.append((center, center))
        
        # Direzione iniziale (casuale)
        self.direction = random.choice(list(Direction))
        
        # Genera la prima posizione del cibo
        self.place_food()
        
        # Azzera il punteggio
        self.score = 0
        
        # Passi senza mangiare cibo
        self.steps_without_food = 0
        
        # Contatore totale di passi
        self.total_steps = 0
        
        # Flag che indica se il gioco è terminato
        self.done = False
        
        # Reset del sistema di ricompense
        self.reward_system.reset()
        
        # Genera lo stato corrente
        return self.get_state()
    
    def get_grid_size(self) -> int:
        """Ottieni la dimensione della griglia."""
        return self.grid_size
    
    def place_food(self):
        """
        Posiziona il cibo in base alla strategia configurata.
        """
        head = self.snake[0]
        
        if self.config["food_placement"] == "smart":
            # Strategia intelligente: posiziona il cibo vicino alla testa con probabilità più alta
            self._place_food_smart(head)
        else:
            # Strategia predefinita casuale
            self._place_food_random()
    
    def _place_food_smart(self, head: Tuple[int, int]):
        """
        Posiziona il cibo con una strategia intelligente.
        
        Args:
            head (Tuple[int, int]): Posizione della testa del serpente
        """
        head_x, head_y = head
        
        # Con probabilità 30%, metti il cibo vicino alla testa per aiutare il modello
        if random.random() < 0.30:
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
                
        # Altrimenti, usa il metodo standard
        self._place_food_random()
    
    def _place_food_random(self):
        """Posiziona il cibo in una posizione casuale non occupata dal serpente."""
        while True:
            self.food = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )
            # Verifica che il cibo non sia posizionato sul serpente
            if self.food not in self.snake:
                break
    
    def get_state(self) -> np.ndarray:
        """
        Genera una rappresentazione dello stato del gioco come matrice.
        
        Returns:
            np.ndarray: Array 2D che rappresenta lo stato del gioco
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
    
    def get_normalized_state(self) -> np.ndarray:
        """
        Genera una rappresentazione normalizzata dello stato per il DQN.
        
        Returns:
            np.ndarray: Array che rappresenta lo stato normalizzato del gioco
        """
        # Informazioni sulla direzione (one-hot encoding)
        direction_state = np.zeros(4, dtype=np.float32)
        direction_state[self.direction.value] = 1.0
        
        # Informazioni sulla testa e cibo
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
        
        # Aggiungiamo dati sulla posizione relativa della coda
        tail_x, tail_y = self.snake[-1]
        tail_distance_x = (tail_x - head_x) / self.grid_size
        tail_distance_y = (tail_y - head_y) / self.grid_size
        
        # Feature sul corpo del serpente: lunghezza normalizzata
        snake_length_normalized = len(self.snake) / (self.grid_size * self.grid_size)
        
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
            # Posizione relativa della coda
            tail_distance_x,
            tail_distance_y,
            # Informazioni sul serpente
            snake_length_normalized
        ], dtype=np.float32)
        
        return features
    
    def get_snake_positions(self) -> List[Tuple[int, int]]:
        """
        Ottiene le posizioni dei segmenti del serpente.
        
        Returns:
            List[Tuple[int, int]]: Lista di coordinate (x, y)
        """
        return list(self.snake)
    
    def get_food_position(self) -> Tuple[int, int]:
        """
        Ottiene la posizione del cibo.
        
        Returns:
            Tuple[int, int]: Coordinate (x, y) del cibo
        """
        return self.food
    
    def get_score(self) -> int:
        """
        Ottiene il punteggio attuale.
        
        Returns:
            int: Punteggio
        """
        return self.score
    
    def _get_new_head(self, direction: Direction) -> Tuple[int, int]:
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
    
    def _turn_right(self, direction: Direction) -> Direction:
        """
        Calcola la direzione ruotando di 90 gradi a destra.
        
        Args:
            direction (Direction): Direzione corrente
            
        Returns:
            Direction: Nuova direzione
        """
        if direction == Direction.RIGHT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.RIGHT
    
    def _turn_left(self, direction: Direction) -> Direction:
        """
        Calcola la direzione ruotando di 90 gradi a sinistra.
        
        Args:
            direction (Direction): Direzione corrente
            
        Returns:
            Direction: Nuova direzione
        """
        if direction == Direction.RIGHT:
            return Direction.UP
        elif direction == Direction.UP:
            return Direction.LEFT
        elif direction == Direction.LEFT:
            return Direction.DOWN
        elif direction == Direction.DOWN:
            return Direction.RIGHT
    
    def is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Verifica se la posizione specificata causa una collisione.
        
        Args:
            position (Tuple[int, int]): Posizione da verificare
            
        Returns:
            bool: True se c'è una collisione, False altrimenti
        """
        x, y = position
        
        # Collisione con i bordi
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        
        # Collisione con il corpo del serpente (esclusa la coda se il serpente si sta muovendo)
        if position in list(self.snake)[:-1]:
            return True
            
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Esegue un'azione nel gioco.
        
        Args:
            action (int): 0 = vai dritto, 1 = gira a destra, 2 = gira a sinistra, 3 = gira indietro
            
        Returns:
            tuple:
                observation (np.ndarray): Nuovo stato del gioco
                reward (float): Ricompensa ottenuta
                done (bool): Flag che indica se il gioco è terminato
                info (dict): Informazioni aggiuntive
        """
        # Se il gioco è già terminato, restituisci lo stato finale
        if self.done:
            return self.get_state(), 0.0, True, {
                "score": self.score,
                "steps": self.total_steps
            }
        
        # Determina la nuova direzione in base all'azione
        new_direction = self._get_new_direction(action)
        
        # Calcola la distanza dal cibo prima del movimento
        head = self.snake[0]
        old_distance_to_food = abs(head[0] - self.food[0]) + abs(head[1] - self.food[1])
        
        # Aggiorna la direzione
        self.direction = new_direction
        
        # Calcola la nuova posizione della testa
        new_head = self._get_new_head(self.direction)
        
        # Verifica collisione
        reward = self.reward_system.get_step_reward()  # Ricompensa/penalità base per ogni passo
        self.total_steps += 1
        self.steps_without_food += 1
        
        # Aggiungi una piccola ricompensa per la sopravvivenza
        reward += self.reward_system.get_survival_reward(len(self.snake))
        
        # Controlla se il serpente ha mangiato il cibo
        ate_food = new_head == self.food
        
        if self.is_collision(new_head):
            # Collisione: fine del gioco
            self.done = True
            reward += self.reward_system.get_death_reward()
        else:
            # Nessuna collisione: sposta il serpente
            self.snake.appendleft(new_head)
            
            # Calcola la nuova distanza dal cibo
            new_distance_to_food = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            
            # Ricompensa/penalità basata sulla distanza dal cibo
            reward += self.reward_system.get_distance_reward(old_distance_to_food, new_distance_to_food)
            
            # Controlla se ci sono movimenti circolari
            reward += self.reward_system.check_circular_movement(new_head)
            
            if ate_food:
                # Il serpente ha mangiato il cibo
                self.score += 1
                self.steps_without_food = 0
                reward += self.reward_system.get_food_reward()
                reward += self.reward_system.get_growth_reward(len(self.snake))
                
                # Genera una nuova posizione per il cibo
                self.place_food()
            else:
                # Rimuovi l'ultimo segmento del corpo se non ha mangiato
                self.snake.pop()
            
            # Timeout: troppe mosse senza mangiare
            if self.steps_without_food >= self.max_steps_without_food:
                self.done = True
                reward += self.reward_system.get_death_reward() * 0.5  # Penalità parziale
        
        # Informazioni aggiuntive
        info = {
            "score": self.score,
            "steps": self.total_steps,
            "steps_without_food": self.steps_without_food,
            "ate_food": ate_food
        }
        
        return self.get_state(), reward, self.done, info
    
    def _get_new_direction(self, action: int) -> Direction:
        """
        Determina la nuova direzione in base all'azione.
        
        Args:
            action (int): Azione (0=dritto, 1=destra, 2=sinistra, 3=indietro)
            
        Returns:
            Direction: Nuova direzione
        """
        if action == 0:  # Vai dritto
            return self.direction
        elif action == 1:  # Gira a destra
            return self._turn_right(self.direction)
        elif action == 2:  # Gira a sinistra
            return self._turn_left(self.direction)
        elif action == 3 and self.allow_backward:  # Gira indietro (180 gradi)
            return self._turn_right(self._turn_right(self.direction))
        else:
            # Fallback: vai dritto
            return self.direction
    
    def render_as_string(self) -> str:
        """
        Genera una rappresentazione testuale del gioco.
        
        Returns:
            str: Rappresentazione testuale
        """
        state = self.get_state()
        rows = []
        
        # Riga superiore
        rows.append("+" + "-" * self.grid_size + "+")
        
        # Righe della griglia
        for y in range(self.grid_size):
            row = "|"
            for x in range(self.grid_size):
                cell_value = state[y, x]
                if cell_value == 0:
                    row += " "  # Cella vuota
                elif cell_value == 1:
                    row += "o"  # Corpo del serpente
                elif cell_value == 2:
                    row += "O"  # Testa del serpente
                elif cell_value == 3:
                    row += "X"  # Cibo
            row += "|"
            rows.append(row)
            
        # Riga inferiore
        rows.append("+" + "-" * self.grid_size + "+")
        
        # Aggiungi informazioni sul punteggio
        rows.append(f"Punteggio: {self.score}")
        rows.append(f"Passi: {self.total_steps}")
        
        return "\n".join(rows)


# Test standalone
if __name__ == "__main__":
    # Test del gioco con controllo manuale
    import time
    
    # Configura il gioco
    config = {
        "grid_size": 10,
        "max_steps_without_food": 50,
        "initial_snake_length": 3,
        "allow_backward": True
    }
    
    game = SnakeGame(config)
    
    # Funzione per pulire la console
    def clear_screen():
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    # Loop principale
    action = 0  # Inizialmente vai dritto
    while not game.done:
        clear_screen()
        print(game.render_as_string())
        
        # Input dell'utente
        key = input("Direzione (w=su, a=sinistra, s=giù, d=destra, q=esci): ")
        
        if key == 'q':
            break
        elif key == 'w':
            action = 2 if game.direction == Direction.RIGHT else 1 if game.direction == Direction.LEFT else 0
        elif key == 's':
            action = 1 if game.direction == Direction.RIGHT else 2 if game.direction == Direction.LEFT else 0
        elif key == 'a':
            action = 2 if game.direction == Direction.DOWN else 1 if game.direction == Direction.UP else 0
        elif key == 'd':
            action = 1 if game.direction == Direction.DOWN else 2 if game.direction == Direction.UP else 0
        else:
            action = 0  # Vai dritto
        
        # Esegui l'azione
        _, reward, _, info = game.step(action)
        print(f"Ricompensa: {reward:.2f}")
        time.sleep(0.1)
    
    # Fine del gioco
    clear_screen()
    print(game.render_as_string())
    print("Game Over!") 