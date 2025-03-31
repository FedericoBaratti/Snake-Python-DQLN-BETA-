#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Ambiente Sintetico
=======================
Implementa un ambiente sintetico per il preaddestramento del modello DQN.

Autore: Federico Baratti
Versione: 1.0
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import random
from typing import Tuple, Dict, Any, Optional, List

class SyntheticSnakeEnv(gym.Env):
    """
    Ambiente sintetico per il preaddestramento del DQN per Snake.
    
    Questo ambiente è una versione semplificata del gioco Snake, ottimizzata
    per produrre rapidamente esperienze di training utili. Utilizza:
    - Una griglia più piccola per ridurre la complessità
    - Reward engineering per guidare l'apprendimento
    - Generazione di stati più frequentemente vicini al cibo
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, grid_size: int = 10, max_steps: int = 100):
        """
        Inizializza l'ambiente sintetico.
        
        Args:
            grid_size (int): Dimensione della griglia (quadrata)
            max_steps (int): Numero massimo di passi per episodio
        """
        super(SyntheticSnakeEnv, self).__init__()
        
        # Parametri della griglia
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Stato del gioco
        self.snake = None
        self.food = None
        self.steps_taken = 0
        self.done = False
        self.score = 0
        self.direction = None
        
        # Direzioni (0: destra, 1: giù, 2: sinistra, 3: su)
        self.directions = [
            (1, 0),   # destra
            (0, 1),   # giù
            (-1, 0),  # sinistra
            (0, -1)   # su
        ]
        
        # Definisci lo spazio delle azioni: 0 (dritto), 1 (destra), 2 (sinistra), 3 (indietro)
        self.action_space = spaces.Discrete(4)
        
        # Spazio di osservazione: vettore di caratteristiche
        # [pericoli (4), direzione (4), distanza cibo (2)]
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(10,), 
            dtype=np.float32
        )
        
        # Inizializza lo stato
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reimposta l'ambiente a uno stato iniziale.
        
        Args:
            seed (int, optional): Seed per la generazione casuale
            options (dict, optional): Opzioni aggiuntive
        
        Returns:
            tuple: Stato osservato e dizionario info
        """
        # Reimposta il generatore di numeri casuali
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Inizializza il serpente (inizia con lunghezza 1)
        start_x, start_y = self._get_random_position()
        self.snake = deque([(start_x, start_y)])
        
        # Imposta una direzione iniziale casuale
        self.direction = random.randint(0, 3)
        
        # Posiziona il cibo
        self._place_food()
        
        # Reimposta altre variabili
        self.steps_taken = 0
        self.done = False
        self.score = 0
        
        # Genera lo stato
        observation = self._get_observation()
        
        return observation, {}
    
    def _get_random_position(self) -> Tuple[int, int]:
        """
        Genera una posizione casuale nella griglia.
        
        Returns:
            tuple: Posizione (x, y)
        """
        return (
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1)
        )
    
    def _place_food(self) -> None:
        """
        Posiziona il cibo in una posizione non occupata dal serpente.
        
        Con una probabilità del 20%, posiziona il cibo vicino alla testa
        del serpente per incentivare l'apprendimento di comportamenti utili.
        """
        if self.snake and random.random() < 0.2:
            # Posiziona il cibo vicino alla testa (raggio 3)
            head_x, head_y = self.snake[0]
            while True:
                offset_x = random.randint(-3, 3)
                offset_y = random.randint(-3, 3)
                food_x = (head_x + offset_x) % self.grid_size
                food_y = (head_y + offset_y) % self.grid_size
                
                # Verifica che il cibo non sia sul serpente
                if (food_x, food_y) not in self.snake:
                    self.food = (food_x, food_y)
                    return
        
        # Altrimenti, posiziona il cibo in modo casuale
        while True:
            self.food = self._get_random_position()
            if self.food not in self.snake:
                break
    
    def _get_head_position(self) -> Tuple[int, int]:
        """
        Restituisce la posizione della testa del serpente.
        
        Returns:
            tuple: Posizione della testa (x, y)
        """
        return self.snake[0]
    
    def _is_collision(self, position: Tuple[int, int]) -> bool:
        """
        Verifica se c'è una collisione con i muri o con il corpo del serpente.
        
        Args:
            position (tuple): Posizione da controllare
        
        Returns:
            bool: True se c'è una collisione, False altrimenti
        """
        x, y = position
        
        # Collisione con i muri
        if x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size:
            return True
        
        # Collisione con il corpo del serpente (esclusa la testa)
        if position in list(self.snake)[1:]:
            return True
        
        return False
    
    def _get_new_direction(self, action: int) -> int:
        """
        Calcola la nuova direzione in base all'azione.
        
        Args:
            action (int): Azione (0: dritto, 1: destra, 2: sinistra, 3: indietro)
        
        Returns:
            int: Nuova direzione (0: destra, 1: giù, 2: sinistra, 3: su)
        """
        if action == 0:  # Dritto
            return self.direction
        elif action == 1:  # Gira a destra
            return (self.direction + 1) % 4
        elif action == 2:  # Gira a sinistra
            return (self.direction - 1) % 4
        elif action == 3:  # Gira indietro
            return (self.direction + 2) % 4
        else:
            return self.direction
    
    def _move_snake(self, direction: int) -> Tuple[bool, bool]:
        """
        Muove il serpente nella direzione specificata.
        
        Args:
            direction (int): Direzione (0: destra, 1: giù, 2: sinistra, 3: su)
        
        Returns:
            tuple: (ha_mangiato, ha_colliso)
        """
        # Ottieni la direzione come vettore
        dir_x, dir_y = self.directions[direction]
        
        # Calcola la nuova posizione della testa
        old_head = self._get_head_position()
        new_head = (
            (old_head[0] + dir_x) % self.grid_size,  # Wrap around per la griglia sintetica
            (old_head[1] + dir_y) % self.grid_size
        )
        
        # Verifica se c'è una collisione (solo con il corpo, non con i muri in griglia sintetica)
        has_collision = new_head in list(self.snake)[1:]
        
        # Se c'è una collisione, non muovere il serpente
        if has_collision:
            return False, True
        
        # Aggiungi la nuova testa
        self.snake.appendleft(new_head)
        
        # Verifica se il serpente ha mangiato il cibo
        has_eaten = new_head == self.food
        
        if has_eaten:
            # Incrementa il punteggio e genera nuovo cibo
            self.score += 1
            self._place_food()
        else:
            # Rimuovi l'ultima parte del serpente se non ha mangiato
            self.snake.pop()
        
        return has_eaten, False
    
    def _get_observation(self) -> np.ndarray:
        """
        Genera una rappresentazione dell'osservazione.
        
        Returns:
            np.ndarray: Vettore delle caratteristiche
        """
        head_x, head_y = self._get_head_position()
        food_x, food_y = self.food
        
        # Distanza normalizzata dal cibo
        food_dx = (food_x - head_x) / self.grid_size
        food_dy = (food_y - head_y) / self.grid_size
        
        # Pericoli nelle quattro direzioni (davanti, destra, sinistra, indietro)
        dangers = np.zeros(4, dtype=np.float32)
        
        # Controlla il pericolo davanti
        front_dir = self.direction
        front_x, front_y = self.directions[front_dir]
        front_pos = ((head_x + front_x) % self.grid_size, (head_y + front_y) % self.grid_size)
        dangers[0] = front_pos in list(self.snake)[1:]
        
        # Controlla il pericolo a destra
        right_dir = (self.direction + 1) % 4
        right_x, right_y = self.directions[right_dir]
        right_pos = ((head_x + right_x) % self.grid_size, (head_y + right_y) % self.grid_size)
        dangers[1] = right_pos in list(self.snake)[1:]
        
        # Controlla il pericolo a sinistra
        left_dir = (self.direction - 1) % 4
        left_x, left_y = self.directions[left_dir]
        left_pos = ((head_x + left_x) % self.grid_size, (head_y + left_y) % self.grid_size)
        dangers[2] = left_pos in list(self.snake)[1:]
        
        # Controlla il pericolo dietro
        back_dir = (self.direction + 2) % 4
        back_x, back_y = self.directions[back_dir]
        back_pos = ((head_x + back_x) % self.grid_size, (head_y + back_y) % self.grid_size)
        dangers[3] = back_pos in list(self.snake)[1:]
        
        # Direzione corrente (one-hot encoding)
        dir_one_hot = np.zeros(4, dtype=np.float32)
        dir_one_hot[self.direction] = 1.0
        
        # Combina tutte le caratteristiche
        observation = np.concatenate([
            dangers,
            dir_one_hot,
            [food_dx, food_dy]
        ])
        
        return observation
    
    def _calculate_reward(self, has_eaten: bool, has_collided: bool) -> float:
        """
        Calcola il reward per l'azione corrente.
        
        Usa reward engineering per accelerare l'apprendimento:
        - Reward positivo grande per mangiare cibo
        - Reward negativo grande per collisioni
        - Piccolo reward positivo per avvicinarsi al cibo
        - Piccolo reward negativo per allontanarsi dal cibo
        
        Args:
            has_eaten (bool): Se il serpente ha mangiato il cibo
            has_collided (bool): Se il serpente ha avuto una collisione
        
        Returns:
            float: Reward
        """
        if has_collided:
            return -10.0
        
        if has_eaten:
            return 10.0
        
        head_x, head_y = self._get_head_position()
        food_x, food_y = self.food
        
        # Calcola la distanza Manhattan dal cibo
        current_distance = abs(head_x - food_x) + abs(head_y - food_y)
        
        # Trova la posizione precedente della testa
        prev_dir_x, prev_dir_y = self.directions[self.direction]
        prev_head_x = (head_x - prev_dir_x) % self.grid_size
        prev_head_y = (head_y - prev_dir_y) % self.grid_size
        prev_distance = abs(prev_head_x - food_x) + abs(prev_head_y - food_y)
        
        # Reward per avvicinarsi/allontanarsi dal cibo
        if current_distance < prev_distance:
            return 0.1  # Si è avvicinato al cibo
        elif current_distance > prev_distance:
            return -0.1  # Si è allontanato dal cibo
        
        return -0.01  # Piccola penalità per incentivare l'efficienza
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un passo nell'ambiente.
        
        Args:
            action (int): Azione da eseguire (0: dritto, 1: destra, 2: sinistra, 3: indietro)
        
        Returns:
            tuple: Osservazione, reward, terminato, troncato, info
        """
        if self.done:
            observation = self._get_observation()
            return observation, 0.0, True, False, {"score": self.score}
        
        # Aggiorna i passi
        self.steps_taken += 1
        
        # Calcola la nuova direzione
        self.direction = self._get_new_direction(action)
        
        # Muovi il serpente
        has_eaten, has_collided = self._move_snake(self.direction)
        
        # Calcola il reward
        reward = self._calculate_reward(has_eaten, has_collided)
        
        # Verifica se l'episodio è terminato
        terminated = has_collided or self.steps_taken >= self.max_steps
        self.done = terminated
        
        # Genera l'osservazione
        observation = self._get_observation()
        
        # Info aggiuntive
        info = {"score": self.score}
        
        return observation, reward, terminated, False, info
    
    def render(self):
        """
        Rendering dell'ambiente (non implementato).
        
        Questo ambiente è pensato per il training, non per la visualizzazione.
        """
        pass
    
    def close(self):
        """Chiude l'ambiente."""
        pass


# Classe per generare batch di esperienze sintetiche
class SyntheticExperienceGenerator:
    """
    Generatore di esperienze sintetiche per il preaddestramento.
    
    Genera batch di esperienze da utilizzare per il preaddestramento
    dell'agente DQN, accelerando la fase iniziale dell'apprendimento.
    """
    
    def __init__(self, grid_size: int = 10, max_steps: int = 100):
        """
        Inizializza il generatore di esperienze.
        
        Args:
            grid_size (int): Dimensione della griglia
            max_steps (int): Numero massimo di passi per episodio
        """
        self.env = SyntheticSnakeEnv(grid_size=grid_size, max_steps=max_steps)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
    
    def generate_episode(self) -> List[Tuple[np.ndarray, int, np.ndarray, float, bool]]:
        """
        Genera un episodio completo di esperienze.
        
        Returns:
            list: Lista di tuple (stato, azione, prossimo stato, reward, done)
        """
        experiences = []
        state, _ = self.env.reset()
        done = False
        
        while not done:
            # Scegli un'azione casuale (per la generazione di dati)
            action = random.randint(0, self.action_dim - 1)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Memorizza l'esperienza
            experiences.append((state, action, next_state, reward, done))
            
            # Aggiorna lo stato
            state = next_state
        
        return experiences
    
    def generate_batch(self, batch_size: int) -> List[Tuple[np.ndarray, int, np.ndarray, float, bool]]:
        """
        Genera un batch di esperienze.
        
        Args:
            batch_size (int): Dimensione del batch
        
        Returns:
            list: Batch di esperienze
        """
        experiences = []
        
        # Genera episodi finché non abbiamo abbastanza esperienze
        while len(experiences) < batch_size:
            episode_experiences = self.generate_episode()
            experiences.extend(episode_experiences)
        
        # Limita il batch alla dimensione richiesta
        return experiences[:batch_size]


if __name__ == "__main__":
    # Test dell'ambiente sintetico
    env = SyntheticSnakeEnv(grid_size=10, max_steps=100)
    generator = SyntheticExperienceGenerator(grid_size=10, max_steps=100)
    
    # Genera un episodio
    print("Generazione di un episodio sintetico...")
    episode = generator.generate_episode()
    print(f"Episodio generato con {len(episode)} esperienze")
    
    # Genera un batch
    batch_size = 1000
    print(f"Generazione di un batch di {batch_size} esperienze...")
    batch = generator.generate_batch(batch_size)
    print(f"Batch generato con {len(batch)} esperienze")
    
    # Analizza le ricompense
    rewards = [exp[3] for exp in batch]
    print(f"Statistiche delle ricompense:")
    print(f"- Min: {min(rewards)}")
    print(f"- Max: {max(rewards)}")
    print(f"- Media: {sum(rewards) / len(rewards)}")
    
    # Statistiche sui terminali
    terminal_states = sum(exp[4] for exp in batch)
    print(f"Stati terminali: {terminal_states} ({terminal_states / len(batch) * 100:.2f}%)") 