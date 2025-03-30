#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Ambiente Gym
==================
Implementa un ambiente compatibile con l'API Gym per il gioco Snake.

Autore: Federico Baratti
Versione: 1.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from backend.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    """
    Ambiente Gym per il gioco Snake.
    
    Implementa l'interfaccia standard di Gym per permettere
    al DQN di interagire con il gioco Snake in modo standardizzato.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self, snake_game=None, grid_size=20, use_normalized_state=True):
        """
        Inizializza l'ambiente Gym per Snake.
        
        Args:
            snake_game (SnakeGame, optional): Istanza del gioco Snake da utilizzare.
                Se None, ne viene creata una nuova.
            grid_size (int): Dimensione della griglia di gioco se snake_game è None.
            use_normalized_state (bool): Se True, usa lo stato normalizzato per DQN.
        """
        super(SnakeEnv, self).__init__()
        
        # Crea o utilizza un'istanza di SnakeGame
        self.snake_game = snake_game if snake_game else SnakeGame(grid_size=grid_size)
        self.grid_size = self.snake_game.get_grid_size()
        self.use_normalized_state = use_normalized_state
        
        # Definisci lo spazio delle azioni: 0 (dritto), 1 (destra), 2 (sinistra)
        self.action_space = spaces.Discrete(3)
        
        if use_normalized_state:
            # Stato normalizzato: vettore di caratteristiche
            # [pericoli (3), direzione (4), distanza cibo (2)]
            self.observation_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(9,), 
                dtype=np.float32
            )
        else:
            # Stato completo: griglia del gioco
            self.observation_space = spaces.Box(
                low=0, 
                high=3, 
                shape=(self.grid_size, self.grid_size), 
                dtype=np.float32
            )
    
    def step(self, action):
        """
        Esegue un'azione nell'ambiente.
        
        Args:
            action (int): Azione da eseguire (0: dritto, 1: destra, 2: sinistra)
            
        Returns:
            tuple:
                observation (np.ndarray): Stato osservato dopo l'azione
                reward (float): Reward ottenuto
                terminated (bool): Flag che indica se l'episodio è terminato
                truncated (bool): Flag che indica se l'episodio è stato troncato 
                                  (non usato in questo ambiente)
                info (dict): Informazioni aggiuntive
        """
        # Esegui il passo nel gioco
        state, reward, done, info = self.snake_game.step(action)
        
        # Trasforma lo stato se necessario
        if self.use_normalized_state:
            observation = self.snake_game.get_normalized_state()
        else:
            observation = state
        
        # In Gym, terminated indica la terminazione naturale dell'episodio (es. collisione)
        # truncated indica l'interruzione prematura per limiti di tempo/step (non usato qui)
        return observation, reward, done, False, info
    
    def reset(self, seed=None, options=None):
        """
        Reimposta l'ambiente a uno stato iniziale.
        
        Args:
            seed (int, optional): Seed per la generazione casuale
            options (dict, optional): Opzioni aggiuntive per il reset
            
        Returns:
            tuple:
                observation (np.ndarray): Stato iniziale osservato
                info (dict): Informazioni aggiuntive
        """
        # Reimposta il generatore di numeri casuali se il seed è specificato
        if seed is not None:
            np.random.seed(seed)
        
        # Reimposta il gioco
        state = self.snake_game.reset()
        
        # Trasforma lo stato se necessario
        if self.use_normalized_state:
            observation = self.snake_game.get_normalized_state()
        else:
            observation = state
        
        return observation, {}
    
    def render(self):
        """
        Rendering dell'ambiente (non implementato direttamente qui).
        
        In questa implementazione, il rendering è gestito dalla UI esterna.
        Questo metodo è mantenuto per compatibilità con l'API Gym.
        
        Returns:
            None
        """
        pass
    
    def close(self):
        """
        Chiude l'ambiente e libera le risorse.
        
        Returns:
            None
        """
        pass
    
    def get_valid_actions(self):
        """
        Restituisce la lista di azioni valide nello stato corrente.
        
        Utile per implementazioni che potrebbero limitare le azioni
        possibili in determinati stati.
        
        Returns:
            list: Lista di azioni valide
        """
        # Nel nostro caso tutte le azioni sono sempre valide
        return [0, 1, 2] 