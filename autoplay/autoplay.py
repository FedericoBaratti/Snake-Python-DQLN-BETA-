#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Autoplay
=============
Implementa il controller per la modalità autoplay del gioco Snake.

Autore: Federico Baratti
Versione: 1.0
"""

import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple, List

from backend.environment import SnakeEnv
from dqn_agent.dqn_agent import DQNAgent
from backend.utils import get_device


class AutoplayController:
    """
    Controller per la modalità autoplay del gioco Snake.
    
    Utilizza un modello DQN pre-addestrato per selezionare le azioni
    in modo autonomo, fornendo un'interfaccia semplice per l'integrazione
    con l'interfaccia utente.
    """
    
    def __init__(self, env: SnakeEnv, model_complexity: str = "base",
                 checkpoint_path: Optional[str] = None, config: Dict[str, Any] = None):
        """
        Inizializza il controller autoplay.
        
        Args:
            env (SnakeEnv): Ambiente Snake su cui operare
            model_complexity (str): Complessità del modello ('base', 'avanzato', 'complesso', 'perfetto')
            checkpoint_path (str, optional): Percorso del checkpoint del modello.
                Se None, cerca automaticamente l'ultimo checkpoint disponibile.
            config (Dict[str, Any], optional): Configurazione personalizzata
        """
        self.env = env
        self.model_complexity = model_complexity
        self.config = config
        self.device = get_device()
        
        # Ottieni le dimensioni di stato e azione dall'ambiente
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        print(f"Dimensione stato: {state_dim}, Dimensione azione: {action_dim}")
        
        # Carica l'agente
        self.agent = self._load_agent(checkpoint_path, state_dim, action_dim)
        
        # Stato corrente
        self.current_state = None
        self.previous_action = 0  # Default: vai dritto
        
        # Debug info
        self.debug_enabled = True
        
        # Metriche
        self.game_count = 0
        self.total_score = 0
        self.max_score = 0
        self.episode_steps = 0
        self.scores = []
        
        # Inizializza lo stato
        self._reset()
    
    def _load_agent(self, checkpoint_path: Optional[str], 
                   state_dim: int, action_dim: int) -> DQNAgent:
        """
        Carica l'agente DQN dal checkpoint specificato o usa uno nuovo.
        
        Args:
            checkpoint_path (str, optional): Percorso del checkpoint
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione
            
        Returns:
            DQNAgent: Agente DQN caricato o creato
        """
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Caricamento agente dal checkpoint: {checkpoint_path}")
            return DQNAgent.load(checkpoint_path, device=self.device)
        else:
            # Se non è stato specificato un checkpoint o non esiste, cerca l'ultimo
            checkpoint_dir = os.path.join("training", "checkpoints")
            if os.path.exists(checkpoint_dir):
                checkpoints = [
                    os.path.join(checkpoint_dir, f) 
                    for f in os.listdir(checkpoint_dir) 
                    if f.startswith(f"dqn_{self.model_complexity}") and f.endswith(".pt")
                ]
                
                if checkpoints:
                    # Ordina per data di modifica (più recente prima)
                    checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    print(f"Caricamento ultimo checkpoint disponibile: {checkpoints[0]}")
                    return DQNAgent.load(checkpoints[0], device=self.device)
            
            # Se non troviamo checkpoint, crea un nuovo agente
            print(f"Nessun checkpoint trovato, creazione nuovo agente di complessità '{self.model_complexity}'")
            return DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                complexity=self.model_complexity,
                config=self.config
            )
    
    def _reset(self) -> np.ndarray:
        """
        Reimposta lo stato.
        
        Returns:
            np.ndarray: Nuovo stato iniziale
        """
        # Resetta l'ambiente
        state, _ = self.env.reset()
        self.current_state = state
        self.episode_steps = 0
        return state
    
    def get_action(self) -> int:
        """
        Ottiene l'azione dal modello.
        
        Returns:
            int: Azione da eseguire (0: dritto, 1: destra, 2: sinistra, 3: indietro)
        """
        if self.current_state is None:
            self._reset()
        
        # Verifica la dimensione dello stato
        if self.debug_enabled and hasattr(self, 'agent') and self.episode_steps == 0:
            expected_dim = self.agent.state_dim
            actual_dim = len(self.current_state)
            print(f"Dimensione stato attesa: {expected_dim}, dimensione stato attuale: {actual_dim}")
            print(f"Stato corrente: {self.current_state}")
        
        # Ottieni l'azione dall'agente (usa epsilon=0 per disabilitare l'esplorazione)
        action = self.agent.select_action(self.current_state, training=False)
        
        # Debug info per comprendere le decisioni del modello
        if self.debug_enabled and self.episode_steps % 10 == 0:
            head = self.env.snake_game.snake[0]
            food = self.env.snake_game.food
            print(f"Posizione testa: {head}, Posizione cibo: {food}, Distanza: {abs(head[0]-food[0]) + abs(head[1]-food[1])}")
            print(f"Azione scelta: {action}")
        
        # Aggiorna lo stato in base all'azione
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Aggiorna lo stato corrente
        self.current_state = next_state
        self.previous_action = action
        self.episode_steps += 1
        
        # Se l'episodio è terminato, aggiorna le statistiche e resetta
        if done:
            score = info.get('score', 0)
            self.game_count += 1
            self.total_score += score
            self.scores.append(score)
            if score > self.max_score:
                self.max_score = score
            
            # Debug info quando termina un episodio
            if self.debug_enabled:
                print(f"Episodio {self.game_count} terminato. Punteggio: {score}, Passi: {self.episode_steps}")
            
            # Resetta per il prossimo episodio
            self._reset()
        
        return action
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche della modalità autoplay.
        
        Returns:
            Dict[str, Any]: Statistiche della modalità autoplay
        """
        avg_score = self.total_score / max(1, self.game_count)
        return {
            "games_played": self.game_count,
            "total_score": self.total_score,
            "max_score": self.max_score,
            "average_score": avg_score,
            "model_complexity": self.model_complexity,
            "current_step": self.episode_steps,
            "last_scores": self.scores[-10:] if self.scores else []
        }


if __name__ == "__main__":
    # Test del controller autoplay
    from backend.snake_game import SnakeGame
    
    # Crea gioco e ambiente
    game = SnakeGame(grid_size=10)
    env = SnakeEnv(game)
    
    # Crea controller
    controller = AutoplayController(env, model_complexity="base")
    
    # Simula alcuni episodi
    total_steps = 0
    max_steps = 1000
    
    print("Inizio simulazione autoplay:")
    while total_steps < max_steps:
        # Ottieni l'azione
        action = controller.get_action()
        
        # Stampa ogni 100 passi
        total_steps += 1
        if total_steps % 100 == 0:
            stats = controller.get_stats()
            print(f"Passo {total_steps}:")
            print(f"- Partite giocate: {stats['games_played']}")
            print(f"- Punteggio massimo: {stats['max_score']}")
            print(f"- Punteggio medio: {stats['average_score']:.2f}")
    
    # Stampa statistiche finali
    stats = controller.get_stats()
    print("\nStatistiche finali:")
    print(f"- Partite giocate: {stats['games_played']}")
    print(f"- Punteggio totale: {stats['total_score']}")
    print(f"- Punteggio massimo: {stats['max_score']}")
    print(f"- Punteggio medio: {stats['average_score']:.2f}")
    print(f"- Ultimi punteggi: {stats['last_scores']}") 