#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Ambiente Gym
==================
Implementa un ambiente compatibile con l'API Gym per il gioco Snake.

Autore: Federico Baratti
Versione: 2.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple, Optional, Union, List

from core.snake_game import SnakeGame

class SnakeEnv(gym.Env):
    """
    Ambiente Gym per il gioco Snake.
    
    Implementa l'interfaccia standard di Gym per permettere
    al DQN di interagire con il gioco Snake in modo standardizzato.
    Supporta configurazioni avanzate e multiple rappresentazioni dello stato.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array', 'ascii']}
    
    def __init__(self, snake_game: Optional[SnakeGame] = None, 
                 config: Optional[Dict[str, Any]] = None, 
                 use_normalized_state: bool = True):
        """
        Inizializza l'ambiente Gym per Snake con configurazione avanzata.
        
        Args:
            snake_game (SnakeGame, optional): Istanza del gioco Snake da utilizzare.
                Se None, ne viene creata una nuova.
            config (Dict[str, Any], optional): Configurazione personalizzata per il gioco
                se snake_game è None.
            use_normalized_state (bool): Se True, usa lo stato normalizzato per DQN.
        """
        super(SnakeEnv, self).__init__()
        
        # Crea o utilizza un'istanza di SnakeGame
        self.snake_game = snake_game if snake_game else SnakeGame(config)
        self.grid_size = self.snake_game.get_grid_size()
        self.use_normalized_state = use_normalized_state
        
        # Definisci lo spazio delle azioni: 0 (dritto), 1 (destra), 2 (sinistra), 3 (indietro)
        self.action_space = spaces.Discrete(4)
        
        if use_normalized_state:
            # Stato normalizzato: vettore di caratteristiche
            # Include pericoli, direzione, distanza cibo, informazioni sul serpente
            self.observation_space = spaces.Box(
                low=-1.0, 
                high=1.0, 
                shape=(13,),  # Dimensione dipende dalle feature estratte
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
        
        # Statistiche per monitoraggio delle performance
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.total_episodes = 0
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Esegue un'azione nell'ambiente.
        
        Args:
            action (int): Azione da eseguire (0: dritto, 1: destra, 2: sinistra, 3: indietro)
            
        Returns:
            tuple:
                observation (np.ndarray): Stato osservato dopo l'azione
                reward (float): Reward ottenuto
                terminated (bool): Flag che indica se l'episodio è terminato
                truncated (bool): Flag che indica se l'episodio è stato troncato
                info (dict): Informazioni aggiuntive
        """
        # Esegui il passo nel gioco
        state, reward, done, info = self.snake_game.step(action)
        
        # Aggiorna le statistiche dell'episodio corrente
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Trasforma lo stato se necessario
        if self.use_normalized_state:
            observation = self.snake_game.get_normalized_state()
        else:
            observation = state
        
        # Verifica che l'osservazione sia nel formato corretto
        if isinstance(observation, np.ndarray) and len(observation.shape) == 1:
            # Corretto: [state_dim]
            pass
        elif isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            # Più dimensioni, appiattisci
            observation = observation.reshape(-1)
        else:
            # Non è un array numpy, converti
            observation = np.array(observation, dtype=np.float32)
        
        # Aggiungi informazioni sulle statistiche dell'episodio corrente
        info.update({
            "episode_reward": self.current_episode_reward,
            "episode_length": self.current_episode_length,
            "total_episodes": self.total_episodes
        })
        
        # In Gym, terminated indica la terminazione naturale dell'episodio (es. collisione)
        # truncated indica l'interruzione prematura per limiti di tempo/step (non usato qui)
        return observation, reward, done, False, info
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        
        # Se questo non è il primo episodio, aggiorna le statistiche
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.total_episodes += 1
        
        # Reimposta il contatore dell'episodio corrente
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
        # Reimposta il gioco
        state = self.snake_game.reset()
        
        # Trasforma lo stato se necessario
        if self.use_normalized_state:
            observation = self.snake_game.get_normalized_state()
        else:
            observation = state
            
        # Verifica che l'osservazione sia nel formato corretto
        if isinstance(observation, np.ndarray) and len(observation.shape) == 1:
            # Corretto: [state_dim]
            pass
        elif isinstance(observation, np.ndarray) and len(observation.shape) > 1:
            # Più dimensioni, appiattisci
            observation = observation.reshape(-1)
        else:
            # Non è un array numpy, converti
            observation = np.array(observation, dtype=np.float32)
            
        # Calcola statistiche aggregate
        info = {}
        if self.episode_rewards:
            info = {
                "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
                "mean_episode_length": np.mean(self.episode_lengths[-100:]),
                "max_episode_reward": np.max(self.episode_rewards) if self.episode_rewards else 0,
                "total_episodes": self.total_episodes
            }
        
        return observation, info
    
    def render(self, mode: str = 'human') -> Optional[Union[str, np.ndarray]]:
        """
        Rendering dell'ambiente.
        
        Args:
            mode (str): Modalità di rendering ('human', 'rgb_array', 'ascii')
            
        Returns:
            Optional[Union[str, np.ndarray]]: Rappresentazione testuale o immagine dell'ambiente,
                                              None per la modalità 'human'
        """
        if mode == 'ascii':
            # Rendering testuale del gioco
            return self.snake_game.render_as_string()
        elif mode == 'rgb_array':
            # Rendering come array RGB (per visualizzazione)
            return self._grid_to_rgb()
        else:
            # Per compatibilità con l'API Gym
            return None
    
    def _grid_to_rgb(self) -> np.ndarray:
        """
        Converte la rappresentazione del gioco in un'immagine RGB.
        
        Returns:
            np.ndarray: Immagine RGB del gioco
        """
        # Ottieni lo stato del gioco
        state = self.snake_game.get_state()
        
        # Definisci i colori (RGB)
        colors = {
            0: [0, 0, 0],      # Sfondo (nero)
            1: [0, 200, 0],    # Corpo del serpente (verde)
            2: [0, 255, 0],    # Testa del serpente (verde brillante)
            3: [255, 0, 0]     # Cibo (rosso)
        }
        
        # Dimensione del pixel in output (per una migliore visualizzazione)
        pixel_size = 10
        
        # Crea l'immagine RGB vuota
        img = np.zeros((self.grid_size * pixel_size, self.grid_size * pixel_size, 3), dtype=np.uint8)
        
        # Riempi l'immagine con i colori appropriati
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                cell_value = int(state[y, x])
                color = colors.get(cell_value, [100, 100, 100])  # Grigio per valori sconosciuti
                
                # Riempi il pixel corrispondente
                y_start, y_end = y * pixel_size, (y + 1) * pixel_size
                x_start, x_end = x * pixel_size, (x + 1) * pixel_size
                img[y_start:y_end, x_start:x_end] = color
                
        return img
    
    def close(self):
        """
        Chiude l'ambiente e libera le risorse.
        
        Returns:
            None
        """
        pass
    
    def get_valid_actions(self) -> List[int]:
        """
        Restituisce la lista di azioni valide nello stato corrente.
        
        Utile per implementazioni che potrebbero limitare le azioni
        possibili in determinati stati.
        
        Returns:
            list: Lista di azioni valide
        """
        # Nel caso standard, tutte le azioni sono valide
        valid_actions = [0, 1, 2]  # Dritto, destra, sinistra
        
        # Se il gioco permette di girare indietro, aggiungi l'azione indietro
        if hasattr(self.snake_game, 'allow_backward') and self.snake_game.allow_backward:
            valid_actions.append(3)
            
        return valid_actions
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche dell'ambiente.
        
        Returns:
            Dict[str, Any]: Statistiche dell'ambiente
        """
        stats = {
            "total_episodes": self.total_episodes,
            "current_episode_length": self.current_episode_length,
            "current_episode_reward": self.current_episode_reward,
        }
        
        if self.episode_rewards:
            stats.update({
                "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
                "mean_episode_length": np.mean(self.episode_lengths[-100:]),
                "max_episode_reward": np.max(self.episode_rewards),
                "min_episode_reward": np.min(self.episode_rewards),
                "last_episode_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
                "last_episode_length": self.episode_lengths[-1] if self.episode_lengths else 0,
            })
            
        return stats
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Imposta il seed per la riproducibilità.
        
        Args:
            seed (int, optional): Seed da impostare
            
        Returns:
            List[int]: Lista dei seed impostati
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]

# Test standalone
if __name__ == "__main__":
    # Test dell'ambiente con azioni casuali
    env = SnakeEnv(use_normalized_state=True)
    observation, info = env.reset()
    
    print("Test dell'ambiente Snake con azioni casuali")
    print(f"Osservazione iniziale: {observation}")
    print(f"Forma dell'osservazione: {observation.shape}")
    
    # Esegui alcune azioni casuali
    for i in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {i+1}")
        print(f"Azione: {action}")
        print(f"Ricompensa: {reward}")
        print(f"Terminato: {terminated}")
        print(f"Info: {info}")
        
        if terminated:
            print("\nEpisodio terminato, reset dell'ambiente")
            observation, info = env.reset()
    
    # Test del rendering
    print("\nTest del rendering ASCII:")
    print(env.render(mode='ascii'))
    
    env.close() 