#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Agente DQN
===============
Implementa l'agente Deep Q-Learning con replay buffer.

Autore: Federico Baratti
Versione: 1.0
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Dict, Any, Tuple, Optional
import time
import pickle
import json
from pathlib import Path

from dqn_agent.models import create_model
from backend.utils import calculate_epsilon, get_device, timeit, create_checkpoint_dir

# Definisci la struttura di una transizione nella memoria
Experience = namedtuple('Experience', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Buffer di memoria per esperienze del DQN.
    
    Memorizza le transizioni (stato, azione, nuovo stato, reward, done)
    e fornisce un metodo per campionare batch casuali per l'allenamento.
    """
    
    def __init__(self, capacity: int, state_dim: int = 9, 
                 device: torch.device = None):
        """
        Inizializza il buffer di replay.
        
        Args:
            capacity (int): Capacità massima del buffer
            state_dim (int): Dimensione dello stato
            device (torch.device): Dispositivo per i tensori (CUDA/CPU)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.device = device if device else get_device()
        self.position = 0
    
    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, 
             reward: float, done: bool) -> None:
        """
        Aggiunge una transizione alla memoria.
        
        Args:
            state (np.ndarray): Stato corrente
            action (int): Azione intrapresa
            next_state (np.ndarray): Stato successivo
            reward (float): Reward ottenuto
            done (bool): Flag che indica se l'episodio è terminato
        """
        # Crea una nuova esperienza
        experience = Experience(state, action, next_state, reward, done)
        
        # Aggiunge l'esperienza alla memoria
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Campiona un batch di esperienze dalla memoria.
        
        Args:
            batch_size (int): Dimensione del batch da campionare
            
        Returns:
            tuple: Tuple di tensori (stati, azioni, stati successivi, rewards, done flags)
        """
        # Limita il batch size alla dimensione della memoria
        batch_size = min(batch_size, len(self.memory))
        
        # Campiona esperienze casuali
        experiences = random.sample(self.memory, batch_size)
        
        # Estrai componenti
        batch = Experience(*zip(*experiences))
        
        # Converti in tensori
        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        return states, actions, next_states, rewards, dones
    
    def __len__(self) -> int:
        """Restituisce il numero di esperienze nel buffer."""
        return len(self.memory)
    
    def can_sample(self, batch_size: int) -> bool:
        """Verifica se è possibile campionare un batch di dimensione specificata."""
        return len(self) >= batch_size
    
    def save(self, path: str) -> None:
        """
        Salva il buffer di replay su disco.
        
        Args:
            path (str): Percorso del file di salvataggio
        """
        with open(path, 'wb') as f:
            pickle.dump({
                'memory': list(self.memory),
                'capacity': self.capacity,
                'state_dim': self.state_dim
            }, f)
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'ReplayBuffer':
        """
        Carica un buffer di replay da disco.
        
        Args:
            path (str): Percorso del file di salvataggio
            device (torch.device): Dispositivo per i tensori
            
        Returns:
            ReplayBuffer: Buffer caricato
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        buffer = cls(capacity=data['capacity'], state_dim=data['state_dim'], device=device)
        buffer.memory = deque(data['memory'], maxlen=buffer.capacity)
        
        return buffer


class DQNAgent:
    """
    Agente che utilizza Deep Q-Learning con experience replay.
    
    Implementa:
    - Policy ε-greedy per bilanciare esplorazione e sfruttamento
    - Replay buffer per memorizzare e campionare esperienze
    - Target network per stabilizzare l'apprendimento
    """
    
    def __init__(self, state_dim: int, action_dim: int, 
                 complexity: str = "base", config: Dict[str, Any] = None):
        """
        Inizializza l'agente DQN.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            complexity (str): Livello di complessità del modello ('base', 'avanzato', 'complesso', 'perfetto')
            config (Dict[str, Any], optional): Configurazione personalizzata
        """
        # Dati dimensionali
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Ottieni la configurazione 
        from dqn_agent.config import get_config
        self.config = config if config else get_config(complexity)
        self.complexity = complexity
        
        # Parametri learning
        self.batch_size = self.config.get("batch_size", 32)
        self.gamma = self.config.get("gamma", 0.99)  # Fattore di sconto
        self.learning_rate = self.config.get("lr", 1e-3)
        self.target_update = self.config.get("target_update", 10)  # Frequenza aggiornamento target network
        
        # Parametri esplorazione
        self.eps_start = self.config.get("eps_start", 1.0)
        self.eps_end = self.config.get("eps_end", 0.01)
        self.eps_decay = self.config.get("eps_decay", 10000)
        self.epsilon = self.eps_start  # Valore iniziale
        
        # Dispositivo (CPU/GPU)
        self.device = get_device()
        
        # Crea i modelli
        self.policy_net = create_model(
            complexity=complexity,
            input_dim=state_dim,
            output_dim=action_dim
        ).to(self.device)
        
        self.target_net = create_model(
            complexity=complexity,
            input_dim=state_dim,
            output_dim=action_dim
        ).to(self.device)
        
        # Copia i pesi dal policy network al target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in modalità valutazione
        
        # Configurazione ottimizzatore
        optimizer_name = self.config.get("optimizer", "Adam")
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Configurazione funzione di perdita
        loss_name = self.config.get("loss", "MSELoss")
        if loss_name == "MSELoss":
            self.criterion = nn.MSELoss()
        elif loss_name == "SmoothL1Loss" or loss_name == "HuberLoss":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # Memoria di replay
        memory_size = self.config.get("memory_size", 10000)
        self.memory = ReplayBuffer(
            capacity=memory_size,
            state_dim=state_dim,
            device=self.device
        )
        
        # Metriche e statistiche
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "loss_history": [],
            "q_values": [],
            "epsilon_history": []
        }
        
        # Parallellizzazione
        self._setup_parallel()
    
    def _setup_parallel(self) -> None:
        """Configura la parallellizzazione del modello se richiesta."""
        if "parallel_strategy" in self.config and self.config["parallel_strategy"] == "DataParallel":
            if torch.cuda.device_count() > 1:
                print(f"Usando {torch.cuda.device_count()} GPU!")
                self.policy_net = nn.DataParallel(self.policy_net)
                self.target_net = nn.DataParallel(self.target_net)
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleziona un'azione usando la policy ε-greedy.
        
        Args:
            state (np.ndarray): Stato corrente
            training (bool): Flag che indica se l'agente è in fase di training
            
        Returns:
            int: Azione selezionata
        """
        # Durante il training, usa ε-greedy policy
        if training and random.random() < self.epsilon:
            # Scegli un'azione casuale
            return random.randint(0, self.action_dim - 1)
        
        # Durante l'inferenza o quando sfruttiamo la policy, usa il modello
        with torch.no_grad():
            # Converti lo stato in un tensore
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Ottieni Q-values dal policy network
            q_values = self.policy_net(state)
            
            # Salva statistiche (media dei Q-values)
            if training:
                self.stats["q_values"].append(q_values.mean().item())
            
            # Scegli l'azione con il Q-value massimo
            return q_values.max(1)[1].item()
    
    @timeit
    def optimize_model(self) -> float:
        """
        Ottimizza il modello usando un batch di esperienze.
        
        Returns:
            float: Valore della loss
        """
        # Verifica se abbiamo abbastanza esperienze
        if not self.memory.can_sample(self.batch_size):
            return 0.0
        
        # Campiona un batch di esperienze
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
        
        # Calcola Q-values correnti
        q_values = self.policy_net(states).gather(1, actions)
        
        # Calcola Q-values target (usando il target network)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Calcola Q-values target con equazione di Bellman
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calcola la loss
        loss = self.criterion(q_values, target_q_values)
        
        # Azzera i gradienti
        self.optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()
        
        # Clip dei gradienti (opzionale)
        # torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        # Aggiorna i pesi
        self.optimizer.step()
        
        # Salva il valore della loss
        loss_value = loss.item()
        self.stats["loss_history"].append(loss_value)
        
        return loss_value
    
    def update_target_network(self) -> None:
        """Aggiorna il target network con i pesi del policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self) -> None:
        """Aggiorna il valore di epsilon per l'esplorazione."""
        self.epsilon = calculate_epsilon(
            self.eps_start, 
            self.eps_end, 
            self.total_steps, 
            self.eps_decay
        )
        self.stats["epsilon_history"].append(self.epsilon)
    
    def remember(self, state: np.ndarray, action: int, next_state: np.ndarray,
                reward: float, done: bool) -> None:
        """
        Memorizza una transizione nel replay buffer.
        
        Args:
            state (np.ndarray): Stato corrente
            action (int): Azione intrapresa
            next_state (np.ndarray): Stato successivo
            reward (float): Reward ottenuto
            done (bool): Flag che indica se l'episodio è terminato
        """
        self.memory.push(state, action, next_state, reward, done)
    
    def train_step(self, state: np.ndarray, action: int, next_state: np.ndarray,
                  reward: float, done: bool) -> float:
        """
        Esegue un passo di training dell'agente.
        
        Args:
            state (np.ndarray): Stato corrente
            action (int): Azione intrapresa
            next_state (np.ndarray): Stato successivo
            reward (float): Reward ottenuto
            done (bool): Flag che indica se l'episodio è terminato
            
        Returns:
            float: Valore della loss (0 se non è stato fatto il training)
        """
        # Incrementa il contatore dei passi
        self.total_steps += 1
        
        # Memorizza l'esperienza
        self.remember(state, action, next_state, reward, done)
        
        # Aggiorna epsilon
        self.update_epsilon()
        
        # Se non abbiamo abbastanza esperienze, salta l'ottimizzazione
        if not self.memory.can_sample(self.batch_size):
            return 0.0
        
        # Ottimizza il modello
        loss = self.optimize_model()
        
        # Aggiorna il target network quando necessario
        if self.total_steps % self.target_update == 0:
            self.update_target_network()
            print(f"Target network aggiornato al passo {self.total_steps}")
        
        return loss
    
    def end_episode(self, episode_reward: float, episode_length: int) -> None:
        """
        Aggiorna le statistiche alla fine di un episodio.
        
        Args:
            episode_reward (float): Reward totale dell'episodio
            episode_length (int): Lunghezza dell'episodio
        """
        self.episode_count += 1
        self.stats["episode_rewards"].append(episode_reward)
        self.stats["episode_lengths"].append(episode_length)
        
        # Aggiorna il miglior reward
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
    
    def save(self, path: str = None, additional_info: Dict[str, Any] = None) -> str:
        """
        Salva il modello e lo stato dell'agente.
        
        Args:
            path (str, optional): Percorso dove salvare il modello
            additional_info (Dict[str, Any], optional): Informazioni aggiuntive da salvare
            
        Returns:
            str: Percorso dove è stato salvato il modello
        """
        # Crea la directory dei checkpoint se necessario
        if path is None:
            checkpoint_dir = create_checkpoint_dir()
            path = os.path.join(checkpoint_dir, f"dqn_{self.complexity}_{int(time.time())}.pt")
        
        # Prepara i dati da salvare
        save_data = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "complexity": self.complexity,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
            "epsilon": self.epsilon,
            "stats": self.stats
        }
        
        # Aggiungi informazioni aggiuntive se presenti
        if additional_info:
            save_data.update(additional_info)
        
        # Salva il modello
        torch.save(save_data, path)
        
        # Salva la memoria di replay separatamente (opzionale)
        memory_path = path.replace(".pt", "_memory.pkl")
        self.memory.save(memory_path)
        
        return path
    
    @classmethod
    def load(cls, path: str, device: torch.device = None) -> 'DQNAgent':
        """
        Carica un agente da un file.
        
        Args:
            path (str): Percorso del file
            device (torch.device, optional): Dispositivo su cui caricare il modello
            
        Returns:
            DQNAgent: Agente caricato
        """
        # Usa il dispositivo specificato o rileva automaticamente
        if device is None:
            device = get_device()
        
        # Carica i dati
        data = torch.load(path, map_location=device)
        
        # Crea un nuovo agente
        agent = cls(
            state_dim=data["state_dim"],
            action_dim=data["action_dim"],
            complexity=data["complexity"],
            config=data["config"]
        )
        
        # Carica i pesi dei modelli
        agent.policy_net.load_state_dict(data["policy_state_dict"])
        agent.target_net.load_state_dict(data["target_state_dict"])
        
        # Carica lo stato dell'ottimizzatore
        agent.optimizer.load_state_dict(data["optimizer_state_dict"])
        
        # Carica lo stato dell'agente
        agent.total_steps = data["total_steps"]
        agent.episode_count = data["episode_count"]
        agent.best_reward = data["best_reward"]
        agent.epsilon = data["epsilon"]
        agent.stats = data["stats"]
        
        # Carica la memoria di replay se esiste
        memory_path = path.replace(".pt", "_memory.pkl")
        if os.path.exists(memory_path):
            try:
                agent.memory = ReplayBuffer.load(memory_path, device)
            except Exception as e:
                print(f"Errore nel caricamento della memoria: {e}")
        
        return agent
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Restituisce le statistiche dell'agente.
        
        Returns:
            Dict[str, Any]: Statistiche dell'agente
        """
        return {
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
            "epsilon": self.epsilon,
            "stats": self.stats,
            "model_complexity": self.complexity,
            "model_params": sum(p.numel() for p in self.policy_net.parameters()),
        }


if __name__ == "__main__":
    # Test dell'agente
    from backend.snake_game import SnakeGame
    from backend.environment import SnakeEnv
    
    # Crea gioco e ambiente
    game = SnakeGame(grid_size=10)
    env = SnakeEnv(game, use_normalized_state=True)
    
    # Ottieni dimensioni
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Crea agente
    agent = DQNAgent(state_dim, action_dim, complexity="base")
    
    # Test di un episodio
    state, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    print("Inizio test episodio:")
    while not done and steps < 100:
        # Seleziona un'azione
        action = agent.select_action(state)
        
        # Esegui l'azione
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Aggiorna statistiche
        total_reward += reward
        steps += 1
        
        # Esegui un passo di training
        loss = agent.train_step(state, action, next_state, reward, done)
        
        # Passa allo stato successivo
        state = next_state
        
        if steps % 10 == 0:
            print(f"Passo {steps}, Reward: {total_reward}, Loss: {loss:.6f}")
    
    # Aggiungi statistiche dell'episodio
    agent.end_episode(total_reward, steps)
    
    # Salva il modello
    save_path = agent.save()
    print(f"Modello salvato in {save_path}")
    
    # Statistiche
    stats = agent.get_stats()
    print("\nStatistiche:")
    print(f"- Passi totali: {stats['total_steps']}")
    print(f"- Episodi: {stats['episode_count']}")
    print(f"- Miglior reward: {stats['best_reward']}")
    print(f"- Complessità modello: {stats['model_complexity']}")
    print(f"- Parametri modello: {stats['model_params']}") 