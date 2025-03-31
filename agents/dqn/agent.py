#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Agente DQN
===============
Implementa l'agente Deep Q-Learning con diverse strategie avanzate.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
import time
import pickle
import json
from pathlib import Path

# Importa moduli interni
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents.dqn.buffer import ReplayBuffer, PrioritizedReplayBuffer
from agents.dqn.network import create_model
from utils.hardware_utils import get_device
from utils.common import timeit, set_seed, create_checkpoint_dir

class DQNAgent:
    """
    Agente che utilizza Deep Q-Learning con funzionalità avanzate.
    
    Implementa:
    - Policy ε-greedy per bilanciare esplorazione e sfruttamento
    - Replay buffer standard o prioritizzato
    - Target network per stabilizzare l'apprendimento
    - Double DQN per ridurre sovrastima dei Q-values
    - Dueling DQN per separazione valore-vantaggio
    - N-step returns per propagazione efficiente del reward
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Inizializza l'agente DQN.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            config (Dict[str, Any]): Configurazione dell'agente
        """
        # Dati dimensionali
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Configurazione
        self.config = config
        self.complexity = config.get("complexity", "base")
        
        # Parametri learning
        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.99)  # Fattore di sconto
        self.learning_rate = config.get("lr", 1e-3)
        self.target_update = config.get("target_update", 10)  # Frequenza aggiornamento target network
        
        # Parametri esplorazione
        self.eps_start = config.get("eps_start", 1.0)
        self.eps_end = config.get("eps_end", 0.01)
        self.eps_decay = config.get("eps_decay", 10000)
        self.epsilon = self.eps_start  # Valore iniziale
        
        # Parametri algoritmo avanzati
        self.use_double_dqn = config.get("use_double_dqn", True)
        self.use_dueling_dqn = config.get("use_dueling_dqn", False)
        self.use_prioritized_replay = config.get("use_prioritized_replay", False)
        self.n_step_returns = config.get("n_step_returns", 1)
        
        # Dispositivo (CPU/GPU)
        self.device = get_device()
        
        # Crea i modelli
        model_config = {
            "complexity": self.complexity,
            "use_dueling": self.use_dueling_dqn,
            "hidden_layers": config.get("hidden_layers", None)
        }
        
        # Determina il tipo di modello
        if self.use_dueling_dqn:
            if config.get("use_noisy", False):
                model_type = "noisy_dueling"
            else:
                model_type = "dueling"
        else:
            if config.get("use_noisy", False):
                model_type = "noisy"
            else:
                model_type = "dqn"
        
        self.policy_net = create_model(
            model_type=model_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.get("hidden_layers", [128, 128])
        ).to(self.device)
        
        self.target_net = create_model(
            model_type=model_type,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=config.get("hidden_layers", [128, 128])
        ).to(self.device)
        
        # Inizializza il target network con gli stessi pesi del policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network sempre in modalità valutazione
        
        # Ottimizzatore
        optimizer_name = config.get("optimizer", "Adam")
        if optimizer_name == "Adam":
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        elif optimizer_name == "RMSprop":
            self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Criterio di loss
        loss_name = config.get("loss", "MSELoss")
        if loss_name == "HuberLoss":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # Scheduler per il learning rate (opzionale)
        if config.get("use_lr_scheduler", False):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='max', 
                factor=0.5, 
                patience=1000, 
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Buffer di memoria
        memory_size = config.get("memory_size", 10000)
        if self.use_prioritized_replay:
            alpha = config.get("prioritized_replay_alpha", 0.6)
            beta = config.get("prioritized_replay_beta", 0.4)
            beta_increment = config.get("prioritized_replay_beta_increment", 0.001)
            self.memory = PrioritizedReplayBuffer(
                capacity=memory_size,
                state_dim=state_dim,
                device=self.device,
                alpha=alpha,
                beta=beta,
                beta_increment=beta_increment
            )
        else:
            self.memory = ReplayBuffer(
                capacity=memory_size,
                state_dim=state_dim,
                device=self.device
            )
        
        # Buffer per n-step returns
        if self.n_step_returns > 1:
            self.n_step_buffer = deque(maxlen=self.n_step_returns)
        
        # Contatori per il training
        self.training_steps = 0
        self.updates = 0
        self.episodes = 0
        
        # Metriche di performance
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []
        self.q_value_history = []
        
        # Flag per la modalità di training
        self.training_mode = True
    
    def select_action(self, state: np.ndarray, training: bool = True, deterministic: bool = False) -> int:
        """
        Seleziona un'azione usando ε-greedy policy.
        
        Args:
            state (np.ndarray): Stato corrente
            training (bool): Se True, usa la policy ε-greedy, altrimenti usa policy greedy
            deterministic (bool): Se True, usa una policy deterministica (equivalente a training=False)
            
        Returns:
            int: Azione selezionata
        """
        # Se deterministic è True, forza training a False
        if deterministic:
            training = False
            
        if training and random.random() < self.epsilon:
            # Esplorazione: azione casuale
            return random.randint(0, self.action_dim - 1)
        else:
            # Sfruttamento: azione con Q-value massimo
            with torch.no_grad():
                # Converti lo stato in un tensore
                if not isinstance(state, torch.Tensor):
                    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # Calcola i Q-values
                q_values = self.policy_net(state)
                
                # Aggiungi alla cronologia per il debug
                if training and len(self.q_value_history) < 1000:
                    self.q_value_history.append(q_values.cpu().numpy().mean())
                
                # Seleziona l'azione con il massimo Q-value
                return q_values.max(1)[1].item()
    
    def get_q_values(self, state: np.ndarray) -> torch.Tensor:
        """
        Calcola i Q-values per uno stato.
        
        Args:
            state (np.ndarray): Stato corrente
            
        Returns:
            torch.Tensor: Tensore con i Q-values per ogni azione
        """
        with torch.no_grad():
            # Converti lo stato in un tensore se necessario
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Calcola i Q-values
            q_values = self.policy_net(state)
            
            return q_values
    
    def update_epsilon(self):
        """Aggiorna il valore di epsilon secondo la policy di decadimento."""
        self.epsilon = max(self.eps_end, self.eps_start - 
                          (self.eps_start - self.eps_end) * min(1.0, self.episodes / self.eps_decay))
    
    def compute_loss(self, states, actions, rewards, next_states, dones):
        """
        Calcola la loss per il DQN.
        
        Args:
            states (torch.Tensor): Batch di stati
            actions (torch.Tensor): Batch di azioni
            rewards (torch.Tensor): Batch di ricompense
            next_states (torch.Tensor): Batch di stati successivi
            dones (torch.Tensor): Batch di flag terminali
            
        Returns:
            torch.Tensor: Loss calcolata
        """
        # Verifica che state_dim sia disponibile, altrimenti ricavalo dallo shape della rete
        if not hasattr(self, 'state_dim') or self.state_dim is None:
            # Prova a inferire la dimensione dello stato dalla prima layer della rete
            for mod in self.policy_net.modules():
                if isinstance(mod, nn.Linear):
                    self.state_dim = mod.in_features
                    print(f"Dimensione stato inferita: {self.state_dim}")
                    break
            # Se ancora non è disponibile, usa una dimensione predefinita
            if not hasattr(self, 'state_dim') or self.state_dim is None:
                if states is not None and hasattr(states, 'shape') and len(states.shape) > 1:
                    self.state_dim = states.shape[1]
                    print(f"Dimensione stato ricavata dagli input: {self.state_dim}")
                else:
                    print("AVVISO: Impossibile determinare la dimensione dello stato!")
                    self.state_dim = 13  # Usa la dimensione predefinita per Snake
        
        # Ottieni il batch_size se possibile
        batch_size = states.shape[0] if len(states.shape) > 1 else 1
        
        # Caso speciale: gestione di next_states con forma [batch_size, 1]
        if next_states.shape[1] == 1 and states.shape[1] == self.state_dim:
            print(f"Rilevato next_states con forma [batch_size, 1]. Espando a [batch_size, state_dim] duplicando")
            # Se next_states ha un solo valore per batch ma states ha tutti i valori,
            # crea un nuovo tensore espanso copiando i valori da states
            expanded_next_states = states.clone()
            # Sostituisci solo il primo valore con il valore di next_states
            expanded_next_states[:, 0:1] = next_states
            # Usa il tensore espanso
            next_states = expanded_next_states
            print(f"Forma next_states espansa: {next_states.shape}")
        
        # Assicura che i tensori abbiano la forma corretta [batch_size, state_dim]
        if len(states.shape) == 1:
            # Solo una dimensione: è un singolo stato o un batch schiacciato
            if states.shape[0] == self.state_dim:
                # È un singolo stato
                states = states.unsqueeze(0)
            else:
                # Potrebbe essere un batch schiacciato
                states = states.view(batch_size, self.state_dim)
        elif len(states.shape) > 1 and states.shape[1] != self.state_dim:
            # Se ha più dimensioni ma la seconda non è state_dim, prova a correggerla
            if states.shape[0] == self.state_dim and states.shape[1] == batch_size:
                # È trasposto
                states = states.transpose(0, 1)
            elif states.shape[1] == 1:
                # Ha una dimensione extra o è schiacciato
                # In questo caso, NON proviamo a ridimensionare a [batch_size, state_dim]
                # poiché non abbiamo abbastanza dati. Invece, usiamo direttamente come vettore 1D
                print(f"Usando states con forma {states.shape} direttamente")
                
        # Applica le stesse verifiche a next_states, ma con cautela
        if len(next_states.shape) == 1:
            if next_states.shape[0] == self.state_dim:
                next_states = next_states.unsqueeze(0)
            else:
                if next_states.numel() == batch_size:
                    # Non abbiamo abbastanza elementi per il reshape
                    print(f"AVVISO: next_states ha solo {next_states.numel()} elementi, insufficienti per reshape a [batch_size, state_dim]")
                    # Crea un tensore fittizio con la forma corretta
                    dummy_next_states = torch.zeros(batch_size, self.state_dim, device=next_states.device)
                    for i in range(batch_size):
                        dummy_next_states[i, 0] = next_states[i]  # Copia il valore disponibile
                    next_states = dummy_next_states
                else:
                    next_states = next_states.view(batch_size, self.state_dim)
        elif len(next_states.shape) > 1 and next_states.shape[1] != self.state_dim:
            if next_states.shape[0] == self.state_dim and next_states.shape[1] == batch_size:
                next_states = next_states.transpose(0, 1)
            elif next_states.shape[1] == 1:
                # Ha una sola dimensione ma non possiamo ridimensionare
                print(f"AVVISO: next_states ha forma {next_states.shape}, impossibile espandere a [batch_size, state_dim]")
                # Crea un tensore fittizio
                dummy_next_states = torch.zeros(batch_size, self.state_dim, device=next_states.device)
                # Copia i valori disponibili
                dummy_next_states[:, 0:1] = next_states
                next_states = dummy_next_states
                
        # Assicurati che azioni, rewards e dones abbiano la forma giusta
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(1)
        if len(rewards.shape) == 1:
            rewards = rewards.unsqueeze(1)
        if len(dones.shape) == 1:
            dones = dones.unsqueeze(1)
            
        # Debug: stampa le forme dei tensori finali
        print(f"FORME FINALI: states {states.shape}, next_states {next_states.shape}, actions {actions.shape}")
        
        # Calcola Q-values correnti
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Calcola Q-values target
        if self.use_double_dqn:
            # Double DQN: usa il policy net per selezionare le azioni e il target net per valutarle
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
        else:
            # DQN standard: usa direttamente il target net per trovare il massimo
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        
        # Calcola Q-values target con equazione di Bellman
        expected_q_values = rewards + (self.gamma ** self.n_step_returns) * next_q_values * (1 - dones)
        
        # Calcola la loss
        if self.use_prioritized_replay:
            # Per semplicità, qui non gestiamo i pesi per le priorità
            # In un'implementazione completa, i pesi dovrebbero essere passati come parametro
            weights = torch.ones_like(rewards)
            elementwise_loss = self.criterion(current_q_values, expected_q_values.detach())
            loss = (elementwise_loss * weights).mean()
        else:
            loss = self.criterion(current_q_values, expected_q_values.detach())
        
        return loss
    
    def store_transition(self, state: np.ndarray, action: int, next_state: np.ndarray, 
                        reward: float, done: bool):
        """
        Memorizza una transizione nel buffer.
        
        Args:
            state (np.ndarray): Stato corrente
            action (int): Azione intrapresa
            next_state (np.ndarray): Stato successivo
            reward (float): Reward ottenuto
            done (bool): Flag di terminazione dell'episodio
        """
        # Verifica e correggi i tipi e la forma degli stati
        if isinstance(state, torch.Tensor):
            state = state.cpu().detach().numpy()
            
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().detach().numpy()
        
        # Verifica dimensionalità dello stato
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
            
        # Correggi la forma se necessario
        if state.size == self.state_dim and len(state.shape) > 1:
            # Stato multidimensionale con numero di elementi giusto
            state = state.reshape(self.state_dim)
            
        if next_state.size == self.state_dim and len(next_state.shape) > 1:
            # Stato multidimensionale con numero di elementi giusto
            next_state = next_state.reshape(self.state_dim)
        
        # Gestione di n-step returns
        if self.n_step_returns > 1:
            # Memorizza transizione nel buffer n-step
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffer) < self.n_step_returns:
                return
                
            # Calcola reward n-step
            n_state, n_action, n_reward, n_next_state, n_done = self.get_n_step_info()
            
            # Memorizza la transizione con n-step return nel buffer principale
            self.memory.push(n_state, n_action, n_next_state, n_reward, n_done)
        else:
            # Memorizza direttamente la transizione
            self.memory.push(state, action, next_state, reward, done)
    
    def get_n_step_info(self) -> Tuple:
        """
        Calcola le informazioni per il n-step return.
        
        Returns:
            tuple: (stato, azione, reward n-step, stato successivo, done)
        """
        # Prendi il primo stato e azione
        first_state, first_action, _, _, _ = self.n_step_buffer[0]
        
        # Calcola il reward cumulativo scontato
        n_reward = 0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_reward += r * (self.gamma ** i)
        
        # Prendi l'ultimo stato e done
        _, _, _, last_next_state, last_done = self.n_step_buffer[-1]
        
        return first_state, first_action, n_reward, last_next_state, last_done
    
    def optimize_model(self) -> float:
        """
        Esegue un passo di ottimizzazione del modello.
        
        Returns:
            float: Loss calcolata
        """
        if not self.memory.can_sample(self.batch_size):
            return 0.0
            
        # Campiona transizioni dal buffer
        if self.use_prioritized_replay:
            states, actions, next_states, rewards, dones, weights, indices = self.memory.sample(self.batch_size)
        else:
            states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)
            weights = torch.ones_like(rewards).to(self.device)
        
        # Calcola la loss
        loss = self.compute_loss(states, actions, rewards, next_states, dones)
        
        # Aggiorna priorità nel buffer prioritized
        if self.use_prioritized_replay:
            # Calcola le priorità usando la loss per ciascun elemento
            with torch.no_grad():
                current_q_values = self.policy_net(states).gather(1, actions)
                if self.use_double_dqn:
                    next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                    next_q_values = self.target_net(next_states).gather(1, next_actions)
                else:
                    next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
                
                expected_q_values = rewards + (self.gamma ** self.n_step_returns) * next_q_values * (1 - dones)
                elementwise_loss = self.criterion(current_q_values, expected_q_values)
                priorities = elementwise_loss.detach().cpu().numpy() + 1e-6  # Evita priorità zero
            
            self.memory.update_priorities(indices, priorities)
        
        # Ottimizzazione
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip del gradiente (opzionale)
        if self.config.get("clip_grad", False):
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.get("max_grad_norm", 1.0))
            
        self.optimizer.step()
        
        # Aggiorna i contatori
        self.updates += 1
        
        # Aggiorna il target network se necessario
        if self.updates % self.target_update == 0:
            self.update_target_network()
        
        # Salva la loss per monitoraggio
        loss_value = loss.item()
        self.loss_history.append(loss_value)
        
        return loss_value
    
    def train(self, num_updates: int = 1) -> Dict[str, Any]:
        """
        Esegue un numero specificato di passi di ottimizzazione.
        
        Args:
            num_updates (int): Numero di passi di ottimizzazione da eseguire
            
        Returns:
            Dict[str, Any]: Metriche di training
        """
        # Imposta il flag di training
        self.training_mode = True
        
        # Assicurati che ci siano abbastanza dati per il training
        if not self.memory.can_sample(self.batch_size):
            return {
                "loss": 0.0,
                "updates_performed": 0,
                "epsilon": self.epsilon,
                "memory_size": len(self.memory)
            }
        
        total_loss = 0.0
        updates_performed = 0
        
        # Esegui il numero specificato di aggiornamenti
        for _ in range(num_updates):
            loss = self.optimize_model()
            total_loss += loss
            updates_performed += 1
            
            # Aggiorna epsilon per la policy ε-greedy
            self.update_epsilon()
            
            # Aggiorna il contatore di passi
            self.training_steps += 1
        
        # Calcola loss media
        avg_loss = total_loss / updates_performed if updates_performed > 0 else 0.0
        
        return {
            "loss": avg_loss,
            "updates_performed": updates_performed,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "training_steps": self.training_steps
        }
    
    def update_target_network(self):
        """Aggiorna il target network con i pesi del policy network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, path: str, save_memory: bool = False, save_optimizer: bool = True):
        """
        Salva lo stato dell'agente su disco.
        
        Args:
            path (str): Percorso in cui salvare il modello
            save_memory (bool): Se True, salva anche il buffer di memoria
            save_optimizer (bool): Se True, salva anche lo stato dell'ottimizzatore
        """
        # Crea la directory se necessario
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepara i dati da salvare
        data = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict() if save_optimizer else None,
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler and save_optimizer else None,
            "config": self.config,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "training_steps": self.training_steps,
            "updates": self.updates,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "loss_history": self.loss_history,
            "q_value_history": self.q_value_history,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
        }
        
        # Salva il modello
        torch.save(data, path)
        
        # Salva la memoria separatamente se richiesto
        if save_memory and len(self.memory) > 0:
            memory_path = os.path.splitext(path)[0] + "_memory.pkl"
            self.memory.save(memory_path)
            
        print(f"Modello salvato in: {path}")
        if save_memory:
            print(f"Memoria salvata in: {memory_path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, 
             load_memory: bool = False) -> 'DQNAgent':
        """
        Carica un agente salvato.
        
        Args:
            path (str): Percorso del file salvato
            device (torch.device, optional): Dispositivo su cui caricare il modello
            load_memory (bool): Se True, carica anche il buffer di memoria
            
        Returns:
            DQNAgent: Istanza caricata
        """
        device = device or get_device()
        
        # Carica il file salvato
        checkpoint = torch.load(path, map_location=device)
        
        # Estrai i dati
        config = checkpoint.get("config", {})
        state_dim = checkpoint.get("state_dim", None)
        action_dim = checkpoint.get("action_dim", None)
        
        if state_dim is None or action_dim is None:
            raise ValueError("Impossibile determinare le dimensioni dello stato/azione dal checkpoint.")
        
        # Crea una nuova istanza dell'agente
        agent = cls(state_dim=state_dim, action_dim=action_dim, config=config)
        
        # Carica i parametri dei modelli
        agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        if "target_state_dict" in checkpoint and checkpoint["target_state_dict"] is not None:
            agent.target_net.load_state_dict(checkpoint["target_state_dict"])
        else:
            agent.target_net.load_state_dict(checkpoint["policy_state_dict"])
        
        # Carica lo stato dell'ottimizzatore se disponibile
        if "optimizer_state_dict" in checkpoint and checkpoint["optimizer_state_dict"] is not None:
            agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Carica lo stato dello scheduler se disponibile
        if "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None and agent.scheduler:
            agent.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Carica i contatori e la cronologia
        agent.training_steps = checkpoint.get("training_steps", 0)
        agent.updates = checkpoint.get("updates", 0)
        agent.episodes = checkpoint.get("episodes", 0)
        agent.epsilon = checkpoint.get("epsilon", agent.eps_end)
        agent.loss_history = checkpoint.get("loss_history", [])
        agent.q_value_history = checkpoint.get("q_value_history", [])
        agent.episode_rewards = checkpoint.get("episode_rewards", [])
        agent.episode_lengths = checkpoint.get("episode_lengths", [])
        
        # Carica il buffer di memoria se richiesto
        if load_memory:
            memory_path = os.path.splitext(path)[0] + "_memory.pkl"
            if os.path.exists(memory_path):
                if agent.use_prioritized_replay:
                    agent.memory = PrioritizedReplayBuffer.load(memory_path, device)
                else:
                    agent.memory = ReplayBuffer.load(memory_path, device)
                print(f"Memoria caricata da: {memory_path}")
            else:
                print(f"File memoria non trovato: {memory_path}")
        
        # Imposta il modello in modalità di valutazione (non training)
        agent.training_mode = False
        agent.policy_net.eval()
        agent.target_net.eval()
        
        return agent
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Ottiene le statistiche dell'agente.
        
        Returns:
            Dict[str, Any]: Statistiche dell'agente
        """
        stats = {
            "training_steps": self.training_steps,
            "updates": self.updates,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
            "memory_size": len(self.memory),
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate
        }
        
        if self.episode_rewards:
            stats.update({
                "mean_episode_reward": np.mean(self.episode_rewards[-100:]),
                "mean_episode_length": np.mean(self.episode_lengths[-100:]),
                "max_episode_reward": np.max(self.episode_rewards) if len(self.episode_rewards) > 0 else 0,
                "total_episodes": len(self.episode_rewards)
            })
            
        if self.loss_history:
            stats.update({
                "mean_loss": np.mean(self.loss_history[-100:]),
                "last_loss": self.loss_history[-1] if len(self.loss_history) > 0 else 0
            })
            
        if self.q_value_history:
            stats.update({
                "mean_q_value": np.mean(self.q_value_history[-100:]),
                "last_q_value": self.q_value_history[-1] if len(self.q_value_history) > 0 else 0
            })
            
        return stats
    
    def train_mode(self):
        """Imposta l'agente in modalità training."""
        self.training_mode = True
        self.policy_net.train()
    
    def eval_mode(self):
        """Imposta l'agente in modalità valutazione."""
        self.training_mode = False
        self.policy_net.eval()
    
    def debug_state_shape(self, state):
        """
        Verifica e corregge la forma di uno stato per debug.
        
        Args:
            state: Lo stato da verificare
            
        Returns:
            tuple: (stato corretto, messaggio di debug)
        """
        debug_msg = f"Stato originale forma: {state.shape if hasattr(state, 'shape') else 'sconosciuta'}, tipo: {type(state)}"
        
        if isinstance(state, np.ndarray):
            corrected_state = state
            
            if len(state.shape) == 1:
                if state.shape[0] == self.state_dim:
                    # È un singolo stato con forma corretta
                    debug_msg += " - OK singolo stato numpy"
                    # Non serve conversione
                else:
                    # Forma errata, prova a correggere
                    corrected_state = state.reshape(1, self.state_dim)
                    debug_msg += f" -> corretto a {corrected_state.shape}"
            
            # Converti in tensore per compatibilità con la rete
            corrected_state = torch.FloatTensor(corrected_state).to(self.device)
        
        elif isinstance(state, torch.Tensor):
            corrected_state = state
            
            if len(state.shape) == 1:
                if state.shape[0] == self.state_dim:
                    # È un singolo stato con forma corretta
                    corrected_state = state.unsqueeze(0)
                    debug_msg += " -> aggiunta dimensione batch a 1"
                else:
                    # Potrebbe essere un batch schiacciato
                    corrected_state = state.view(1, self.state_dim)
                    debug_msg += f" -> rimodellato a {corrected_state.shape}"
            
            # Assicurati che sia sul dispositivo corretto
            corrected_state = corrected_state.to(self.device)
        
        else:
            # Tipo non supportato
            debug_msg += " - TIPO NON SUPPORTATO!"
            corrected_state = state
            
        debug_msg += f", forma finale: {corrected_state.shape if hasattr(corrected_state, 'shape') else 'sconosciuta'}"
        return corrected_state, debug_msg

# Test standalone
if __name__ == "__main__":
    # Test delle funzionalità dell'agente
    from gymnasium import spaces
    import gymnasium as gym
    
    print("Test dell'agente DQN")
    
    # Crea un ambiente semplice per il test
    state_dim = 4
    action_dim = 2
    
    # Configurazione per il test
    config = {
        "complexity": "base",
        "hidden_layers": [16, 16],
        "batch_size": 4,
        "memory_size": 100,
        "gamma": 0.9,
        "lr": 0.01,
        "use_double_dqn": True,
        "n_step_returns": 2
    }
    
    # Crea l'agente
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, config=config)
    
    print(f"Stato agente:\n{agent.get_stats()}")
    
    # Test salvataggio e caricamento
    test_path = "test_agent.pt"
    agent.save(test_path)
    
    loaded_agent = DQNAgent.load(test_path)
    print(f"Stato agente caricato:\n{loaded_agent.get_stats()}")
    
    # Pulizia
    os.remove(test_path) 