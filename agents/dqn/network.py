#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Rete Neurale
====================
Implementazioni di reti neurali per DQN.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
import math

class NoisyLinear(nn.Module):
    """
    Layer lineare con rumore parametrizzato per l'esplorazione.
    
    Implementazione del layer Noisy Network per l'esplorazione intrinseca
    nella rete, in alternativa all'epsilon-greedy.
    """
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """
        Inizializza il layer Noisy Linear.
        
        Args:
            in_features (int): Numero di feature in input
            out_features (int): Numero di feature in output
            std_init (float): Deviazione standard iniziale per il rumore
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Parametri del layer
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        # Inizializzazione dei parametri
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Inizializza i parametri del layer."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """
        Genera rumore normalizzato.
        
        Args:
            size (int): Dimensione del rumore
            
        Returns:
            torch.Tensor: Rumore normalizzato
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())
    
    def reset_noise(self):
        """Genera nuovo rumore per il layer."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Prodotto esterno
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del layer.
        
        Args:
            x (torch.Tensor): Tensore di input
            
        Returns:
            torch.Tensor: Output del layer
        """
        if self.training:
            return F.linear(x, 
                           self.weight_mu + self.weight_sigma * self.weight_epsilon,
                           self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQNModel(nn.Module):
    """
    Rete neurale base per DQN.
    
    Implementa una rete neurale fully connected per approssimare
    la funzione Q(s, a) per l'apprendimento con rinforzo.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: List[int] = [128, 128],
                 activation: Callable = F.relu):
        """
        Inizializza la rete DQN standard.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            hidden_dim (List[int]): Lista con le dimensioni dei layer nascosti
            activation (Callable): Funzione di attivazione
        """
        super(DQNModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        # Costruisci i layer
        layers = []
        
        prev_dim = state_dim
        for dim in hidden_dim:
            layers.append(nn.Linear(prev_dim, dim))
            prev_dim = dim
        
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, action_dim)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della rete.
        
        Args:
            state (torch.Tensor): Tensore dello stato
            
        Returns:
            torch.Tensor: Q-values per ogni azione
        """
        x = state
        
        # Forward attraverso i layer nascosti
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Layer di output
        q_values = self.output_layer(x)
        
        return q_values
    
    def save(self, path: str) -> None:
        """
        Salva i parametri della rete su disco.
        
        Args:
            path (str): Percorso dove salvare il modello
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva il modello
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'DQNModel':
        """
        Carica i parametri della rete da disco.
        
        Args:
            path (str): Percorso da cui caricare il modello
            device (torch.device, optional): Dispositivo su cui caricare il modello
            
        Returns:
            DQNModel: Modello caricato
        """
        # Carica il checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Crea un nuovo modello con le dimensioni salvate
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        # Carica i parametri
        model.load_state_dict(checkpoint['state_dict'])
        
        return model


class DuelingDQNModel(nn.Module):
    """
    Rete neurale Dueling DQN.
    
    Implementa una rete con architettura dueling che separa
    la stima del valore dello stato e il vantaggio di ogni azione.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: List[int] = [128, 128],
                 activation: Callable = F.relu):
        """
        Inizializza la rete Dueling DQN.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            hidden_dim (List[int]): Lista con le dimensioni dei layer nascosti
            activation (Callable): Funzione di attivazione
        """
        super(DuelingDQNModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        
        # Feature extractor condiviso
        self.features = nn.Sequential()
        
        prev_dim = state_dim
        for i, dim in enumerate(hidden_dim[:-1]):  # Tutti tranne l'ultimo
            self.features.add_module(f"layer{i}", nn.Linear(prev_dim, dim))
            self.features.add_module(f"activation{i}", nn.ReLU())
            prev_dim = dim
        
        # Stream del valore (V)
        self.value_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[-1], 1)
        )
        
        # Stream del vantaggio (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_dim, hidden_dim[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[-1], action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della rete.
        
        Args:
            state (torch.Tensor): Tensore dello stato
            
        Returns:
            torch.Tensor: Q-values per ogni azione
        """
        # Estrai le features comuni
        features = self.features(state)
        
        # Calcola valore e vantaggio
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Q-values = V(s) + [A(s,a) - media(A(s,a))]
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values
    
    def save(self, path: str) -> None:
        """
        Salva i parametri della rete su disco.
        
        Args:
            path (str): Percorso dove salvare il modello
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva il modello
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'DuelingDQNModel':
        """
        Carica i parametri della rete da disco.
        
        Args:
            path (str): Percorso da cui caricare il modello
            device (torch.device, optional): Dispositivo su cui caricare il modello
            
        Returns:
            DuelingDQNModel: Modello caricato
        """
        # Carica il checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Crea un nuovo modello con le dimensioni salvate
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim']
        )
        
        # Carica i parametri
        model.load_state_dict(checkpoint['state_dict'])
        
        return model


class NoisyDQNModel(nn.Module):
    """
    Rete neurale Noisy DQN.
    
    Implementa una rete con layer rumorosi che permettono
    un'esplorazione intrinseca senza bisogno di epsilon-greedy.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: List[int] = [128, 128],
                 activation: Callable = F.relu, std_init: float = 0.5):
        """
        Inizializza la rete Noisy DQN.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            hidden_dim (List[int]): Lista con le dimensioni dei layer nascosti
            activation (Callable): Funzione di attivazione
            std_init (float): Deviazione standard iniziale per i layer rumorosi
        """
        super(NoisyDQNModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.std_init = std_init
        
        # Primo layer (non rumoroso)
        self.input_layer = nn.Linear(state_dim, hidden_dim[0])
        
        # Layer nascosti (rumorosi)
        noisy_layers = []
        
        for i in range(len(hidden_dim) - 1):
            noisy_layers.append(NoisyLinear(hidden_dim[i], hidden_dim[i+1], std_init))
        
        self.hidden_layers = nn.ModuleList(noisy_layers)
        
        # Layer di output (rumoroso)
        self.output_layer = NoisyLinear(hidden_dim[-1], action_dim, std_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della rete.
        
        Args:
            state (torch.Tensor): Tensore dello stato
            
        Returns:
            torch.Tensor: Q-values per ogni azione
        """
        x = self.activation(self.input_layer(state))
        
        # Forward attraverso i layer nascosti
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        # Layer di output
        q_values = self.output_layer(x)
        
        return q_values
    
    def reset_noise(self):
        """Resetta il rumore in tutti i layer rumorosi."""
        for layer in self.hidden_layers:
            layer.reset_noise()
        self.output_layer.reset_noise()
    
    def save(self, path: str) -> None:
        """
        Salva i parametri della rete su disco.
        
        Args:
            path (str): Percorso dove salvare il modello
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva il modello
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'std_init': self.std_init
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'NoisyDQNModel':
        """
        Carica i parametri della rete da disco.
        
        Args:
            path (str): Percorso da cui caricare il modello
            device (torch.device, optional): Dispositivo su cui caricare il modello
            
        Returns:
            NoisyDQNModel: Modello caricato
        """
        # Carica il checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Crea un nuovo modello con le dimensioni salvate
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            std_init=checkpoint.get('std_init', 0.5)
        )
        
        # Carica i parametri
        model.load_state_dict(checkpoint['state_dict'])
        
        return model


class NoisyDuelingDQNModel(nn.Module):
    """
    Rete neurale Noisy Dueling DQN.
    
    Combina l'architettura dueling con i layer rumorosi.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: List[int] = [128, 128],
                 activation: Callable = F.relu, std_init: float = 0.5):
        """
        Inizializza la rete Noisy Dueling DQN.
        
        Args:
            state_dim (int): Dimensione dello stato
            action_dim (int): Dimensione dell'azione (numero di azioni possibili)
            hidden_dim (List[int]): Lista con le dimensioni dei layer nascosti
            activation (Callable): Funzione di attivazione
            std_init (float): Deviazione standard iniziale per i layer rumorosi
        """
        super(NoisyDuelingDQNModel, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.std_init = std_init
        
        # Feature extractor condiviso (non rumoroso)
        self.features = nn.Sequential()
        
        prev_dim = state_dim
        for i, dim in enumerate(hidden_dim[:-1]):  # Tutti tranne l'ultimo
            self.features.add_module(f"layer{i}", nn.Linear(prev_dim, dim))
            self.features.add_module(f"activation{i}", nn.ReLU())
            prev_dim = dim
        
        # Stream del valore (V) - rumoroso
        self.value_hidden = NoisyLinear(prev_dim, hidden_dim[-1], std_init)
        self.value_output = NoisyLinear(hidden_dim[-1], 1, std_init)
        
        # Stream del vantaggio (A) - rumoroso
        self.advantage_hidden = NoisyLinear(prev_dim, hidden_dim[-1], std_init)
        self.advantage_output = NoisyLinear(hidden_dim[-1], action_dim, std_init)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass della rete.
        
        Args:
            state (torch.Tensor): Tensore dello stato
            
        Returns:
            torch.Tensor: Q-values per ogni azione
        """
        # Estrai le features comuni
        features = self.features(state)
        
        # Calcola valore
        value = self.activation(self.value_hidden(features))
        value = self.value_output(value)
        
        # Calcola vantaggio
        advantage = self.activation(self.advantage_hidden(features))
        advantage = self.advantage_output(advantage)
        
        # Q-values = V(s) + [A(s,a) - media(A(s,a))]
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def reset_noise(self):
        """Resetta il rumore in tutti i layer rumorosi."""
        self.value_hidden.reset_noise()
        self.value_output.reset_noise()
        self.advantage_hidden.reset_noise()
        self.advantage_output.reset_noise()
    
    def save(self, path: str) -> None:
        """
        Salva i parametri della rete su disco.
        
        Args:
            path (str): Percorso dove salvare il modello
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Salva il modello
        torch.save({
            'state_dict': self.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'std_init': self.std_init
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'NoisyDuelingDQNModel':
        """
        Carica i parametri della rete da disco.
        
        Args:
            path (str): Percorso da cui caricare il modello
            device (torch.device, optional): Dispositivo su cui caricare il modello
            
        Returns:
            NoisyDuelingDQNModel: Modello caricato
        """
        # Carica il checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Crea un nuovo modello con le dimensioni salvate
        model = cls(
            state_dim=checkpoint['state_dim'],
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            std_init=checkpoint.get('std_init', 0.5)
        )
        
        # Carica i parametri
        model.load_state_dict(checkpoint['state_dict'])
        
        return model


def create_model(model_type: str, state_dim: int, action_dim: int, 
                hidden_dim: List[int] = [128, 128], **kwargs) -> nn.Module:
    """
    Factory function per creare un modello DQN.
    
    Args:
        model_type (str): Tipo di modello ('dqn', 'dueling', 'noisy', 'noisy_dueling')
        state_dim (int): Dimensione dello stato
        action_dim (int): Dimensione dell'azione (numero di azioni possibili)
        hidden_dim (List[int]): Lista con le dimensioni dei layer nascosti
        **kwargs: Parametri aggiuntivi specifici del modello
        
    Returns:
        nn.Module: Istanza del modello richiesto
        
    Raises:
        ValueError: Se il tipo di modello non è riconosciuto
    """
    if model_type.lower() == 'dqn':
        return DQNModel(state_dim, action_dim, hidden_dim, **kwargs)
    elif model_type.lower() == 'dueling':
        return DuelingDQNModel(state_dim, action_dim, hidden_dim, **kwargs)
    elif model_type.lower() == 'noisy':
        return NoisyDQNModel(state_dim, action_dim, hidden_dim, **kwargs)
    elif model_type.lower() == 'noisy_dueling':
        return NoisyDuelingDQNModel(state_dim, action_dim, hidden_dim, **kwargs)
    else:
        raise ValueError(f"Tipo di modello non supportato: {model_type}")


# Test standalone
if __name__ == "__main__":
    # Test dei modelli DQN
    import torch
    
    print("Test dei modelli DQN")
    
    # Parametri di test
    state_dim = 4
    action_dim = 2
    batch_size = 5
    hidden_dim = [64, 64]
    
    # Test del DQNModel standard
    print("\nTest del DQNModel standard")
    model = DQNModel(state_dim, action_dim, hidden_dim)
    
    # Input di test
    x = torch.randn(batch_size, state_dim)
    
    # Forward pass
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    
    # Test salvataggio e caricamento
    test_path = "test_model.pt"
    model.save(test_path)
    
    loaded_model = DQNModel.load(test_path)
    print(f"Loaded model state_dim: {loaded_model.state_dim}, action_dim: {loaded_model.action_dim}")
    
    # Test del DuelingDQNModel
    print("\nTest del DuelingDQNModel")
    dueling_model = DuelingDQNModel(state_dim, action_dim, hidden_dim)
    
    # Forward pass
    dueling_output = dueling_model(x)
    print(f"Dueling output shape: {dueling_output.shape}")
    print(f"Dueling output: {dueling_output}")
    
    # Test del NoisyDQNModel
    print("\nTest del NoisyDQNModel")
    noisy_model = NoisyDQNModel(state_dim, action_dim, hidden_dim)
    
    # Forward pass
    noisy_output = noisy_model(x)
    print(f"Noisy output shape: {noisy_output.shape}")
    print(f"Noisy output: {noisy_output}")
    
    # Resetta il rumore
    noisy_model.reset_noise()
    print("Noise reset.")
    
    # Test del NoisyDuelingDQNModel
    print("\nTest del NoisyDuelingDQNModel")
    noisy_dueling_model = NoisyDuelingDQNModel(state_dim, action_dim, hidden_dim)
    
    # Forward pass
    noisy_dueling_output = noisy_dueling_model(x)
    print(f"Noisy Dueling output shape: {noisy_dueling_output.shape}")
    print(f"Noisy Dueling output: {noisy_dueling_output}")
    
    # Resetta il rumore
    noisy_dueling_model.reset_noise()
    print("Noise reset.")
    
    # Test della factory function
    print("\nTest della factory function")
    models = {
        'dqn': create_model('dqn', state_dim, action_dim, hidden_dim),
        'dueling': create_model('dueling', state_dim, action_dim, hidden_dim),
        'noisy': create_model('noisy', state_dim, action_dim, hidden_dim),
        'noisy_dueling': create_model('noisy_dueling', state_dim, action_dim, hidden_dim)
    }
    
    for name, model in models.items():
        output = model(x)
        print(f"{name.upper()} output shape: {output.shape}")
    
    # Pulizia
    import os
    if os.path.exists(test_path):
        os.remove(test_path) 