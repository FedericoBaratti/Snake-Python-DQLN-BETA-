#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Modelli DQN
================
Implementa le diverse architetture di rete neurale per gli agenti DQN.

Autore: Federico Baratti
Versione: 1.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Any

class DQN(nn.Module):
    """
    Rete neurale di base per Deep Q-Network (DQN).
    
    Questa classe implementa una rete neurale completamente connessa (fully-connected)
    con un numero configurabile di layer nascosti e neuroni per layer.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], 
                 activation: str = "ReLU"):
        """
        Inizializza la rete neurale DQN.
        
        Args:
            input_dim (int): Dimensione dell'input (caratteristiche dello stato)
            output_dim (int): Dimensione dell'output (numero di azioni possibili)
            hidden_layers (List[int]): Lista con dimensioni dei layer nascosti
            activation (str): Funzione di attivazione ('ReLU', 'LeakyReLU', 'ELU', 'Tanh')
        """
        super(DQN, self).__init__()
        
        # Salva le dimensioni
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Seleziona la funzione di attivazione
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default
            
        # Costruisci i layer della rete
        self.layers = nn.ModuleList()
        
        # Layer di input
        self.layers.append(nn.Linear(input_dim, hidden_layers[0]))
        
        # Layer nascosti
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Layer di output
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)
        
        # Inizializzazione dei pesi per migliorare la convergenza
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializza i pesi della rete con Kaiming initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
        # Inizializza il layer di output con una varianza più piccola
        nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity='relu')
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagazione in avanti dell'input attraverso la rete.
        
        Args:
            x (torch.Tensor): Input alla rete (batch di stati)
            
        Returns:
            torch.Tensor: Output della rete (Q-values per ogni azione)
        """
        # Propaga l'input attraverso tutti i layer nascosti
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Layer di output (senza attivazione per Q-values)
        x = self.output_layer(x)
        
        return x
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Calcola i Q-values per uno stato dato.
        
        Args:
            state (np.ndarray): Stato del gioco
            
        Returns:
            np.ndarray: Q-values per ogni azione possibile
        """
        # Converti lo stato in un tensore
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)  # Aggiungi dimensione batch
        
        # Disabilita il calcolo del gradiente
        with torch.no_grad():
            q_values = self.forward(state)
        
        # Ritorna come numpy array
        return q_values.cpu().data.numpy()
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Seleziona un'azione usando ε-greedy policy.
        
        Args:
            state (np.ndarray): Stato corrente del gioco
            epsilon (float): Probabilità di scegliere un'azione casuale
            
        Returns:
            int: Azione scelta
        """
        # Scegli un'azione casuale con probabilità epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.output_dim)
        
        # Altrimenti, scegli l'azione con il Q-value più alto
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serializza il modello in un dizionario.
        
        Returns:
            Dict[str, Any]: Dizionario con i parametri del modello
        """
        return {
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers
        }
    
    @classmethod
    def deserialize(cls, serialized_model: Dict[str, Any]) -> 'DQN':
        """
        Crea un modello da un dizionario serializzato.
        
        Args:
            serialized_model (Dict[str, Any]): Dizionario con i parametri del modello
            
        Returns:
            DQN: Modello caricato
        """
        model = cls(
            input_dim=serialized_model["input_dim"],
            output_dim=serialized_model["output_dim"],
            hidden_layers=serialized_model["hidden_layers"]
        )
        model.load_state_dict(serialized_model["model_state"])
        return model


class DuelingDQN(nn.Module):
    """
    Dueling DQN che separa il valore di stato da advantage function.
    
    Questa architettura avanzata del DQN separa il calcolo del valore dello stato
    (V(s)) dall'advantage (A(s,a)) per ogni azione, migliorando la stima dei Q-values.
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int], 
                 activation: str = "ReLU"):
        """
        Inizializza la rete neurale Dueling DQN.
        
        Args:
            input_dim (int): Dimensione dell'input (caratteristiche dello stato)
            output_dim (int): Dimensione dell'output (numero di azioni possibili)
            hidden_layers (List[int]): Lista con dimensioni dei layer nascosti
            activation (str): Funzione di attivazione ('ReLU', 'LeakyReLU', 'ELU', 'Tanh')
        """
        super(DuelingDQN, self).__init__()
        
        # Salva le dimensioni
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        
        # Seleziona la funzione di attivazione
        if activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()  # Default
        
        # Feature layer comune
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layers[0]),
            self.activation
        )
        
        # Layers per il valore di stato
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            self.activation,
            nn.Linear(hidden_layers[1], 1)  # Output: V(s)
        )
        
        # Layers per l'advantage
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            self.activation,
            nn.Linear(hidden_layers[1], output_dim)  # Output: A(s,a)
        )
        
        # Inizializzazione dei pesi
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inizializza i pesi con Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Propagazione in avanti dell'input attraverso la rete.
        
        Args:
            x (torch.Tensor): Input alla rete (batch di stati)
            
        Returns:
            torch.Tensor: Output della rete (Q-values per ogni azione)
        """
        # Estrai le features dallo stato
        features = self.feature_layer(x)
        
        # Calcola il valore dello stato e l'advantage per ogni azione
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combina valore e advantage per ottenere i Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """
        Calcola i Q-values per uno stato dato.
        
        Args:
            state (np.ndarray): Stato del gioco
            
        Returns:
            np.ndarray: Q-values per ogni azione possibile
        """
        # Converti lo stato in un tensore
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)  # Aggiungi dimensione batch
        
        # Disabilita il calcolo del gradiente
        with torch.no_grad():
            q_values = self.forward(state)
        
        # Ritorna come numpy array
        return q_values.cpu().data.numpy()
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """
        Seleziona un'azione usando ε-greedy policy.
        
        Args:
            state (np.ndarray): Stato corrente del gioco
            epsilon (float): Probabilità di scegliere un'azione casuale
            
        Returns:
            int: Azione scelta
        """
        # Scegli un'azione casuale con probabilità epsilon
        if np.random.random() < epsilon:
            return np.random.randint(self.output_dim)
        
        # Altrimenti, scegli l'azione con il Q-value più alto
        q_values = self.get_q_values(state)
        return np.argmax(q_values)
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serializza il modello in un dizionario.
        
        Returns:
            Dict[str, Any]: Dizionario con i parametri del modello
        """
        return {
            "model_state": self.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_layers": self.hidden_layers
        }
    
    @classmethod
    def deserialize(cls, serialized_model: Dict[str, Any]) -> 'DuelingDQN':
        """
        Crea un modello da un dizionario serializzato.
        
        Args:
            serialized_model (Dict[str, Any]): Dizionario con i parametri del modello
            
        Returns:
            DuelingDQN: Modello caricato
        """
        model = cls(
            input_dim=serialized_model["input_dim"],
            output_dim=serialized_model["output_dim"],
            hidden_layers=serialized_model["hidden_layers"]
        )
        model.load_state_dict(serialized_model["model_state"])
        return model


def create_model(complexity: str, input_dim: int, output_dim: int) -> nn.Module:
    """
    Crea un modello DQN in base al livello di complessità specificato.
    
    Args:
        complexity (str): Livello di complessità ('base', 'avanzato', 'complesso', 'perfetto')
        input_dim (int): Dimensione dell'input (caratteristiche dello stato)
        output_dim (int): Dimensione dell'output (numero di azioni possibili)
        
    Returns:
        nn.Module: Modello DQN con l'architettura specificata
    """
    from dqn_agent.config import get_config
    
    # Ottieni la configurazione per il livello specificato
    config = get_config(complexity)
    
    # Determina l'architettura in base al livello di complessità
    if complexity in ["base", "avanzato"]:
        return DQN(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_layers=config["hidden_layers"],
            activation=config["activation"]
        )
    else:  # complesso, perfetto
        return DuelingDQN(
            input_dim=input_dim, 
            output_dim=output_dim, 
            hidden_layers=config["hidden_layers"],
            activation=config["activation"]
        )


if __name__ == "__main__":
    # Test dei modelli
    input_dim = 9  # Caratteristiche dello stato (esempio per Snake)
    output_dim = 3  # Azioni possibili (0: dritto, 1: destra, 2: sinistra)
    
    # Test DQN base
    model_base = create_model("base", input_dim, output_dim)
    print(f"Modello base: {model_base}")
    print(f"Numero di parametri: {sum(p.numel() for p in model_base.parameters())}")
    
    # Test con input
    test_input = torch.randn(32, input_dim)  # Batch di 32 stati
    output = model_base(test_input)
    print(f"Shape output: {output.shape}")  # Dovrebbe essere [32, 3]
    
    # Test DQN avanzato
    model_advanced = create_model("complesso", input_dim, output_dim)
    print(f"\nModello complesso: {model_advanced}")
    print(f"Numero di parametri: {sum(p.numel() for p in model_advanced.parameters())}")
    
    # Test con input
    output = model_advanced(test_input)
    print(f"Shape output: {output.shape}")  # Dovrebbe essere [32, 3] 