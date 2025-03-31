#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Buffer di Memoria
====================
Implementazioni di buffer di memoria per DQN.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import random
import numpy as np
import torch
from collections import deque, namedtuple
from typing import List, Dict, Any, Tuple, Optional, Union
import pickle
from pathlib import Path

# Definisci la struttura di una transizione nella memoria
Experience = namedtuple('Experience', 
                        ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    """
    Buffer di memoria standard per esperienze del DQN.
    
    Memorizza le transizioni (stato, azione, nuovo stato, reward, done)
    e fornisce un metodo per campionare batch casuali uniformi per l'allenamento.
    """
    
    def __init__(self, capacity: int, state_dim: int = None, 
                 device: torch.device = None):
        """
        Inizializza il buffer di replay.
        
        Args:
            capacity (int): Capacità massima del buffer
            state_dim (int, optional): Dimensione dello stato
            device (torch.device, optional): Dispositivo per i tensori (CUDA/CPU)
        """
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.state_dim = state_dim
        self.device = device if device else torch.device("cpu")
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
        # Verifica e correggi i tipi e le forme degli stati
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"Errore nella conversione di state: {e}, tipo={type(state)}")
                if hasattr(state, '__len__'):
                    state = np.array([float(x) for x in state], dtype=np.float32)
                else:
                    state = np.array([float(state)], dtype=np.float32)
        
        if not isinstance(next_state, np.ndarray):
            try:
                next_state = np.array(next_state, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"Errore nella conversione di next_state: {e}, tipo={type(next_state)}")
                if hasattr(next_state, '__len__'):
                    next_state = np.array([float(x) for x in next_state], dtype=np.float32)
                else:
                    next_state = np.array([float(next_state)], dtype=np.float32)
                    
        # Verifica che le dimensioni degli stati corrispondano a state_dim se definito
        if self.state_dim is not None:
            if len(state.shape) == 1 and state.shape[0] != self.state_dim:
                print(f"AVVISO: dimensione stato non corrisponde a state_dim ({state.shape[0]} vs {self.state_dim})")
                # Prova a correggere
                if state.size == self.state_dim:
                    state = state.reshape(self.state_dim)
                else:
                    # Non possiamo correggere, ma possiamo avvisare
                    print(f"Impossibile correggere la dimensione dello stato")
                    
            if len(next_state.shape) == 1 and next_state.shape[0] != self.state_dim:
                print(f"AVVISO: dimensione next_state non corrisponde a state_dim ({next_state.shape[0]} vs {self.state_dim})")
                # Prova a correggere
                if next_state.size == self.state_dim:
                    next_state = next_state.reshape(self.state_dim)
                else:
                    # Non possiamo correggere, ma possiamo avvisare
                    print(f"Impossibile correggere la dimensione del next_state")
        
        # Debug occasionale
        if random.random() < 0.001:  # 0.1% delle volte
            print(f"Debug Push - state={state.shape}, next_state={next_state.shape}, action={action}, reward={reward}")
                    
        # Crea una nuova esperienza
        experience = Experience(state, action, next_state, reward, done)
        
        # Aggiunge l'esperienza alla memoria
        self.memory.append(experience)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Campiona un batch di esperienze dalla memoria (campionamento uniforme).
        
        Args:
            batch_size (int): Dimensione del batch da campionare
            
        Returns:
            tuple: Tuple di tensori (stati, azioni, stati successivi, rewards, done flags)
        """
        # Limita il batch size alla dimensione della memoria
        batch_size = min(batch_size, len(self.memory))
        
        # Campiona esperienze casuali
        experiences = random.sample(list(self.memory), batch_size)
        
        # Estrai componenti
        batch = Experience(*zip(*experiences))
        
        # Controllo debug
        if random.random() < 0.01:  # Log occasionale
            # Stampa informazioni sullo stato per debug
            print(f"Debug Buffer - Esempio stato: tipo={type(batch.state[0])}, forma={np.array(batch.state[0]).shape}")
        
        # Converti in array numpy per manipolazione
        states_list = list(batch.state)
        next_states_list = list(batch.next_state)
        
        # Determina la dimensione corretta dello stato
        state_dim = self.state_dim
        if state_dim is None:
            # Se state_dim non è noto, prova a dedurlo dal primo stato che sembra completo
            for s in states_list:
                if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] > 1:
                    state_dim = s.shape[0]
                    break
            
            if state_dim is None:
                # Fallback a una dimensione tipica se non è stato possibile dedurre
                state_dim = 13  # Dimensione tipica per Snake
                print(f"AVVISO: impossibile dedurre state_dim, usando valore predefinito {state_dim}")
                
        # Controllo pre-processing - conta quanti stati sono corretti
        correct_states = sum(1 for s in states_list if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] == state_dim)
        correct_next_states = sum(1 for s in next_states_list if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] == state_dim)
        
        if random.random() < 0.01:  # Log occasionale
            print(f"Debug Buffer - Stati corretti: {correct_states}/{len(states_list)}, Next stati corretti: {correct_next_states}/{len(next_states_list)}")
        
        # Se la maggior parte degli stati è incompleta, c'è un problema più serio
        if correct_states < len(states_list) // 2 or correct_next_states < len(next_states_list) // 2:
            print(f"AVVISO: La maggior parte degli stati nel batch è incompleta. Questo potrebbe indicare un problema nel flusso di dati.")
        
        # Assicurati che ogni stato abbia dimensione corretta
        for i in range(len(states_list)):
            # Verifica lo stato
            if not isinstance(states_list[i], np.ndarray):
                states_list[i] = np.array(states_list[i], dtype=np.float32)
            
            # Se lo stato è scalare o non ha la forma corretta, crea un nuovo stato con dimensione corretta
            if len(states_list[i].shape) == 0 or (len(states_list[i].shape) == 1 and states_list[i].shape[0] != state_dim):
                # Se abbiamo almeno un elemento, usiamolo come primo valore
                if states_list[i].size > 0:
                    first_value = float(states_list[i].flat[0])
                else:
                    first_value = 0.0  # Valore di default
                
                # Crea un nuovo array con dimensione corretta
                new_state = np.zeros(state_dim, dtype=np.float32)
                new_state[0] = first_value  # Usa il valore disponibile come primo elemento
                
                # Riempimento casuale per il resto dell'array per evitare bias
                if state_dim > 1:
                    new_state[1:] = np.random.randn(state_dim - 1) * 0.01
                
                # Sostituisci l'array originale
                states_list[i] = new_state
            
            # Verifica next_state con lo stesso approccio
            if not isinstance(next_states_list[i], np.ndarray):
                next_states_list[i] = np.array(next_states_list[i], dtype=np.float32)
            
            if len(next_states_list[i].shape) == 0 or (len(next_states_list[i].shape) == 1 and next_states_list[i].shape[0] != state_dim):
                # Se abbiamo almeno un elemento, usiamolo come primo valore
                if next_states_list[i].size > 0:
                    first_value = float(next_states_list[i].flat[0])
                else:
                    first_value = 0.0
                
                # Copia lo stato corrente se disponibile con la forma corretta
                if len(states_list[i].shape) == 1 and states_list[i].shape[0] == state_dim:
                    new_next_state = states_list[i].copy()
                    # Modifica solo il primo valore
                    new_next_state[0] = first_value
                else:
                    # Altrimenti crea un nuovo array
                    new_next_state = np.zeros(state_dim, dtype=np.float32)
                    new_next_state[0] = first_value
                    
                    # Riempimento casuale per il resto dell'array
                    if state_dim > 1:
                        new_next_state[1:] = np.random.randn(state_dim - 1) * 0.01
                
                # Sostituisci l'array originale
                next_states_list[i] = new_next_state
        
        # Converti in array
        try:
            states_array = np.array(states_list, dtype=np.float32)
            next_states_array = np.array(next_states_list, dtype=np.float32)
            
            # Controllo di debug
            if random.random() < 0.01:  # Log occasionale
                print(f"Debug Buffer - Forme array: states={states_array.shape}, next_states={next_states_array.shape}")
                
            # Verifica finale della forma
            if len(states_array.shape) != 2 or states_array.shape[1] != state_dim:
                print(f"AVVISO: states_array ha forma problematica {states_array.shape}, correzione forzata")
                states_array = states_array.reshape(batch_size, state_dim)
                
            if len(next_states_array.shape) != 2 or next_states_array.shape[1] != state_dim:
                print(f"AVVISO: next_states_array ha forma problematica {next_states_array.shape}, correzione forzata")
                next_states_array = next_states_array.reshape(batch_size, state_dim)
        
        except (ValueError, TypeError) as e:
            # In caso di errore, costruisci manualmente gli array
            print(f"Errore nella conversione in array: {e}. Costruzione manuale...")
            
            # Costruisci manualmente con dimesione fissa state_dim
            states_array = np.zeros((batch_size, state_dim), dtype=np.float32)
            next_states_array = np.zeros((batch_size, state_dim), dtype=np.float32)
            
            for i in range(batch_size):
                # Converti stati
                if i < len(states_list):
                    s = states_list[i]
                    if isinstance(s, np.ndarray) and s.size > 0:
                        # Copia il primo valore e genera casualmente il resto
                        states_array[i, 0] = s.flat[0]
                        if state_dim > 1:
                            states_array[i, 1:] = np.random.randn(state_dim - 1) * 0.01
                    else:
                        # Genera completamente casuale
                        states_array[i] = np.random.randn(state_dim) * 0.01
                
                # Converti next_states
                if i < len(next_states_list):
                    ns = next_states_list[i]
                    if isinstance(ns, np.ndarray) and ns.size > 0:
                        # Copia il primo valore e genera casualmente il resto
                        next_states_array[i, 0] = ns.flat[0]
                        if state_dim > 1:
                            next_states_array[i, 1:] = states_array[i, 1:] * 1.01  # Leggera variazione
                    else:
                        # Copia lo stato corrente con leggere variazioni
                        next_states_array[i] = states_array[i] * 1.01
            
            print(f"Array costruiti manualmente: states={states_array.shape}, next_states={next_states_array.shape}")
        
        # Converti in tensori PyTorch
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).unsqueeze(1).to(self.device)
        
        # Controllo finale della forma
        if random.random() < 0.01:  # Log occasionale
            print(f"Debug Buffer - Forme finali: states={states.shape}, next_states={next_states.shape}")
        
        # Verifica ASSOLUTA: se le forme non corrispondono, correggi
        if states.shape[1] != state_dim or next_states.shape[1] != state_dim:
            print(f"ERRORE CRITICO: Forme tensori non valide. Correzione forzata.")
            
            if states.shape[1] != state_dim:
                temp_states = torch.zeros(batch_size, state_dim, device=self.device)
                if states.shape[1] > 0:
                    # Copia tutti i valori che possiamo
                    min_dim = min(states.shape[1], state_dim)
                    temp_states[:, :min_dim] = states[:, :min_dim]
                states = temp_states
            
            if next_states.shape[1] != state_dim:
                temp_next_states = torch.zeros(batch_size, state_dim, device=self.device)
                if next_states.shape[1] > 0:
                    # Copia tutti i valori che possiamo
                    min_dim = min(next_states.shape[1], state_dim)
                    temp_next_states[:, :min_dim] = next_states[:, :min_dim]
                # Se possible, copia anche dallo stato corrente per avere valori sensati
                if states.shape[1] == state_dim:
                    temp_next_states = states.clone()
                    # Aggiungi una piccola variazione per evitare stato identico
                    if temp_next_states.shape[1] > 0:
                        temp_next_states[:, 0] = next_states[:, 0] if next_states.shape[1] > 0 else temp_next_states[:, 0] * 1.01
                next_states = temp_next_states
            
            print(f"Forme corrette: states={states.shape}, next_states={next_states.shape}")
        
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
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'memory': list(self.memory),
                'capacity': self.capacity,
                'state_dim': self.state_dim
            }, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'ReplayBuffer':
        """
        Carica un buffer di replay da disco.
        
        Args:
            path (str): Percorso del file di salvataggio
            device (torch.device, optional): Dispositivo per i tensori
            
        Returns:
            ReplayBuffer: Buffer caricato
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        buffer = cls(capacity=data['capacity'], state_dim=data['state_dim'], device=device)
        buffer.memory = deque(data['memory'], maxlen=buffer.capacity)
        
        return buffer


class PrioritizedReplayBuffer:
    """
    Buffer di memoria con campionamento prioritizzato.
    
    Assegna priorità alle transizioni in base all'errore TD per campionare
    più frequentemente le esperienze con maggiore potenziale di apprendimento.
    """
    
    def __init__(self, capacity: int, state_dim: int = None, 
                 device: torch.device = None, alpha: float = 0.6, 
                 beta: float = 0.4, beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        Inizializza il buffer di replay prioritizzato.
        
        Args:
            capacity (int): Capacità massima del buffer
            state_dim (int, optional): Dimensione dello stato
            device (torch.device, optional): Dispositivo per i tensori
            alpha (float): Esponente che determina quanto la priorità influenza il campionamento
            beta (float): Esponente per la correzione dell'importance sampling
            beta_increment (float): Incremento di beta ad ogni campionamento
            epsilon (float): Piccolo valore per evitare priorità zero
        """
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.state_dim = state_dim
        self.device = device if device else torch.device("cpu")
        self.position = 0
        self.size = 0
        
        # Parametri per il campionamento prioritizzato
        self.alpha = alpha  # Quanto la priorità influenza il campionamento (0 = uniforme, 1 = completamente prioritizzato)
        self.beta = beta    # Correzione dell'importance sampling (0 = nessuna correzione, 1 = completa correzione)
        self.beta_increment = beta_increment  # Incremento di beta ad ogni campionamento
        self.epsilon = epsilon  # Piccolo valore per evitare priorità zero
        
        # Priorità massima iniziale
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, 
             reward: float, done: bool) -> None:
        """
        Aggiunge una transizione alla memoria con priorità massima.
        
        Args:
            state (np.ndarray): Stato corrente
            action (int): Azione intrapresa
            next_state (np.ndarray): Stato successivo
            reward (float): Reward ottenuto
            done (bool): Flag che indica se l'episodio è terminato
        """
        # Verifica e correggi i tipi e le forme degli stati
        if not isinstance(state, np.ndarray):
            try:
                state = np.array(state, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"Errore nella conversione di state: {e}, tipo={type(state)}")
                if hasattr(state, '__len__'):
                    state = np.array([float(x) for x in state], dtype=np.float32)
                else:
                    state = np.array([float(state)], dtype=np.float32)
        
        if not isinstance(next_state, np.ndarray):
            try:
                next_state = np.array(next_state, dtype=np.float32)
            except (ValueError, TypeError) as e:
                print(f"Errore nella conversione di next_state: {e}, tipo={type(next_state)}")
                if hasattr(next_state, '__len__'):
                    next_state = np.array([float(x) for x in next_state], dtype=np.float32)
                else:
                    next_state = np.array([float(next_state)], dtype=np.float32)
                    
        # Verifica che le dimensioni degli stati corrispondano a state_dim se definito
        if self.state_dim is not None:
            if len(state.shape) == 1 and state.shape[0] != self.state_dim:
                print(f"AVVISO: dimensione stato non corrisponde a state_dim ({state.shape[0]} vs {self.state_dim})")
                # Prova a correggere
                if state.size == self.state_dim:
                    state = state.reshape(self.state_dim)
                else:
                    # Non possiamo correggere, ma possiamo avvisare
                    print(f"Impossibile correggere la dimensione dello stato")
                    
            if len(next_state.shape) == 1 and next_state.shape[0] != self.state_dim:
                print(f"AVVISO: dimensione next_state non corrisponde a state_dim ({next_state.shape[0]} vs {self.state_dim})")
                # Prova a correggere
                if next_state.size == self.state_dim:
                    next_state = next_state.reshape(self.state_dim)
                else:
                    # Non possiamo correggere, ma possiamo avvisare
                    print(f"Impossibile correggere la dimensione del next_state")
        
        # Debug occasionale
        if random.random() < 0.001:  # 0.1% delle volte
            print(f"Debug Pri-Push - state={state.shape}, next_state={next_state.shape}, action={action}, reward={reward}")
        
        # Crea una nuova esperienza
        experience = Experience(state, action, next_state, reward, done)
        
        # Calcola la posizione nel buffer
        if len(self.memory) < self.capacity:
            self.memory.append(experience)
        else:
            self.memory[self.position] = experience
            
        # Assegna la priorità massima alle nuove esperienze
        self.priorities[self.position] = self.max_priority
        
        # Aggiorna la posizione
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Campiona un batch di esperienze dalla memoria con priorità.
        
        Args:
            batch_size (int): Dimensione del batch da campionare
            
        Returns:
            tuple: Tuple di tensori (stati, azioni, stati successivi, rewards, done flags, weights, indices)
        """
        # Limita il batch size alla dimensione della memoria
        batch_size = min(batch_size, self.size)
        
        # Calcola le probabilità di campionamento in base alle priorità
        # Applica l'esponente alpha alle priorità
        priorities = self.priorities[:self.size] ** self.alpha
        
        # Normalizza le probabilità
        probs = priorities / np.sum(priorities)
        
        # Campiona gli indici in base alle probabilità
        indices = np.random.choice(self.size, batch_size, replace=False, p=probs)
        
        # Campiona le esperienze
        experiences = [self.memory[idx] for idx in indices]
        
        # Calcola i pesi per l'importance sampling
        # Usa il massimo weight per normalizzare
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # Incrementa beta (avvicina a 1 col tempo)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Estrai componenti
        batch = Experience(*zip(*experiences))
        
        # Controllo debug
        if random.random() < 0.01:  # Log occasionale
            # Stampa informazioni sullo stato per debug
            print(f"Debug Pri-Buffer - Esempio stato: tipo={type(batch.state[0])}, forma={np.array(batch.state[0]).shape}")
        
        # Converti in array numpy per manipolazione
        states_list = list(batch.state)
        next_states_list = list(batch.next_state)
        
        # Determina la dimensione corretta dello stato
        state_dim = self.state_dim
        if state_dim is None:
            # Se state_dim non è noto, prova a dedurlo dal primo stato che sembra completo
            for s in states_list:
                if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] > 1:
                    state_dim = s.shape[0]
                    break
            
            if state_dim is None:
                # Fallback a una dimensione tipica se non è stato possibile dedurre
                state_dim = 13  # Dimensione tipica per Snake
                print(f"AVVISO: impossibile dedurre state_dim, usando valore predefinito {state_dim}")
        
        # Controllo pre-processing - conta quanti stati sono corretti
        correct_states = sum(1 for s in states_list if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] == state_dim)
        correct_next_states = sum(1 for s in next_states_list if isinstance(s, np.ndarray) and len(s.shape) == 1 and s.shape[0] == state_dim)
        
        if random.random() < 0.01:  # Log occasionale
            print(f"Debug Pri-Buffer - Stati corretti: {correct_states}/{len(states_list)}, Next stati corretti: {correct_next_states}/{len(next_states_list)}")
        
        # Se la maggior parte degli stati è incompleta, c'è un problema più serio
        if correct_states < len(states_list) // 2 or correct_next_states < len(next_states_list) // 2:
            print(f"AVVISO: La maggior parte degli stati nel batch è incompleta. Questo potrebbe indicare un problema nel flusso di dati.")
        
        # Assicurati che ogni stato abbia dimensione corretta
        for i in range(len(states_list)):
            # Verifica lo stato
            if not isinstance(states_list[i], np.ndarray):
                states_list[i] = np.array(states_list[i], dtype=np.float32)
            
            # Se lo stato è scalare o non ha la forma corretta, crea un nuovo stato con dimensione corretta
            if len(states_list[i].shape) == 0 or (len(states_list[i].shape) == 1 and states_list[i].shape[0] != state_dim):
                # Se abbiamo almeno un elemento, usiamolo come primo valore
                if states_list[i].size > 0:
                    first_value = float(states_list[i].flat[0])
                else:
                    first_value = 0.0  # Valore di default
                
                # Crea un nuovo array con dimensione corretta
                new_state = np.zeros(state_dim, dtype=np.float32)
                new_state[0] = first_value  # Usa il valore disponibile come primo elemento
                
                # Riempimento casuale per il resto dell'array per evitare bias
                if state_dim > 1:
                    new_state[1:] = np.random.randn(state_dim - 1) * 0.01
                
                # Sostituisci l'array originale
                states_list[i] = new_state
            
            # Verifica next_state con lo stesso approccio
            if not isinstance(next_states_list[i], np.ndarray):
                next_states_list[i] = np.array(next_states_list[i], dtype=np.float32)
            
            if len(next_states_list[i].shape) == 0 or (len(next_states_list[i].shape) == 1 and next_states_list[i].shape[0] != state_dim):
                # Se abbiamo almeno un elemento, usiamolo come primo valore
                if next_states_list[i].size > 0:
                    first_value = float(next_states_list[i].flat[0])
                else:
                    first_value = 0.0
                
                # Copia lo stato corrente se disponibile con la forma corretta
                if len(states_list[i].shape) == 1 and states_list[i].shape[0] == state_dim:
                    new_next_state = states_list[i].copy()
                    # Modifica solo il primo valore
                    new_next_state[0] = first_value
                else:
                    # Altrimenti crea un nuovo array
                    new_next_state = np.zeros(state_dim, dtype=np.float32)
                    new_next_state[0] = first_value
                    
                    # Riempimento casuale per il resto dell'array
                    if state_dim > 1:
                        new_next_state[1:] = np.random.randn(state_dim - 1) * 0.01
                
                # Sostituisci l'array originale
                next_states_list[i] = new_next_state
        
        # Converti in array
        try:
            states_array = np.array(states_list, dtype=np.float32)
            next_states_array = np.array(next_states_list, dtype=np.float32)
            
            # Controllo di debug
            if random.random() < 0.01:  # Log occasionale
                print(f"Debug Pri-Buffer - Forme array: states={states_array.shape}, next_states={next_states_array.shape}")
                
            # Verifica finale della forma
            if len(states_array.shape) != 2 or states_array.shape[1] != state_dim:
                print(f"AVVISO: states_array ha forma problematica {states_array.shape}, correzione forzata")
                states_array = states_array.reshape(batch_size, state_dim)
                
            if len(next_states_array.shape) != 2 or next_states_array.shape[1] != state_dim:
                print(f"AVVISO: next_states_array ha forma problematica {next_states_array.shape}, correzione forzata")
                next_states_array = next_states_array.reshape(batch_size, state_dim)
        
        except (ValueError, TypeError) as e:
            # In caso di errore, costruisci manualmente gli array
            print(f"Errore nella conversione in array: {e}. Costruzione manuale...")
            
            # Costruisci manualmente con dimesione fissa state_dim
            states_array = np.zeros((batch_size, state_dim), dtype=np.float32)
            next_states_array = np.zeros((batch_size, state_dim), dtype=np.float32)
            
            for i in range(batch_size):
                # Converti stati
                if i < len(states_list):
                    s = states_list[i]
                    if isinstance(s, np.ndarray) and s.size > 0:
                        # Copia il primo valore e genera casualmente il resto
                        states_array[i, 0] = s.flat[0]
                        if state_dim > 1:
                            states_array[i, 1:] = np.random.randn(state_dim - 1) * 0.01
                    else:
                        # Genera completamente casuale
                        states_array[i] = np.random.randn(state_dim) * 0.01
                
                # Converti next_states
                if i < len(next_states_list):
                    ns = next_states_list[i]
                    if isinstance(ns, np.ndarray) and ns.size > 0:
                        # Copia il primo valore e genera casualmente il resto
                        next_states_array[i, 0] = ns.flat[0]
                        if state_dim > 1:
                            next_states_array[i, 1:] = states_array[i, 1:] * 1.01  # Leggera variazione
                    else:
                        # Copia lo stato corrente con leggere variazioni
                        next_states_array[i] = states_array[i] * 1.01
            
            print(f"Array costruiti manualmente: states={states_array.shape}, next_states={next_states_array.shape}")
        
        # Converti in tensori PyTorch
        states = torch.FloatTensor(states_array).to(self.device)
        actions = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(batch.reward)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states_array).to(self.device)
        dones = torch.FloatTensor(np.array(batch.done, dtype=np.float32)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Controllo finale della forma
        if random.random() < 0.01:  # Log occasionale
            print(f"Debug Pri-Buffer - Forme finali: states={states.shape}, next_states={next_states.shape}")
        
        # Verifica ASSOLUTA: se le forme non corrispondono, correggi
        if states.shape[1] != state_dim or next_states.shape[1] != state_dim:
            print(f"ERRORE CRITICO: Forme tensori non valide. Correzione forzata.")
            
            if states.shape[1] != state_dim:
                temp_states = torch.zeros(batch_size, state_dim, device=self.device)
                if states.shape[1] > 0:
                    # Copia tutti i valori che possiamo
                    min_dim = min(states.shape[1], state_dim)
                    temp_states[:, :min_dim] = states[:, :min_dim]
                states = temp_states
            
            if next_states.shape[1] != state_dim:
                temp_next_states = torch.zeros(batch_size, state_dim, device=self.device)
                if next_states.shape[1] > 0:
                    # Copia tutti i valori che possiamo
                    min_dim = min(next_states.shape[1], state_dim)
                    temp_next_states[:, :min_dim] = next_states[:, :min_dim]
                # Se possible, copia anche dallo stato corrente per avere valori sensati
                if states.shape[1] == state_dim:
                    temp_next_states = states.clone()
                    # Aggiungi una piccola variazione per evitare stato identico
                    if temp_next_states.shape[1] > 0:
                        temp_next_states[:, 0] = next_states[:, 0] if next_states.shape[1] > 0 else temp_next_states[:, 0] * 1.01
                next_states = temp_next_states
            
            print(f"Forme corrette: states={states.shape}, next_states={next_states.shape}")
        
        return states, actions, next_states, rewards, dones, weights, indices
    
    def update_priorities(self, indices: Union[List[int], np.ndarray], 
                          priorities: Union[List[float], np.ndarray]) -> None:
        """
        Aggiorna le priorità delle transizioni.
        
        Args:
            indices (Union[List[int], np.ndarray]): Indici delle transizioni
            priorities (Union[List[float], np.ndarray]): Nuove priorità
        """
        for idx, priority in zip(indices, priorities):
            # Aggiungi epsilon per evitare priorità zero
            priority = float(priority) + self.epsilon
            
            # Aggiorna il valore massimo di priorità
            self.max_priority = max(self.max_priority, priority)
            
            # Aggiorna la priorità
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Restituisce il numero di esperienze nel buffer."""
        return self.size
    
    def can_sample(self, batch_size: int) -> bool:
        """Verifica se è possibile campionare un batch di dimensione specificata."""
        return self.size >= batch_size
    
    def save(self, path: str) -> None:
        """
        Salva il buffer di replay prioritizzato su disco.
        
        Args:
            path (str): Percorso del file di salvataggio
        """
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump({
                'memory': self.memory[:self.size],
                'priorities': self.priorities[:self.size],
                'capacity': self.capacity,
                'state_dim': self.state_dim,
                'alpha': self.alpha,
                'beta': self.beta,
                'beta_increment': self.beta_increment,
                'epsilon': self.epsilon,
                'max_priority': self.max_priority
            }, f)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PrioritizedReplayBuffer':
        """
        Carica un buffer di replay prioritizzato da disco.
        
        Args:
            path (str): Percorso del file di salvataggio
            device (torch.device, optional): Dispositivo per i tensori
            
        Returns:
            PrioritizedReplayBuffer: Buffer caricato
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        buffer = cls(
            capacity=data['capacity'], 
            state_dim=data['state_dim'], 
            device=device,
            alpha=data['alpha'],
            beta=data['beta'],
            beta_increment=data['beta_increment'],
            epsilon=data['epsilon']
        )
        
        buffer.memory = data['memory']
        buffer.priorities[:len(data['priorities'])] = data['priorities']
        buffer.size = len(data['memory'])
        buffer.position = buffer.size % buffer.capacity
        buffer.max_priority = data['max_priority']
        
        return buffer

# Test standalone
if __name__ == "__main__":
    # Test dei buffer di memoria
    import torch
    
    print("Test dei buffer di memoria")
    
    # Test del ReplayBuffer standard
    print("\nTest del ReplayBuffer standard")
    buffer = ReplayBuffer(capacity=100, state_dim=4)
    
    # Aggiungi alcune transizioni di test
    for i in range(10):
        state = np.array([i, i+1, i+2, i+3], dtype=np.float32)
        action = i % 2
        next_state = np.array([i+1, i+2, i+3, i+4], dtype=np.float32)
        reward = float(i)
        done = (i == 9)
        
        buffer.push(state, action, next_state, reward, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    # Campiona un batch
    if buffer.can_sample(5):
        states, actions, next_states, rewards, dones = buffer.sample(5)
        print(f"Batch shape - States: {states.shape}, Actions: {actions.shape}")
        print(f"Sample - States:\n{states.numpy()}")
        print(f"Sample - Actions:\n{actions.numpy()}")
    
    # Test salvataggio e caricamento
    test_path = "test_buffer.pkl"
    buffer.save(test_path)
    
    loaded_buffer = ReplayBuffer.load(test_path)
    print(f"Loaded buffer size: {len(loaded_buffer)}")
    
    # Test del PrioritizedReplayBuffer
    print("\nTest del PrioritizedReplayBuffer")
    pri_buffer = PrioritizedReplayBuffer(capacity=100, state_dim=4, alpha=0.6, beta=0.4)
    
    # Aggiungi alcune transizioni di test
    for i in range(10):
        state = np.array([i, i+1, i+2, i+3], dtype=np.float32)
        action = i % 2
        next_state = np.array([i+1, i+2, i+3, i+4], dtype=np.float32)
        reward = float(i)
        done = (i == 9)
        
        pri_buffer.push(state, action, next_state, reward, done)
    
    print(f"Prioritized buffer size: {len(pri_buffer)}")
    
    # Campiona un batch
    if pri_buffer.can_sample(5):
        states, actions, next_states, rewards, dones, weights, indices = pri_buffer.sample(5)
        print(f"Batch shape - States: {states.shape}, Weights: {weights.shape}")
        print(f"Sample - States:\n{states.numpy()}")
        print(f"Sample - Weights:\n{weights.numpy()}")
        
        # Aggiorna le priorità
        new_priorities = np.random.rand(len(indices))
        pri_buffer.update_priorities(indices, new_priorities)
        print(f"Updated priorities for indices: {indices}")
    
    # Test salvataggio e caricamento
    test_pri_path = "test_pri_buffer.pkl"
    pri_buffer.save(test_pri_path)
    
    loaded_pri_buffer = PrioritizedReplayBuffer.load(test_pri_path)
    print(f"Loaded prioritized buffer size: {len(loaded_pri_buffer)}")
    
    # Pulizia
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
    if os.path.exists(test_pri_path):
        os.remove(test_pri_path) 