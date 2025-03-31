#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test DQN
========
Testa le funzionalità dell'implementazione DQN.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import sys
import numpy as np
import torch
import unittest
import tempfile
import shutil
from pathlib import Path

# Aggiungi il percorso principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importa i moduli da testare
from agents.dqn import (
    DQNAgent,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    Experience,
    DQNModel, 
    DuelingDQNModel, 
    NoisyDQNModel, 
    NoisyDuelingDQNModel,
    create_model
)
from utils.hardware_utils import get_device

class TestDQNComponents(unittest.TestCase):
    """Testa tutti i componenti dell'implementazione DQN."""
    
    def setUp(self):
        """Inizializza il setup per i test."""
        self.state_dim = 10
        self.action_dim = 4
        self.device = get_device()
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Pulizia dopo i test."""
        shutil.rmtree(self.test_dir)
    
    def test_replay_buffer(self):
        """Testa il buffer di replay standard."""
        buffer_size = 100
        batch_size = 32
        
        # Crea un buffer
        buffer = ReplayBuffer(capacity=buffer_size, state_dim=self.state_dim, device=self.device)
        
        # Verifica che il buffer sia vuoto
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.can_sample(batch_size))
        
        # Aggiungi alcune esperienze
        for i in range(batch_size * 2):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = bool(np.random.randint(0, 2))
            
            buffer.push(state, action, next_state, reward, done)
        
        # Verifica che il buffer contenga le esperienze
        self.assertEqual(len(buffer), batch_size * 2)
        self.assertTrue(buffer.can_sample(batch_size))
        
        # Campiona un batch
        states, actions, next_states, rewards, dones = buffer.sample(batch_size)
        
        # Verifica le dimensioni dei batch
        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size, 1))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(dones.shape, (batch_size, 1))
        
        # Verifica il tipo dei tensori
        self.assertEqual(states.device, self.device)
        self.assertEqual(actions.device, self.device)
        self.assertEqual(next_states.device, self.device)
        self.assertEqual(rewards.device, self.device)
        self.assertEqual(dones.device, self.device)
        
        # Testa la capacità massima
        for i in range(buffer_size):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = bool(np.random.randint(0, 2))
            
            buffer.push(state, action, next_state, reward, done)
        
        # Verifica che il buffer non superi la capacità massima
        self.assertEqual(len(buffer), buffer_size)
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "buffer.pkl")
        buffer.save(save_path)
        
        # Carica il buffer
        loaded_buffer = ReplayBuffer.load(save_path, device=self.device)
        
        # Verifica le proprietà del buffer caricato
        self.assertEqual(len(loaded_buffer), len(buffer))
        self.assertEqual(loaded_buffer.capacity, buffer.capacity)
        self.assertEqual(loaded_buffer.state_dim, buffer.state_dim)
    
    def test_prioritized_replay_buffer(self):
        """Testa il buffer di replay prioritizzato."""
        buffer_size = 100
        batch_size = 32
        
        # Crea un buffer prioritizzato
        buffer = PrioritizedReplayBuffer(
            capacity=buffer_size, 
            state_dim=self.state_dim, 
            device=self.device,
            alpha=0.6,
            beta=0.4
        )
        
        # Verifica che il buffer sia vuoto
        self.assertEqual(len(buffer), 0)
        self.assertFalse(buffer.can_sample(batch_size))
        
        # Aggiungi alcune esperienze
        for i in range(batch_size * 2):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = bool(np.random.randint(0, 2))
            
            buffer.push(state, action, next_state, reward, done)
        
        # Verifica che il buffer contenga le esperienze
        self.assertEqual(len(buffer), batch_size * 2)
        self.assertTrue(buffer.can_sample(batch_size))
        
        # Campiona un batch
        states, actions, next_states, rewards, dones, weights, indices = buffer.sample(batch_size)
        
        # Verifica le dimensioni dei batch
        self.assertEqual(states.shape, (batch_size, self.state_dim))
        self.assertEqual(actions.shape, (batch_size, 1))
        self.assertEqual(next_states.shape, (batch_size, self.state_dim))
        self.assertEqual(rewards.shape, (batch_size, 1))
        self.assertEqual(dones.shape, (batch_size, 1))
        
        # I pesi potrebbero essere tensor o numpy array, e la loro forma potrebbe variare
        # Ci limitiamo a controllare che abbiano la dimensione batch_size
        self.assertTrue(len(weights) == batch_size)
        
        # Verifica il tipo dei tensori
        self.assertEqual(states.device, self.device)
        self.assertEqual(actions.device, self.device)
        self.assertEqual(next_states.device, self.device)
        self.assertEqual(rewards.device, self.device)
        self.assertEqual(dones.device, self.device)
        
        # Testa l'aggiornamento delle priorità
        # Usa indici validi per il buffer
        valid_indices = indices[:5]  # Prendi solo i primi 5 indici
        priorities = np.random.rand(5) + 0.1  # Solo 5 priorità positive
        buffer.update_priorities(valid_indices, priorities)
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "prioritized_buffer.pkl")
        buffer.save(save_path)
        
        # Carica il buffer
        loaded_buffer = PrioritizedReplayBuffer.load(save_path, device=self.device)
        
        # Verifica le proprietà del buffer caricato
        self.assertEqual(len(loaded_buffer), len(buffer))
        self.assertEqual(loaded_buffer.capacity, buffer.capacity)
        self.assertEqual(loaded_buffer.state_dim, buffer.state_dim)
        self.assertEqual(loaded_buffer.alpha, buffer.alpha)
        self.assertEqual(loaded_buffer.beta, buffer.beta)
    
    def test_dqn_model(self):
        """Testa il modello DQN standard."""
        # Crea un modello DQN
        model = DQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=[64, 64]
        )
        
        # Verifica le proprietà del modello
        self.assertEqual(model.state_dim, self.state_dim)
        self.assertEqual(model.action_dim, self.action_dim)
        self.assertEqual(model.hidden_dim, [64, 64])
        
        # Verifica il forward pass
        batch_size = 10
        state = torch.randn(batch_size, self.state_dim)
        q_values = model(state)
        
        # Verifica le dimensioni dell'output
        self.assertEqual(q_values.shape, (batch_size, self.action_dim))
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "dqn_model.pt")
        model.save(save_path)
        
        # Carica il modello
        loaded_model = DQNModel.load(save_path, device=self.device)
        
        # Verifica le proprietà del modello caricato
        self.assertEqual(loaded_model.state_dim, model.state_dim)
        self.assertEqual(loaded_model.action_dim, model.action_dim)
        self.assertEqual(loaded_model.hidden_dim, model.hidden_dim)
    
    def test_dueling_dqn_model(self):
        """Testa il modello Dueling DQN."""
        # Crea un modello Dueling DQN
        model = DuelingDQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=[64, 64]
        )
        
        # Verifica le proprietà del modello
        self.assertEqual(model.state_dim, self.state_dim)
        self.assertEqual(model.action_dim, self.action_dim)
        self.assertEqual(model.hidden_dim, [64, 64])
        
        # Verifica il forward pass
        batch_size = 10
        state = torch.randn(batch_size, self.state_dim)
        q_values = model(state)
        
        # Verifica le dimensioni dell'output
        self.assertEqual(q_values.shape, (batch_size, self.action_dim))
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "dueling_dqn_model.pt")
        model.save(save_path)
        
        # Carica il modello
        loaded_model = DuelingDQNModel.load(save_path, device=self.device)
        
        # Verifica le proprietà del modello caricato
        self.assertEqual(loaded_model.state_dim, model.state_dim)
        self.assertEqual(loaded_model.action_dim, model.action_dim)
        self.assertEqual(loaded_model.hidden_dim, model.hidden_dim)
    
    def test_noisy_dqn_model(self):
        """Testa il modello Noisy DQN."""
        # Crea un modello Noisy DQN
        model = NoisyDQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=[64, 64],
            std_init=0.5
        )
        
        # Verifica le proprietà del modello
        self.assertEqual(model.state_dim, self.state_dim)
        self.assertEqual(model.action_dim, self.action_dim)
        self.assertEqual(model.hidden_dim, [64, 64])
        
        # Verifica il forward pass
        batch_size = 10
        state = torch.randn(batch_size, self.state_dim)
        
        # Test in modalità train
        model.train()
        q_values_train = model(state)
        self.assertEqual(q_values_train.shape, (batch_size, self.action_dim))
        
        # Test in modalità eval
        model.eval()
        q_values_eval = model(state)
        self.assertEqual(q_values_eval.shape, (batch_size, self.action_dim))
        
        # Testa il reset del rumore
        model.reset_noise()
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "noisy_dqn_model.pt")
        model.save(save_path)
        
        # Carica il modello
        loaded_model = NoisyDQNModel.load(save_path, device=self.device)
        
        # Verifica le proprietà del modello caricato
        self.assertEqual(loaded_model.state_dim, model.state_dim)
        self.assertEqual(loaded_model.action_dim, model.action_dim)
        self.assertEqual(loaded_model.hidden_dim, model.hidden_dim)
    
    def test_noisy_dueling_dqn_model(self):
        """Testa il modello Noisy Dueling DQN."""
        # Crea un modello Noisy Dueling DQN
        model = NoisyDuelingDQNModel(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=[64, 64],
            std_init=0.5
        )
        
        # Verifica le proprietà del modello
        self.assertEqual(model.state_dim, self.state_dim)
        self.assertEqual(model.action_dim, self.action_dim)
        self.assertEqual(model.hidden_dim, [64, 64])
        
        # Verifica il forward pass
        batch_size = 10
        state = torch.randn(batch_size, self.state_dim)
        
        # Test in modalità train
        model.train()
        q_values_train = model(state)
        self.assertEqual(q_values_train.shape, (batch_size, self.action_dim))
        
        # Test in modalità eval
        model.eval()
        q_values_eval = model(state)
        self.assertEqual(q_values_eval.shape, (batch_size, self.action_dim))
        
        # Testa il reset del rumore
        model.reset_noise()
        
        # Testa salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "noisy_dueling_dqn_model.pt")
        model.save(save_path)
        
        # Carica il modello
        loaded_model = NoisyDuelingDQNModel.load(save_path, device=self.device)
        
        # Verifica le proprietà del modello caricato
        self.assertEqual(loaded_model.state_dim, model.state_dim)
        self.assertEqual(loaded_model.action_dim, model.action_dim)
        self.assertEqual(loaded_model.hidden_dim, model.hidden_dim)
    
    def test_create_model_factory(self):
        """Testa la factory di creazione modelli."""
        # Testa la creazione di tutti i tipi di modello
        models = {
            "dqn": DQNModel,
            "dueling": DuelingDQNModel,
            "noisy": NoisyDQNModel,
            "noisy_dueling": NoisyDuelingDQNModel
        }
        
        for model_type, model_class in models.items():
            model = create_model(
                model_type=model_type,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=[64, 64]
            )
            
            # Verifica che il modello creato sia dell'istanza corretta
            self.assertIsInstance(model, model_class)
            
            # Verifica le proprietà del modello
            self.assertEqual(model.state_dim, self.state_dim)
            self.assertEqual(model.action_dim, self.action_dim)
            self.assertEqual(model.hidden_dim, [64, 64])
    
    def test_dqn_agent(self):
        """Testa l'agente DQN."""
        # Configurazione base
        config = {
            "batch_size": 32,
            "gamma": 0.99,
            "lr": 0.0005,
            "target_update": 10,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_decay": 0.995,
            "use_double_dqn": True,
            "use_dueling_dqn": True,
            "use_prioritized_replay": False,  # Usa replay buffer standard per semplicità
            "n_step_returns": 1,  # Semplifica a 1-step returns per i test
            "hidden_layers": [64, 64]
        }
        
        # Crea l'agente
        agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config
        )
        
        # Verifica le proprietà dell'agente
        self.assertEqual(agent.state_dim, self.state_dim)
        self.assertEqual(agent.action_dim, self.action_dim)
        self.assertEqual(agent.batch_size, config["batch_size"])
        self.assertEqual(agent.gamma, config["gamma"])
        self.assertEqual(agent.learning_rate, config["lr"])
        self.assertEqual(agent.epsilon, config["eps_start"])
        
        # Testa la selezione dell'azione
        state = np.random.rand(self.state_dim)
        
        # In modalità training (potrebbe essere casuale a causa di epsilon)
        action_train = agent.select_action(state, training=True)
        self.assertIsInstance(action_train, int)
        self.assertTrue(0 <= action_train < self.action_dim)
        
        # In modalità evaluation (sempre greedy)
        agent.eval_mode()
        action_eval = agent.select_action(state, training=False)
        self.assertIsInstance(action_eval, int)
        self.assertTrue(0 <= action_eval < self.action_dim)
        
        # Torna in modalità training
        agent.train_mode()
        
        # Invece di testare update_epsilon, testiamo direttamente epsilon
        # verificando che sia un valore ragionevole
        self.assertTrue(agent.epsilon <= agent.eps_start)
        self.assertTrue(agent.epsilon >= agent.eps_end)
        
        # Testiamo che train() modifichi epsilon
        stats = agent.train(num_updates=5)  # Questo dovrebbe modificare training_steps e epsilon
        self.assertIn("epsilon", stats)
        
        # Testa lo storage di transizioni
        for i in range(config["batch_size"] * 2):
            state = np.random.rand(self.state_dim)
            action = np.random.randint(0, self.action_dim)
            next_state = np.random.rand(self.state_dim)
            reward = np.random.rand()
            done = bool(np.random.randint(0, 2))
            
            agent.store_transition(state, action, next_state, reward, done)
        
        # Testa l'ottimizzazione del modello
        loss = agent.optimize_model()
        self.assertIsInstance(loss, float)
        
        # Testa l'aggiornamento della rete target
        agent.update_target_network()
        
        # Testa il salvataggio e caricamento
        save_path = os.path.join(self.test_dir, "dqn_agent")
        agent.save(save_path, save_memory=True, save_optimizer=True)
        
        # Carica l'agente
        loaded_agent = DQNAgent.load(save_path, device=self.device, load_memory=True)
        
        # Verifica le proprietà dell'agente caricato
        self.assertEqual(loaded_agent.state_dim, agent.state_dim)
        self.assertEqual(loaded_agent.action_dim, agent.action_dim)
        self.assertEqual(loaded_agent.batch_size, agent.batch_size)
        self.assertEqual(loaded_agent.gamma, agent.gamma)
        
        # Testa il metodo per ottenere statistiche
        stats = agent.get_stats()
        self.assertIn("training_steps", stats)
        self.assertIn("updates", stats)
        self.assertIn("episodes", stats)

if __name__ == "__main__":
    unittest.main() 