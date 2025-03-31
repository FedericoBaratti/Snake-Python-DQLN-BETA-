#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Integrazione Snake-DQN
===========================
Testa l'integrazione tra l'agente DQN e l'ambiente Snake.

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
from agents.dqn import DQNAgent
from core.environment import SnakeEnv as SnakeEnvironment

class TestSnakeDQNIntegration(unittest.TestCase):
    """Testa l'integrazione tra l'agente DQN e l'ambiente Snake."""
    
    def setUp(self):
        """Inizializza il setup per i test."""
        # Configurazione dell'ambiente
        self.env_config = {
            "grid_size": 10,
            "max_steps": 100,
            "reward_apple": 10.0,
            "reward_death": -10.0,
            "reward_step": -0.01,
            "reward_distance": True
        }
        
        # Crea l'ambiente
        self.env = SnakeEnvironment(config=self.env_config)
        
        # Dimensioni dello stato e dell'azione
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Configurazione dell'agente
        self.agent_config = {
            "batch_size": 32,
            "gamma": 0.99,
            "lr": 0.0005,
            "target_update": 10,
            "eps_start": 1.0,
            "eps_end": 0.01,
            "eps_decay": 0.995,
            "use_double_dqn": True,
            "use_dueling_dqn": True,
            "use_prioritized_replay": False,  # Usiamo standard replay per semplicità
            "n_step_returns": 1,
            "hidden_layers": [64, 64],
            "memory_size": 1000
        }
        
        # Crea l'agente
        self.agent = DQNAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.agent_config
        )
        
        # Directory temporanea per i salvataggi
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Pulizia dopo i test."""
        shutil.rmtree(self.test_dir)
    
    def test_environment_reset(self):
        """Testa il reset dell'ambiente."""
        # Reset dell'ambiente
        state, _ = self.env.reset()
        
        # Verifica che lo stato abbia la dimensione corretta
        self.assertEqual(len(state), self.state_dim)
    
    def test_environment_step(self):
        """Testa lo step dell'ambiente."""
        # Reset dell'ambiente
        state, _ = self.env.reset()
        
        # Esegui un'azione
        action = np.random.randint(0, self.action_dim)
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Verifica che lo stato successivo abbia la dimensione corretta
        self.assertEqual(len(next_state), self.state_dim)
        
        # Verifica che il reward sia un numero
        self.assertIsInstance(reward, (int, float))
        
        # Verifica che done sia un booleano
        self.assertIsInstance(done, bool)
        
        # Verifica che info sia un dizionario
        self.assertIsInstance(info, dict)
    
    def test_agent_action_selection(self):
        """Testa la selezione dell'azione da parte dell'agente."""
        # Reset dell'ambiente
        state, _ = self.env.reset()
        
        # Seleziona un'azione
        action = self.agent.select_action(state)
        
        # Verifica che l'azione sia valida
        self.assertIsInstance(action, int)
        self.assertTrue(0 <= action < self.action_dim)
    
    def test_full_episode(self):
        """Testa un episodio completo di interazione tra agente e ambiente."""
        # Reset dell'ambiente
        state, _ = self.env.reset()
        
        # Lista per tracciare i reward
        total_reward = 0
        steps = 0
        done = False
        
        # Esegui un episodio
        while not done and steps < self.env_config["max_steps"]:
            # Seleziona un'azione
            action = self.agent.select_action(state)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Memorizza la transizione
            self.agent.store_transition(state, action, next_state, reward, done)
            
            # Aggiorna lo stato
            state = next_state
            
            # Aggiorna il reward totale
            total_reward += reward
            steps += 1
            
            # Ottimizza il modello se è possibile
            if len(self.agent.memory) >= self.agent_config["batch_size"]:
                self.agent.optimize_model()
        
        # Verifica che l'episodio sia terminato
        self.assertTrue(done or steps >= self.env_config["max_steps"])
        
        # Verifica che il reward totale sia un numero
        self.assertIsInstance(total_reward, (int, float))
    
    def test_training_loop(self):
        """Testa un ciclo di training completo."""
        # Numero di episodi di training
        n_episodes = 5
        
        for episode in range(n_episodes):
            # Reset dell'ambiente
            state, _ = self.env.reset()
            
            # Lista per tracciare i reward
            total_reward = 0
            steps = 0
            done = False
            
            # Esegui un episodio
            while not done and steps < self.env_config["max_steps"]:
                # Seleziona un'azione
                action = self.agent.select_action(state)
                
                # Esegui l'azione
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Memorizza la transizione
                self.agent.store_transition(state, action, next_state, reward, done)
                
                # Aggiorna lo stato
                state = next_state
                
                # Aggiorna il reward totale
                total_reward += reward
                steps += 1
                
                # Ottimizza il modello se è possibile
                if len(self.agent.memory) >= self.agent_config["batch_size"]:
                    self.agent.optimize_model()
            
            # Aggiorna epsilon
            self.agent.update_epsilon()
            
            # Aggiorna la rete target periodicamente
            if episode % self.agent_config["target_update"] == 0:
                self.agent.update_target_network()
        
        # Verifica che il modello sia stato addestrato
        self.assertGreater(len(self.agent.memory), 0)
    
    def test_save_load_agent(self):
        """Testa il salvataggio e il caricamento dell'agente."""
        # Path per il salvataggio
        save_path = os.path.join(self.test_dir, "dqn_agent")
        
        # Esegui un episodio per popolare la memoria
        state, _ = self.env.reset()
        for _ in range(10):
            action = self.agent.select_action(state)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            self.agent.store_transition(state, action, next_state, reward, done)
            if done:
                state, _ = self.env.reset()
            else:
                state = next_state
            
            # Ottimizza il modello se è possibile
            if len(self.agent.memory) >= self.agent_config["batch_size"]:
                self.agent.optimize_model()
        
        # Salva l'agente
        self.agent.save(save_path, save_memory=True)
        
        # Carica l'agente in un nuovo agente
        new_agent = DQNAgent.load(save_path, load_memory=True)
        
        # Verifica che i parametri siano stati caricati correttamente
        self.assertEqual(len(new_agent.memory), len(self.agent.memory))
        
        # Testa che il nuovo agente possa eseguire azioni
        state, _ = self.env.reset()
        action = new_agent.select_action(state)
        self.assertTrue(0 <= action < self.action_dim)
    
    def test_evaluation(self):
        """Testa la valutazione dell'agente."""
        # Esegui un episodio in modalità di valutazione
        state, _ = self.env.reset()
        
        # Disabilita l'esplorazione
        old_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        
        # Esegui un episodio
        steps = 0
        total_reward = 0
        done = False
        
        while not done and steps < self.env_config["max_steps"]:
            # Seleziona un'azione (senza esplorazione)
            action = self.agent.select_action(state, training=False)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Aggiorna lo stato
            state = next_state
            
            # Aggiorna il reward totale
            total_reward += reward
            steps += 1
        
        # Ripristina epsilon
        self.agent.epsilon = old_epsilon
        
        # Verifica che la valutazione sia stata completata
        self.assertTrue(done or steps >= self.env_config["max_steps"])
        self.assertIsInstance(total_reward, (int, float))

if __name__ == "__main__":
    unittest.main() 