#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo di gioco
===============
Script per giocare a Snake con un agente DQN addestrato o manualmente.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import pygame
import matplotlib.pyplot as plt
from pathlib import Path
import json

from core.environment import SnakeEnv as SnakeEnvironment
from agents import get_agent
from utils.config import ConfigManager as Config
from utils.hardware_utils import get_device


def parse_args(args=None):
    """Analizza gli argomenti da linea di comando."""
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Giocare a Snake con un agente DQN')
    
    # Modalità di gioco
    parser.add_argument('--mode', type=str, choices=['ai', 'human', 'compare'], default='ai',
                      help='Modalità di gioco: ai, human, compare')
    
    # Parametri dell'agente
    parser.add_argument('--model_path', type=str, default=None,
                      help='Percorso del modello addestrato')
    parser.add_argument('--agent', type=str, default='dqn',
                      help='Tipo di agente da utilizzare')
    parser.add_argument('--model_path2', type=str, default=None,
                      help='Secondo modello per la modalità compare')
    
    # Parametri dell'ambiente
    parser.add_argument('--grid_size', type=int, default=10,
                      help='Dimensione della griglia')
    parser.add_argument('--speed', type=int, default=5,
                      help='Velocità del gioco (fps)')
    parser.add_argument('--max_steps', type=int, default=1000,
                      help='Numero massimo di passi per episodio')
    
    # Parametri di visualizzazione
    parser.add_argument('--cell_size', type=int, default=40,
                      help='Dimensione delle celle in pixel')
    parser.add_argument('--show_grid', action='store_true',
                      help='Mostra la griglia')
    parser.add_argument('--stats', action='store_true',
                      help='Mostra statistiche')
    
    # Altri parametri
    parser.add_argument('--device', type=str, default='auto',
                      help='Dispositivo (cuda:0, cpu)')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Numero di episodi da giocare')
    parser.add_argument('--debug', action='store_true',
                      help='Abilita modalità debug con output aggiuntivi')
    
    return parser.parse_args(args)


class SnakeGame:
    """Classe per gestire il gioco Snake con rendering."""
    
    # Colori per il rendering
    COLORS = {
        'background': (15, 15, 15),
        'grid': (30, 30, 30),
        'snake': (0, 255, 0),
        'head': (0, 200, 0),
        'apple': (255, 0, 0),
        'text': (255, 255, 255),
        'border': (50, 50, 50),
        'agent1': (0, 255, 0),  # Verde per agente 1
        'agent2': (0, 120, 255)  # Blu per agente 2
    }
    
    def __init__(self, grid_size=10, cell_size=40, speed=5, max_steps=1000,
                show_grid=True, show_stats=True):
        """
        Inizializza il gioco Snake.
        
        Args:
            grid_size: Dimensione della griglia
            cell_size: Dimensione delle celle in pixel
            speed: Velocità del gioco (fps)
            max_steps: Numero massimo di passi per episodio
            show_grid: Se mostrare la griglia
            show_stats: Se mostrare le statistiche
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.speed = speed
        self.max_steps = max_steps
        self.show_grid = show_grid
        self.show_stats = show_stats
        
        # Crea l'ambiente
        env_config = {
            "grid_size": grid_size,
            "max_steps": max_steps,
            "reward_apple": 10.0,
            "reward_death": -10.0
        }
        self.env = SnakeEnvironment(config=env_config)
        
        # Inizializza pygame
        pygame.init()
        
        # Calcola dimensioni della finestra
        self.window_width = grid_size * cell_size
        self.window_height = grid_size * cell_size
        
        if show_stats:
            # Aggiunge spazio per le statistiche
            self.window_height += 100
        
        # Crea la finestra
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Snake - DQN")
        
        # Crea il font
        self.font = pygame.font.SysFont('Arial', 16)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Crea il clock
        self.clock = pygame.time.Clock()
        
        # Variabili per le statistiche
        self.episode = 0
        self.steps = 0
        self.score = 0
        self.total_reward = 0.0
        self.episodes_stats = []
        
        # Variabili per la modalità compare
        self.compare_mode = False
        self.current_agent = 0  # 0 o 1
        self.compare_stats = [{'scores': [], 'steps': [], 'rewards': []},
                             {'scores': [], 'steps': [], 'rewards': []}]
    
    def reset(self):
        """Resetta l'ambiente e inizializza un nuovo episodio."""
        state, _ = self.env.reset()
        self.steps = 0
        self.score = 0
        self.total_reward = 0.0
        return state
    
    def render(self, action=None, agent_q_values=None, mode='ai'):
        """
        Disegna lo stato corrente del gioco.
        
        Args:
            action: Azione corrente
            agent_q_values: Q-values dell'agente
            mode: Modalità di gioco
        """
        # Ottieni lo stato di gioco
        snake_position = list(self.env.snake_game.snake)
        head_position = snake_position[0]
        apple_position = self.env.snake_game.food
        
        # Pulisci la finestra
        self.window.fill(self.COLORS['background'])
        
        # Disegna la griglia
        if self.show_grid:
            for x in range(0, self.window_width, self.cell_size):
                pygame.draw.line(self.window, self.COLORS['grid'], 
                               (x, 0), (x, self.grid_size * self.cell_size))
            for y in range(0, self.grid_size * self.cell_size, self.cell_size):
                pygame.draw.line(self.window, self.COLORS['grid'], 
                               (0, y), (self.window_width, y))
        
        # Disegna il serpente
        for i, segment in enumerate(snake_position):
            x, y = segment
            color = self.COLORS['head'] if i == 0 else self.COLORS['snake']
            
            # In modalità confronto, usa colori diversi
            if self.compare_mode:
                color = self.COLORS['agent1'] if self.current_agent == 0 else self.COLORS['agent2']
                if i == 0:  # Testa leggermente più scura
                    color = tuple(max(0, c - 50) for c in color)
            
            # Disegna il segmento del serpente
            pygame.draw.rect(self.window, color, 
                           (x * self.cell_size, y * self.cell_size, 
                            self.cell_size, self.cell_size))
            
            # Disegna un bordo più scuro
            pygame.draw.rect(self.window, self.COLORS['border'], 
                           (x * self.cell_size, y * self.cell_size, 
                            self.cell_size, self.cell_size), 1)
        
        # Disegna la mela
        x, y = apple_position
        pygame.draw.rect(self.window, self.COLORS['apple'], 
                       (x * self.cell_size, y * self.cell_size, 
                        self.cell_size, self.cell_size))
        
        # Disegna le statistiche
        if self.show_stats:
            stats_y = self.grid_size * self.cell_size + 10
            
            # Disegna titolo
            if mode == 'ai':
                title = "Modalità AI"
            elif mode == 'human':
                title = "Modalità Giocatore"
            elif mode == 'compare':
                agent_name = "Agente 1" if self.current_agent == 0 else "Agente 2"
                title = f"Confronto - {agent_name}"
            
            title_surface = self.title_font.render(title, True, self.COLORS['text'])
            self.window.blit(title_surface, (10, stats_y))
            
            # Disegna statistiche
            stats_y += 30
            stats = [
                f"Episodio: {self.episode}",
                f"Passi: {self.steps}",
                f"Punteggio: {self.score}",
                f"Reward: {self.total_reward:.2f}"
            ]
            
            for i, stat in enumerate(stats):
                stat_surface = self.font.render(stat, True, self.COLORS['text'])
                self.window.blit(stat_surface, (10, stats_y + i * 20))
            
            # Disegna le Q-values se disponibili
            if agent_q_values is not None:
                q_surface = self.font.render(
                    f"Q-values: {' | '.join([f'{q:.2f}' for q in agent_q_values])}",
                    True, self.COLORS['text']
                )
                self.window.blit(q_surface, (200, stats_y))
                
                # Disegna direzioni
                directions = ["SU", "GIÙ", "SINISTRA", "DESTRA"]
                dir_surface = self.font.render(
                    f"Direzioni: {' | '.join(directions)}",
                    True, self.COLORS['text']
                )
                self.window.blit(dir_surface, (200, stats_y + 20))
                
                # Evidenzia l'azione selezionata
                if action is not None:
                    action_text = f"Azione: {directions[action]}"
                    action_surface = self.font.render(action_text, True, self.COLORS['text'])
                    self.window.blit(action_surface, (200, stats_y + 40))
        
        # Aggiorna il display
        pygame.display.flip()
    
    def human_play(self, episodes=10):
        """Modalità di gioco per giocatore umano."""
        print("Inizializzazione modalità giocatore umano...")
        print("Usa le frecce per muoverti. Premi 'Q' per uscire.")
        
        self.episode = 0
        episodes_completed = 0
        action_map = {
            pygame.K_UP: 0,     # Su
            pygame.K_DOWN: 1,   # Giù
            pygame.K_LEFT: 2,   # Sinistra
            pygame.K_RIGHT: 3   # Destra
        }
        
        # Loop degli episodi
        while episodes_completed < episodes:
            # Aggiorna il contatore dell'episodio
            self.episode = episodes_completed + 1
            
            # Resetta l'ambiente
            state = self.reset()
            done = False
            
            # Renderizza lo stato iniziale
            self.render(mode='human')
            
            # Azione iniziale (destra)
            action = 3
            
            # Loop del gioco
            while not done:
                # Gestisci gli eventi
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            return
                        elif event.key in action_map:
                            action = action_map[event.key]
                
                # Esegui l'azione
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Aggiorna lo stato e le statistiche
                state = next_state
                self.steps += 1
                self.total_reward += reward
                
                if info.get('apple_eaten', False):
                    self.score += 1
                
                # Renderizza
                self.render(action=action, mode='human')
                
                # Limita la velocità
                self.clock.tick(self.speed)
            
            # Episodio completato
            episodes_completed += 1
            
            # Registra le statistiche
            self.episodes_stats.append({
                'score': self.score,
                'steps': self.steps,
                'reward': self.total_reward
            })
            
            print(f"Episodio {self.episode} completato. "
                 f"Punteggio: {self.score}, Passi: {self.steps}")
            
            # Piccola pausa tra gli episodi
            time.sleep(1)
        
        # Stampa statistiche finali
        self.print_final_stats()
    
    def ai_play(self, agent, episodes=10):
        """
        Modalità di gioco con agente AI.
        
        Args:
            agent: L'agente da utilizzare
            episodes: Numero di episodi da giocare
        """
        print("Inizializzazione modalità AI...")
        
        self.episode = 0
        episodes_completed = 0
        
        # Loop degli episodi
        while episodes_completed < episodes:
            # Aggiorna il contatore dell'episodio
            self.episode = episodes_completed + 1
            
            # Resetta l'ambiente
            state = self.reset()
            done = False
            
            # Loop del gioco
            while not done:
                # Gestisci gli eventi
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            return
                
                # Seleziona un'azione
                q_values = agent.get_q_values(state)
                action = agent.select_action(state, deterministic=True)
                
                # Esegui l'azione
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Aggiorna lo stato e le statistiche
                state = next_state
                self.steps += 1
                self.total_reward += reward
                
                if info.get('apple_eaten', False):
                    self.score += 1
                
                # Renderizza
                self.render(action=action, agent_q_values=q_values.cpu().numpy(), mode='ai')
                
                # Limita la velocità
                self.clock.tick(self.speed)
            
            # Episodio completato
            episodes_completed += 1
            
            # Registra le statistiche
            self.episodes_stats.append({
                'score': self.score,
                'steps': self.steps,
                'reward': self.total_reward
            })
            
            print(f"Episodio {self.episode} completato. "
                 f"Punteggio: {self.score}, Passi: {self.steps}")
            
            # Piccola pausa tra gli episodi
            time.sleep(1)
        
        # Stampa statistiche finali
        self.print_final_stats()
    
    def compare_play(self, agent1, agent2, episodes=10):
        """
        Modalità di confronto tra due agenti.
        
        Args:
            agent1: Primo agente
            agent2: Secondo agente
            episodes: Numero di episodi da giocare
        """
        print("Inizializzazione modalità confronto...")
        
        self.compare_mode = True
        self.episode = 0
        episodes_completed = 0
        
        # Loop degli episodi
        while episodes_completed < episodes:
            # Aggiorna il contatore dell'episodio
            self.episode = episodes_completed + 1
            
            # Alterna tra i due agenti
            self.current_agent = episodes_completed % 2
            current_agent = agent1 if self.current_agent == 0 else agent2
            
            # Resetta l'ambiente
            state = self.reset()
            done = False
            
            # Loop del gioco
            while not done:
                # Gestisci gli eventi
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            pygame.quit()
                            return
                
                # Seleziona un'azione
                q_values = current_agent.get_q_values(state)
                action = current_agent.select_action(state, deterministic=True)
                
                # Esegui l'azione
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Aggiorna lo stato e le statistiche
                state = next_state
                self.steps += 1
                self.total_reward += reward
                
                if info.get('apple_eaten', False):
                    self.score += 1
                
                # Renderizza
                self.render(action=action, agent_q_values=q_values.cpu().numpy(), mode='compare')
                
                # Limita la velocità
                self.clock.tick(self.speed)
            
            # Episodio completato
            episodes_completed += 1
            
            # Registra le statistiche per l'agente corrente
            agent_stats = self.compare_stats[self.current_agent]
            agent_stats['scores'].append(self.score)
            agent_stats['steps'].append(self.steps)
            agent_stats['rewards'].append(self.total_reward)
            
            print(f"Episodio {self.episode} (Agente {self.current_agent+1}) completato. "
                 f"Punteggio: {self.score}, Passi: {self.steps}")
            
            # Piccola pausa tra gli episodi
            time.sleep(1)
        
        # Stampa statistiche di confronto
        self.print_compare_stats()
    
    def print_final_stats(self):
        """Stampa le statistiche finali."""
        scores = [s['score'] for s in self.episodes_stats]
        steps = [s['steps'] for s in self.episodes_stats]
        rewards = [s['reward'] for s in self.episodes_stats]
        
        print("\nStatistiche finali:")
        print(f"Episodi giocati: {len(self.episodes_stats)}")
        print(f"Punteggio medio: {np.mean(scores):.2f} (±{np.std(scores):.2f})")
        print(f"Punteggio massimo: {np.max(scores)}")
        print(f"Passi medi: {np.mean(steps):.2f}")
        print(f"Reward medio: {np.mean(rewards):.2f}")
    
    def print_compare_stats(self):
        """Stampa le statistiche di confronto."""
        agent1_stats = self.compare_stats[0]
        agent2_stats = self.compare_stats[1]
        
        print("\nStatistiche di confronto:")
        print(f"Episodi giocati: {len(agent1_stats['scores']) + len(agent2_stats['scores'])}")
        
        print("\nAgente 1:")
        print(f"Punteggio medio: {np.mean(agent1_stats['scores']):.2f} (±{np.std(agent1_stats['scores']):.2f})")
        print(f"Punteggio massimo: {np.max(agent1_stats['scores']) if agent1_stats['scores'] else 0}")
        print(f"Passi medi: {np.mean(agent1_stats['steps']):.2f}")
        print(f"Reward medio: {np.mean(agent1_stats['rewards']):.2f}")
        
        print("\nAgente 2:")
        print(f"Punteggio medio: {np.mean(agent2_stats['scores']):.2f} (±{np.std(agent2_stats['scores']):.2f})")
        print(f"Punteggio massimo: {np.max(agent2_stats['scores']) if agent2_stats['scores'] else 0}")
        print(f"Passi medi: {np.mean(agent2_stats['steps']):.2f}")
        print(f"Reward medio: {np.mean(agent2_stats['rewards']):.2f}")
        
        # Confronto diretto
        if len(agent1_stats['scores']) > 0 and len(agent2_stats['scores']) > 0:
            if np.mean(agent1_stats['scores']) > np.mean(agent2_stats['scores']):
                print("\nL'Agente 1 ha un punteggio medio più alto!")
            elif np.mean(agent1_stats['scores']) < np.mean(agent2_stats['scores']):
                print("\nL'Agente 2 ha un punteggio medio più alto!")
            else:
                print("\nI due agenti hanno lo stesso punteggio medio!")
        
        # Plot di confronto
        self.plot_compare_stats()
    
    def plot_compare_stats(self):
        """Genera un grafico di confronto tra i due agenti."""
        agent1_stats = self.compare_stats[0]
        agent2_stats = self.compare_stats[1]
        
        plt.figure(figsize=(12, 8))
        
        # Plot dei punteggi
        plt.subplot(2, 2, 1)
        plt.bar(['Agente 1', 'Agente 2'], 
               [np.mean(agent1_stats['scores']), np.mean(agent2_stats['scores'])],
               yerr=[np.std(agent1_stats['scores']), np.std(agent2_stats['scores'])],
               capsize=10, color=['green', 'blue'])
        plt.title('Punteggio medio')
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        # Plot degli step
        plt.subplot(2, 2, 2)
        plt.bar(['Agente 1', 'Agente 2'], 
               [np.mean(agent1_stats['steps']), np.mean(agent2_stats['steps'])],
               yerr=[np.std(agent1_stats['steps']), np.std(agent2_stats['steps'])],
               capsize=10, color=['green', 'blue'])
        plt.title('Passi medi')
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        # Plot delle reward
        plt.subplot(2, 2, 3)
        plt.bar(['Agente 1', 'Agente 2'], 
               [np.mean(agent1_stats['rewards']), np.mean(agent2_stats['rewards'])],
               yerr=[np.std(agent1_stats['rewards']), np.std(agent2_stats['rewards'])],
               capsize=10, color=['green', 'blue'])
        plt.title('Reward medio')
        plt.grid(True, alpha=0.3)
        
        # Plot dei punteggi massimi
        plt.subplot(2, 2, 4)
        plt.bar(['Agente 1', 'Agente 2'], 
               [np.max(agent1_stats['scores']) if agent1_stats['scores'] else 0,
                np.max(agent2_stats['scores']) if agent2_stats['scores'] else 0],
               color=['green', 'blue'])
        plt.title('Punteggio massimo')
        plt.ylim(bottom=0)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('compare_stats.png')
        plt.close()
        
        print("Grafico di confronto salvato in 'compare_stats.png'")


def load_agent(agent_type, model_path, state_dim, action_dim, device):
    """
    Carica un agente da un file salvato.
    
    Args:
        agent_type: Tipo di agente
        model_path: Percorso del modello
        state_dim: Dimensione dello stato
        action_dim: Dimensione dell'azione
        device: Dispositivo
        
    Returns:
        L'agente caricato
    """
    # Crea l'agente
    agent = get_agent(agent_type, 
                     state_dim=state_dim, 
                     action_dim=action_dim,
                     config={})
    
    # Carica il modello
    agent.load(model_path)
    
    return agent


def main(args):
    """Funzione principale."""
    # Se args è una lista di stringhe, analizzale come argomenti
    if isinstance(args, list):
        args = parse_args(args)
    
    # Configura il logging in modalità debug
    if args.debug:
        import logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logging.debug("Modalità debug attivata")
        print("Modalità debug attivata: i log dettagliati saranno mostrati in console")
        
    # Imposta il dispositivo
    device = get_device()
    print(f"Dispositivo utilizzato: {device}")
    
    # Crea il gioco
    game = SnakeGame(
        grid_size=args.grid_size,
        cell_size=args.cell_size,
        speed=args.speed,
        max_steps=args.max_steps,
        show_grid=args.show_grid,
        show_stats=args.stats
    )
    
    # Ottieni le dimensioni dell'ambiente
    state_dim = game.env.observation_space.shape[0]
    action_dim = game.env.action_space.n
    
    # Modalità di gioco
    if args.mode == 'human':
        # Modalità giocatore umano
        try:
            game.human_play(args.episodes)
        except Exception as e:
            if args.debug:
                import logging
                import traceback
                logging.error(f"Errore durante la modalità giocatore umano: {e}")
                logging.error(traceback.format_exc())
            raise
    
    elif args.mode == 'ai':
        # Verifica che il percorso del modello sia specificato
        if args.model_path is None:
            print("Errore: devi specificare il percorso del modello con --model_path")
            return
        
        # Carica l'agente
        try:
            agent = load_agent(args.agent, args.model_path, state_dim, action_dim, device)
            
            # Modalità AI
            game.ai_play(agent, args.episodes)
        except Exception as e:
            if args.debug:
                import logging
                import traceback
                logging.error(f"Errore durante la modalità AI: {e}")
                logging.error(traceback.format_exc())
            raise
    
    elif args.mode == 'compare':
        # Verifica che i percorsi dei modelli siano specificati
        if args.model_path is None or args.model_path2 is None:
            print("Errore: devi specificare i percorsi di entrambi i modelli")
            print("Usa --model_path per il primo e --model_path2 per il secondo")
            return
        
        try:
            # Carica gli agenti
            agent1 = load_agent(args.agent, args.model_path, state_dim, action_dim, device)
            agent2 = load_agent(args.agent, args.model_path2, state_dim, action_dim, device)
            
            # Modalità confronto
            game.compare_play(agent1, agent2, args.episodes)
        except Exception as e:
            if args.debug:
                import logging
                import traceback
                logging.error(f"Errore durante la modalità confronto: {e}")
                logging.error(traceback.format_exc())
            raise


if __name__ == "__main__":
    # Analizza gli argomenti da linea di comando
    args = parse_args()
    
    try:
        # Avvia il gioco
        main(args)
    except Exception as e:
        if hasattr(args, 'debug') and args.debug:
            import logging
            import traceback
            logging.error(f"Errore non gestito: {e}")
            logging.error(traceback.format_exc())
        raise 