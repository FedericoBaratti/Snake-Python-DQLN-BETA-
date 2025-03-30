#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake con UI e Deep Q-Learning (DQN)
====================================
Entry point principale del progetto che avvia l'interfaccia utente
e permette di scegliere tra modalità di gioco manuale e autoplay.

Autore: Baratti Federico
Versione: 1.0
"""

import argparse
import os
import sys
from pathlib import Path

def setup_path():
    """Aggiunge il percorso della root al sys.path per permettere le importazioni relative."""
    root_dir = Path(__file__).resolve().parent
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

if __name__ == "__main__":
    setup_path()
    
    from frontend.ui import GameUI
    from backend.snake_game import SnakeGame
    from backend.environment import SnakeEnv
    from dqn_agent.config import get_config
    from autoplay.autoplay import AutoplayController
    
    parser = argparse.ArgumentParser(description='Snake Game con DQN Autoplay')
    parser.add_argument('--mode', type=str, default='manual', 
                        choices=['manual', 'autoplay', 'train', 'train-and-play'],
                        help='Modalità di gioco: manuale, autoplay, training, o training seguito da autoplay')
    parser.add_argument('--model', type=str, default='base',
                        choices=['base', 'avanzato', 'complesso', 'perfetto'],
                        help='Complessità del modello DQN per autoplay/training')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Percorso al checkpoint del modello (opzionale)')
    parser.add_argument('--grid-size', type=int, default=20,
                        help='Dimensione della griglia di gioco')
    parser.add_argument('--speed', type=int, default=10,
                        help='Velocità iniziale del gioco (1-20)')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Numero di episodi di training (usa default se non specificato)')
    parser.add_argument('--demo', action='store_true',
                        help='Modalità demo: addestra un modello base per pochi episodi (300) su una griglia piccola (10x10)')
    
    args = parser.parse_args()
    
    # Ottieni la configurazione in base al livello di complessità
    config = get_config(complexity=args.model)
    
    # Imposta la modalità demo (addestramento rapido)
    if args.demo:
        args.mode = 'train-and-play'
        args.model = 'base'
        args.grid_size = 10
        args.episodes = 300
        print("Modalità demo attiva: addestramento rapido di un modello base su griglia 10x10 per 300 episodi")
    
    # Inizializza il modello basato su checkpoint o addestramento
    trained_agent = None
    checkpoint_path = args.checkpoint
    
    if args.mode == 'train' or args.mode == 'train-and-play':
        # Avvia il training
        from training.train import train_agent
        
        # Crea le directory necessarie se non esistono
        checkpoint_dir = Path("training/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Addestra l'agente
        trained_agent = train_agent(
            model_complexity=args.model, 
            checkpoint_path=args.checkpoint,
            config=config,
            num_episodes=args.episodes,
            grid_size=args.grid_size
        )
        
        # Salva il modello finale
        checkpoint_path = f"training/checkpoints/dqn_{args.model}_latest.pt"
        trained_agent.save(path=checkpoint_path)
        print(f"Modello salvato in: {checkpoint_path}")
        
        # Se si tratta solo di addestramento, esci
        if args.mode == 'train':
            sys.exit(0)
    
    # Inizializza il gioco e l'ambiente
    game = SnakeGame(grid_size=args.grid_size)
    env = SnakeEnv(game)
    
    if args.mode == 'autoplay' or args.mode == 'train-and-play':
        # Modalità autoplay
        controller = AutoplayController(
            env=env, 
            model_complexity=args.model,
            checkpoint_path=checkpoint_path,
            config=config
        )
        # Avvia l'interfaccia con il controller autoplay
        ui = GameUI(game=game, speed=args.speed, autoplay_controller=controller)
        ui.run()
    else:
        # Modalità manuale
        ui = GameUI(game=game, speed=args.speed)
        ui.run() 