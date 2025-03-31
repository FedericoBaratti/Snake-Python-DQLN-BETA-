#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo di addestramento
=======================
Script per addestrare l'agente DQN sul gioco Snake.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
import logging

from core.environment import SnakeEnv as SnakeEnvironment
from agents import get_agent
from utils.config import ConfigManager as Config
from utils.hardware_utils import get_device, set_hardware_optimization, enable_mixed_precision


def parse_args():
    """Analizza gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Addestramento agente DQN per Snake')
    
    # Parametri generali
    parser.add_argument('--config', type=str, default='config/training_config.json',
                      help='Percorso del file di configurazione')
    parser.add_argument('--resume', action='store_true',
                      help='Riprendi addestramento da checkpoint')
    parser.add_argument('--render', action='store_true',
                      help='Visualizza il gioco durante l\'addestramento')
    parser.add_argument('--render_delay', type=float, default=0.01,
                      help='Ritardo di rendering in secondi (default: 0.01)')
    parser.add_argument('--debug', action='store_true',
                      help='Abilita modalità debug con output aggiuntivi')
    parser.add_argument('--exp_name', type=str, default=None,
                      help='Nome dell\'esperimento (cartella dei risultati)')
    parser.add_argument('--log_file', type=str, default=None,
                      help='Percorso del file di log per il debugging')
    parser.add_argument('--log_level', type=str, default='info',
                      choices=['debug', 'info', 'warning', 'error', 'critical'],
                      help='Livello di logging (debug, info, warning, error, critical)')
    
    # Parametri hardware
    parser.add_argument('--device', type=str, default=None,
                      help='Dispositivo (cuda:0, cpu)')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Numero di worker per caricamento dati')
    
    # Parametri di addestramento
    parser.add_argument('--mixed_precision', type=str, default=None,
                      help='Abilita addestramento con precisione mista (true/false)')
    parser.add_argument('--agent', type=str, default=None,
                      help='Tipo di agente (dqn, ...)')
    parser.add_argument('--model', type=str, default=None,
                      help='Tipo di modello (dqn, dueling, noisy, noisy_dueling)')
    parser.add_argument('--model_type', type=str, default=None,
                      help='Alias per --model (dqn, dueling, noisy, noisy_dueling)')
    parser.add_argument('--buffer', type=str, default=None,
                      help='Tipo di buffer (standard, prioritized)')
    parser.add_argument('--double', action='store_true',
                      help='Usa Double DQN')
    parser.add_argument('--eps_decay', type=float, default=None,
                      help='Decadimento epsilon (per esplorazione)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Dimensione batch')
    parser.add_argument('--hidden_dim', type=str, default=None,
                      help='Dimensioni layer nascosti, formato: "64,64"')
    parser.add_argument('--buffer_size', type=int, default=None,
                      help='Dimensione buffer di memoria')
    parser.add_argument('--lr', type=float, default=None,
                      help='Learning rate')
    parser.add_argument('--gamma', type=float, default=None,
                      help='Fattore di sconto')
    parser.add_argument('--n_steps', type=int, default=None,
                      help='N-step return (1 = one-step)')
    parser.add_argument('--target_update', type=int, default=None,
                      help='Frequenza aggiornamento rete target')
    parser.add_argument('--total_episodes', type=int, default=None,
                      help='Numero totale di episodi di addestramento')
    parser.add_argument('--eval_freq', type=int, default=None,
                      help='Frequenza di valutazione (episodi)')
    parser.add_argument('--grid_size', type=int, default=None,
                      help='Dimensione della griglia di gioco')
    
    # Parametri di ricompensa personalizzati
    parser.add_argument('--env_reward_apple', type=float, default=None,
                      help='Ricompensa per aver mangiato una mela')
    parser.add_argument('--env_reward_death', type=float, default=None,
                      help='Penalità per la morte')
    parser.add_argument('--env_reward_step', type=float, default=None,
                      help='Ricompensa/penalità per ogni passo')
    parser.add_argument('--env_reward_closer_to_food', type=float, default=None,
                      help='Ricompensa per avvicinarsi al cibo')
    parser.add_argument('--env_reward_farther_from_food', type=float, default=None,
                      help='Penalità per allontanarsi dal cibo')
    parser.add_argument('--food_reward', type=float, default=None,
                      help='Alias per --env_reward_apple')
    parser.add_argument('--death_penalty', type=float, default=None,
                      help='Alias per --env_reward_death')
    parser.add_argument('--move_closer_reward', type=float, default=None,
                      help='Alias per --env_reward_closer_to_food')
    parser.add_argument('--move_away_penalty', type=float, default=None,
                      help='Alias per --env_reward_farther_from_food')
    
    args = parser.parse_args()
    return args


def load_config(args):
    """Carica la configurazione combinando file e argomenti CLI."""
    # Inizializza con configurazione di default o da file
    if os.path.exists(args.config):
        config = Config(args.config)
    else:
        config = Config()
        print(f"File di configurazione {args.config} non trovato, uso valori predefiniti.")
    
    # Gestisci model_type come alias per model
    if args.model_type is not None and args.model is None:
        args.model = args.model_type
    
    # Aggiorna con argomenti CLI, solo se specificati
    arg_dict = vars(args)
    for key, value in arg_dict.items():
        if value is not None:
            # Gestione speciale per liste da stringhe
            if key == 'hidden_dim' and isinstance(value, str):
                value = [int(x) for x in value.split(',')]
            
            # Gestione speciale per grid_size
            if key == 'grid_size':
                key = 'env_grid_size'
            
            # Aggiorna la configurazione
            if hasattr(config, key):
                setattr(config, key, value)
            elif '.' in key:  # Gestione di parametri nidificati
                main_key, sub_key = key.split('.', 1)
                if hasattr(config, main_key):
                    main_obj = getattr(config, main_key)
                    if isinstance(main_obj, dict) and sub_key in main_obj:
                        main_obj[sub_key] = value
    
    # Crea il nome dell'esperimento se non specificato
    if not hasattr(config, 'exp_name') or not config.exp_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_type = getattr(config, 'agent', 'dqn')
        model_type = getattr(config, 'model', 'default')
        config.exp_name = f"{agent_type}_{model_type}_{timestamp}"
    
    # Assicurati che gli attributi essenziali esistano
    if not hasattr(config, 'agent'):
        config.agent = 'dqn'
    if not hasattr(config, 'model'):
        config.model = 'default'
    if not hasattr(config, 'buffer'):
        config.buffer = 'standard'
    if not hasattr(config, 'device'):
        config.device = 'auto'
    
    # Crea la directory di output
    if not hasattr(config, 'output_dir'):
        config.output_dir = os.path.join("results", config.exp_name)
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Aggiungi attributi essenziali se mancanti
    if not hasattr(config, 'hidden_dim'):
        config.hidden_dim = [64, 64]
    if not hasattr(config, 'lr'):
        config.lr = 0.001
    if not hasattr(config, 'gamma'):
        config.gamma = 0.99
    if not hasattr(config, 'buffer_size'):
        config.buffer_size = 10000
    if not hasattr(config, 'batch_size'):
        config.batch_size = 32
    if not hasattr(config, 'eps_start'):
        config.eps_start = 1.0
    if not hasattr(config, 'eps_end'):
        config.eps_end = 0.01
    if not hasattr(config, 'eps_decay'):
        config.eps_decay = 0.995
    if not hasattr(config, 'target_update'):
        config.target_update = 10
    if not hasattr(config, 'double'):
        config.double = False
    if not hasattr(config, 'n_steps'):
        config.n_steps = 1
    if not hasattr(config, 'total_episodes'):
        config.total_episodes = 10
    if not hasattr(config, 'eval_freq'):
        config.eval_freq = 5
    if not hasattr(config, 'eval_episodes'):
        config.eval_episodes = 3
    if not hasattr(config, 'checkpoint_freq'):
        config.checkpoint_freq = 5
    if not hasattr(config, 'render'):
        config.render = False
    if not hasattr(config, 'render_delay'):
        config.render_delay = 0.01
    if not hasattr(config, 'resume'):
        config.resume = False
    if not hasattr(config, 'mixed_precision'):
        config.mixed_precision = False
    
    # Gestione dei parametri di logging
    if args.log_file is not None:
        config.log_file = args.log_file
    if args.log_level is not None:
        config.log_level = args.log_level
    if args.debug and not hasattr(config, 'log_level'):
        config.log_level = 'debug'
    
    # Gestione degli alias dei parametri di ricompensa
    if args.food_reward is not None:
        config.env_reward_apple = args.food_reward
    if args.death_penalty is not None:
        config.env_reward_death = args.death_penalty
    if args.move_closer_reward is not None:
        config.env_reward_closer_to_food = args.move_closer_reward
    if args.move_away_penalty is not None:
        config.env_reward_farther_from_food = args.move_away_penalty
    
    # Salva la configurazione
    config.save_config(os.path.join(config.output_dir, "config.json"))
    
    return config


def evaluate_agent(agent, env, num_episodes=5, render=False, render_delay=0.05):
    """
    Valuta l'agente in modalità deterministica.
    
    Args:
        agent: L'agente da valutare
        env: L'ambiente di valutazione
        num_episodes: Numero di episodi di valutazione
        render: Se visualizzare l'ambiente
        render_delay: Ritardo di rendering in secondi
        
    Returns:
        dict: Statistiche di valutazione
    """
    rewards = []
    lengths = []
    apples = []
    
    for i in range(num_episodes):
        done = False
        total_reward = 0
        state, _ = env.reset()
        step = 0
        apple_count = 0
        
        while not done:
            # Azione deterministica (greedy)
            action = agent.select_action(state, training=False)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Aggiorna contatori
            total_reward += reward
            step += 1
            
            # Se abbiamo preso una mela
            if info.get('apple_eaten', False):
                apple_count += 1
            
            # Aggiorna lo stato
            state = next_state
            
            # Rendering
            if render:
                env.render()
                time.sleep(render_delay)
        
        # Salva statistiche
        rewards.append(total_reward)
        lengths.append(step)
        apples.append(apple_count)
    
    # Calcola statistiche
    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'mean_apples': np.mean(apples),
        'max_apples': np.max(apples),
    }
    
    return stats


def plot_training_curves(stats, output_path):
    """
    Genera grafici delle curve di addestramento.
    
    Args:
        stats (dict): Statistiche di addestramento
        output_path (str): Directory di output
    """
    episodes = stats['episodes']
    
    # Crea la figura con più subplot
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward
    axs[0, 0].plot(episodes, stats['rewards'], label='Reward per episodio')
    if len(stats['eval_rewards']) > 0:
        # Crea un array di episodi di valutazione (1 valutazione ogni eval_freq episodi)
        eval_episodes = episodes[::len(episodes) // len(stats['eval_rewards']) or 1][:len(stats['eval_rewards'])]
        axs[0, 0].plot(eval_episodes, stats['eval_rewards'], 'r-', label='Reward valutazione')
    axs[0, 0].set_xlabel('Episodio')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].set_title('Reward durante addestramento')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Lunghezza episodio
    axs[0, 1].plot(episodes, stats['lengths'], label='Lunghezza episodio')
    if len(stats['eval_lengths']) > 0:
        # Usa gli stessi episodi di valutazione
        eval_episodes = episodes[::len(episodes) // len(stats['eval_lengths']) or 1][:len(stats['eval_lengths'])]
        axs[0, 1].plot(eval_episodes, stats['eval_lengths'], 'r-', label='Lunghezza valutazione')
    axs[0, 1].set_xlabel('Episodio')
    axs[0, 1].set_ylabel('Passi')
    axs[0, 1].set_title('Lunghezza episodi')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Perdita
    axs[1, 0].plot(episodes, stats['losses'], label='Perdita')
    axs[1, 0].set_xlabel('Episodio')
    axs[1, 0].set_ylabel('Perdita')
    axs[1, 0].set_title('Perdita durante addestramento')
    axs[1, 0].grid(True)
    
    # Epsilon
    axs[1, 1].plot(episodes, stats['epsilons'], label='Epsilon')
    axs[1, 1].set_xlabel('Episodio')
    axs[1, 1].set_ylabel('Epsilon')
    axs[1, 1].set_title('Decadimento epsilon')
    axs[1, 1].grid(True)
    
    # Titolo generale
    plt.suptitle('Curve di addestramento', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Salva il grafico
    plt.savefig(os.path.join(output_path, 'training_curves.png'))
    plt.close()
    
    # Grafico mele raccolte
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, stats['apples'], label='Mele per episodio')
    if len(stats['eval_apples']) > 0:
        # Usa gli stessi episodi di valutazione
        eval_episodes = episodes[::len(episodes) // len(stats['eval_apples']) or 1][:len(stats['eval_apples'])]
        plt.plot(eval_episodes, stats['eval_apples'], 'r-', label='Mele valutazione')
    plt.xlabel('Episodio')
    plt.ylabel('Mele raccolte')
    plt.title('Mele raccolte durante addestramento')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'apples.png'))
    plt.close()


def train(config):
    """
    Addestra l'agente DQN.
    
    Args:
        config: Configurazione di addestramento
    """
    # Configura il logging
    log_level = getattr(config, 'log_level', 'info').upper()
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configura il logger
    numeric_level = getattr(logging, log_level, logging.INFO)
    
    if hasattr(config, 'log_file') and config.log_file:
        # Assicurati che la directory di log esista
        log_dir = os.path.dirname(config.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Configura il logging su file
        logging.basicConfig(
            level=numeric_level,
            format=log_format,
            filename=config.log_file,
            filemode='w'  # 'w' sovrascrive, 'a' aggiunge
        )
        
        # Aggiungi handler per la console
        console = logging.StreamHandler()
        console.setLevel(numeric_level)
        console.setFormatter(logging.Formatter(log_format))
        logging.getLogger('').addHandler(console)
        
        logging.info(f"Logging configurato al livello {log_level} nel file {config.log_file}")
    elif getattr(config, 'debug', False):
        # Configura il logging in console in modalità debug
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format
        )
        logging.debug("Modalità debug attivata: i log dettagliati saranno mostrati in console")
    else:
        # Configura il logging in console con il livello specificato
        logging.basicConfig(
            level=numeric_level,
            format=log_format
        )
    
    # Imposta il dispositivo
    device = get_device() if config.device == 'auto' else torch.device(config.device)
    logging.info(f"Utilizzo dispositivo: {device}")
    print(f"Utilizzo dispositivo: {device}")
    
    # Set ottimizzazioni hardware
    set_hardware_optimization()
    
    # Configura precisione mista se richiesta
    use_mixed_precision = False
    if hasattr(config, 'mixed_precision') and (
            config.mixed_precision is True or 
            (isinstance(config.mixed_precision, str) and config.mixed_precision.lower() == 'true')):
        use_mixed_precision = enable_mixed_precision(True)
        if use_mixed_precision:
            logging.info("Precisione mista abilitata")
            print("Precisione mista abilitata")
            scaler = torch.cuda.amp.GradScaler()
        else:
            logging.info("Precisione mista non supportata o non abilitata")
            print("Precisione mista non supportata o non abilitata")
    
    # Crea l'ambiente
    env_config = {
        "grid_size": getattr(config, 'env_grid_size', 10),
        "max_steps": getattr(config, 'env_max_steps', 1000),
        "reward_food": getattr(config, 'env_reward_apple', 10.0),
        "reward_death": getattr(config, 'env_reward_death', -10.0),
        "reward_step": getattr(config, 'env_reward_step', -0.01),
        "reward_closer_to_food": getattr(config, 'env_reward_closer_to_food', 0.1),
        "reward_farther_from_food": getattr(config, 'env_reward_farther_from_food', -0.1),
        "reward_distance": getattr(config, 'env_reward_distance', True)
    }
    
    # Stampa informazioni sulla configurazione delle ricompense
    logging.info(f"Configurazione ricompense:")
    logging.info(f"  - Mela: {env_config['reward_food']}")
    logging.info(f"  - Morte: {env_config['reward_death']}")
    logging.info(f"  - Passo: {env_config['reward_step']}")
    logging.info(f"  - Avvicinamento: {env_config['reward_closer_to_food']}")
    logging.info(f"  - Allontanamento: {env_config['reward_farther_from_food']}")
    
    print(f"Configurazione ricompense:")
    print(f"  - Mela: {env_config['reward_food']}")
    print(f"  - Morte: {env_config['reward_death']}")
    print(f"  - Passo: {env_config['reward_step']}")
    print(f"  - Avvicinamento: {env_config['reward_closer_to_food']}")
    print(f"  - Allontanamento: {env_config['reward_farther_from_food']}")
    
    env = SnakeEnvironment(config=env_config)
    
    # Ottieni le dimensioni dell'ambiente
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    logging.info(f"Dimensione stato: {state_dim}, Dimensione azione: {action_dim}")
    print(f"Dimensione stato: {state_dim}, Dimensione azione: {action_dim}")
    
    # Inizializza le statistiche
    stats = {
        'episodes': [],
        'rewards': [],
        'lengths': [],
        'losses': [],
        'epsilons': [],
        'apples': [],
        'eval_rewards': [],
        'eval_lengths': [],
        'eval_apples': [],
        'best_eval_reward': float('-inf'),
        'best_eval_episode': 0
    }
    
    # Prepara i parametri per l'agente
    agent_config = {
        'hidden_layers': getattr(config, 'hidden_dim', [64, 64]),
        'lr': getattr(config, 'lr', 0.001),
        'gamma': getattr(config, 'gamma', 0.99),
        'memory_size': getattr(config, 'buffer_size', 10000),
        'batch_size': getattr(config, 'batch_size', 32),
        'eps_start': getattr(config, 'eps_start', 1.0),
        'eps_end': getattr(config, 'eps_end', 0.01),
        'eps_decay': getattr(config, 'eps_decay', 0.995),
        'target_update': getattr(config, 'target_update', 10),
        'use_double_dqn': getattr(config, 'double', False),
        'n_step_returns': getattr(config, 'n_steps', 1),
    }
    
    # Aggiungi il parametro use_noisy se il modello è di tipo noisy
    if config.model.lower() in ['noisy', 'noisy_dueling']:
        agent_config['use_noisy'] = True
    
    # Crea l'agente
    agent = get_agent(config.agent, state_dim=state_dim, action_dim=action_dim, config=agent_config)
    
    # Checkpoint e ripresa dell'addestramento
    checkpoint_path = os.path.join(config.output_dir, 'checkpoint.pt')
    stats_path = os.path.join(config.output_dir, 'stats.json')
    best_model_path = os.path.join(config.output_dir, 'best_model.pt')
    
    start_episode = 0
    
    if config.resume and os.path.exists(checkpoint_path) and os.path.exists(stats_path):
        logging.info("Ripresa addestramento da checkpoint...")
        # Carica l'agente
        agent.load(checkpoint_path)
        
        # Carica le statistiche
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Trova l'ultimo episodio
        start_episode = stats['episodes'][-1] + 1
        
        logging.info(f"Addestramento ripreso dall'episodio {start_episode}")
    
    # Ciclo principale di addestramento
    try:
        total_steps = 0
        start_time = time.time()
        
        for episode in range(start_episode, config.total_episodes):
            # Reset dell'ambiente
            state, _ = env.reset()
            done = False
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0
            apple_count = 0
            
            # Esegui un episodio
            while not done:
                # Seleziona un'azione
                action = agent.select_action(state)
                
                # Esegui l'azione
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Salva la transizione
                agent.store_transition(state, action, next_state, reward, done)
                
                # Aggiorna contatori
                episode_reward += reward
                episode_steps += 1
                total_steps += 1
                
                # Se abbiamo preso una mela
                if info.get('apple_eaten', False):
                    apple_count += 1
                
                # Aggiorna lo stato
                state = next_state
                
                # Addestra l'agente
                if len(agent.memory) >= agent.batch_size:
                    # Estrai un batch dalla replay memory
                    batch = agent.memory.sample(config.batch_size)
                    state_batch, action_batch, reward_batch, next_state_batch, done_batch = batch
                    
                    # Converte tutto in tensori numpy/torch se necessario
                    state_batch = torch.FloatTensor(state_batch).to(device)
                    action_batch = torch.LongTensor(action_batch).to(device)
                    reward_batch = torch.FloatTensor(reward_batch).to(device)
                    next_state_batch = torch.FloatTensor(next_state_batch).to(device)
                    done_batch = torch.FloatTensor(done_batch).to(device)
                    
                    # Debug: stampa le forme dei tensori
                    if episode <= 5 or episode % 100 == 0:
                        logging.debug(f"Debug - Forme batch: stato {state_batch.shape}, azioni {action_batch.shape}, next_state {next_state_batch.shape}")
                    
                    # Caso speciale: se next_state_batch ha forma [batch_size, 1] ma state_batch ha forma [batch_size, state_dim]
                    if next_state_batch.shape[1] == 1 and state_batch.shape[1] == state_dim:
                        logging.debug(f"Rilevato next_state_batch con forma {next_state_batch.shape} mentre state_batch ha forma {state_batch.shape}")
                        logging.debug(f"Clonando state_batch e sostituendo il primo valore con next_state_batch")
                        expanded_next_state = state_batch.clone()
                        # Sostituisci solo il primo valore
                        expanded_next_state[:, 0:1] = next_state_batch
                        next_state_batch = expanded_next_state
                    
                    # Assicurati che i tensori abbiano la forma corretta [batch_size, state_dim]
                    if len(state_batch.shape) == 1:
                        state_batch = state_batch.unsqueeze(0)  # Aggiunge dimensione batch
                    elif state_batch.shape[1] == 1 and state_batch.shape[0] == config.batch_size:
                        # Lo stato ha forma [batch_size, 1], ma serve verificare che abbia abbastanza elementi
                        if state_batch.numel() == config.batch_size:  # Solo 1 elemento per batch
                            # Questo è un tensore con un solo elemento per batch, dobbiamo ricampionare
                            logging.warning(f"AVVISO: stato con un solo elemento per batch, impossibile ridimensionare a [batch_size, state_dim]")
                            # Crea un tensore fittizio con la forma corretta
                            dummy_state = torch.zeros(config.batch_size, state_dim, device=device)
                            # Copia i valori disponibili
                            dummy_state[:, 0:1] = state_batch
                            state_batch = dummy_state
                        else:
                            # Prova a ridimensionare con controllo
                            try:
                                state_batch = state_batch.view(config.batch_size, state_dim)
                            except RuntimeError:
                                logging.warning(f"AVVISO: impossibile ridimensionare stato da {state_batch.shape} a [{config.batch_size}, {state_dim}]")
                                # Crea un tensore fittizio
                                dummy_state = torch.zeros(config.batch_size, state_dim, device=device)
                                dummy_state[:, 0:1] = state_batch
                                state_batch = dummy_state
                        
                    if len(next_state_batch.shape) == 1:
                        next_state_batch = next_state_batch.unsqueeze(0)  # Aggiunge dimensione batch
                    elif next_state_batch.shape[1] == 1 and next_state_batch.shape[0] == config.batch_size:
                        # Lo stato successivo ha forma [batch_size, 1], ma serve verificare che abbia abbastanza elementi
                        if next_state_batch.numel() == config.batch_size:  # Solo 1 elemento per batch
                            # Questo è un tensore con un solo elemento per batch
                            logging.warning(f"AVVISO: next_state con un solo elemento per batch, impossibile ridimensionare a [batch_size, state_dim]")
                            # Crea un tensore fittizio con la forma corretta
                            dummy_next_state = torch.zeros(config.batch_size, state_dim, device=device)
                            # Copia i valori disponibili
                            dummy_next_state[:, 0:1] = next_state_batch
                            next_state_batch = dummy_next_state
                        else:
                            # Prova a ridimensionare con controllo
                            try:
                                next_state_batch = next_state_batch.view(config.batch_size, state_dim)
                            except RuntimeError:
                                logging.warning(f"AVVISO: impossibile ridimensionare next_state da {next_state_batch.shape} a [{config.batch_size}, {state_dim}]")
                                # Crea un tensore fittizio
                                dummy_next_state = torch.zeros(config.batch_size, state_dim, device=device)
                                dummy_next_state[:, 0:1] = next_state_batch
                                next_state_batch = dummy_next_state
                    
                    # Verifica che i tensori di stato abbiano la forma corretta [batch_size, state_dim]
                    if len(state_batch.shape) == 2 and state_batch.shape[0] == 1 and state_batch.shape[1] == config.batch_size:
                        # È trasposto [1, batch_size] invece di [batch_size, state_dim]
                        state_batch = state_batch.transpose(0, 1)
                    
                    if len(next_state_batch.shape) == 2 and next_state_batch.shape[0] == 1 and next_state_batch.shape[1] == config.batch_size:
                        # È trasposto [1, batch_size] invece di [batch_size, state_dim]
                        next_state_batch = next_state_batch.transpose(0, 1)
                    
                    # Debug: stampa le forme dei tensori dopo le correzioni
                    if episode <= 5 or episode % 100 == 0:
                        logging.debug(f"Debug - Forme corrette: stato {state_batch.shape}, azioni {action_batch.shape}, next_state {next_state_batch.shape}")
                        print(f"Debug - Forme corrette: stato {state_batch.shape}, azioni {action_batch.shape}, next_state {next_state_batch.shape}")
                    
                    # Calcola la loss e ottimizza
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            loss = agent.compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                        
                        # Ottimizzazione con scaler per mixed precision
                        agent.optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(agent.optimizer)
                        scaler.update()
                    else:
                        # Ottimizzazione normale
                        loss = agent.compute_loss(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
                        agent.optimizer.zero_grad()
                        loss.backward()
                        agent.optimizer.step()
                
                # Rendering
                if config.render:
                    env.render()
                    time.sleep(config.render_delay)
            
            # Fine dell'episodio
            epsilon = agent.epsilon
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            
            # Aggiorna statistiche
            stats['episodes'].append(episode)
            stats['rewards'].append(episode_reward)
            stats['lengths'].append(episode_steps)
            stats['losses'].append(avg_loss)
            stats['epsilons'].append(epsilon)
            stats['apples'].append(apple_count)
            
            # Valutazione periodica
            if episode % config.eval_freq == 0 or episode == config.total_episodes - 1:
                eval_stats = evaluate_agent(
                    agent, env, num_episodes=config.eval_episodes, render=config.render,
                    render_delay=config.render_delay
                )
                
                # Stampa le statistiche di valutazione
                eval_message = (f"Episodio {episode}/{config.total_episodes} | "
                              f"Reward: {episode_reward:.2f} | "
                              f"Passi: {episode_steps} | "
                              f"Mele: {apple_count} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"Epsilon: {epsilon:.4f} | "
                              f"Eval reward: {eval_stats['mean_reward']:.2f} | "
                              f"Eval mele: {eval_stats['mean_apples']:.2f}")
                logging.info(eval_message)
                print(eval_message)
                
                # Aggiorna statistiche di valutazione
                stats['eval_rewards'].append(eval_stats['mean_reward'])
                stats['eval_lengths'].append(eval_stats['mean_length'])
                stats['eval_apples'].append(eval_stats['mean_apples'])
                
                # Salva il miglior modello
                if eval_stats['mean_reward'] > stats['best_eval_reward']:
                    stats['best_eval_reward'] = eval_stats['mean_reward']
                    stats['best_eval_episode'] = episode
                    agent.save(best_model_path)
                    logging.info(f"Nuovo miglior modello salvato: {eval_stats['mean_reward']:.2f}")
                    print(f"Nuovo miglior modello salvato: {eval_stats['mean_reward']:.2f}")
                
                # Genera grafici
                plot_training_curves(stats, config.output_dir)
            else:
                # Stampa solo le statistiche dell'episodio
                logging.info(f"Episodio {episode}/{config.total_episodes} | "
                              f"Reward: {episode_reward:.2f} | "
                              f"Passi: {episode_steps} | "
                              f"Mele: {apple_count} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"Epsilon: {epsilon:.4f}")
                print(f"Episodio {episode}/{config.total_episodes} | "
                      f"Reward: {episode_reward:.2f} | "
                      f"Passi: {episode_steps} | "
                      f"Mele: {apple_count} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Epsilon: {epsilon:.4f}")
            
            # Salva checkpoint e statistiche
            if episode % config.checkpoint_freq == 0 or episode == config.total_episodes - 1:
                agent.save(checkpoint_path)
                with open(stats_path, 'w') as f:
                    json.dump(stats, f)
                logging.info(f"Checkpoint salvato all'episodio {episode}")
                print(f"Checkpoint salvato all'episodio {episode}")
        
        # Fine dell'addestramento
        elapsed_time = time.time() - start_time
        logging.info(f"Addestramento completato in {elapsed_time/60:.2f} minuti")
        logging.info(f"Miglior reward di valutazione: {stats['best_eval_reward']:.2f} all'episodio {stats['best_eval_episode']}")
        print(f"Addestramento completato in {elapsed_time/60:.2f} minuti")
        print(f"Miglior reward di valutazione: {stats['best_eval_reward']:.2f} all'episodio {stats['best_eval_episode']}")
        
        # Salva modello finale
        agent.save(os.path.join(config.output_dir, 'final_model.pt'))
        
    except KeyboardInterrupt:
        logging.warning("Addestramento interrotto dall'utente")
        print("Addestramento interrotto dall'utente")
        
        # Salva il modello e le statistiche
        agent.save(os.path.join(config.output_dir, 'interrupted_model.pt'))
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        logging.info(f"Modello salvato in {config.output_dir}")
        print(f"Modello salvato in {config.output_dir}")


def main():
    """Funzione principale."""
    # Analizza gli argomenti da linea di comando
    args = parse_args()
    
    # Carica la configurazione
    config = load_config(args)
    
    # Messaggio informativo sul logging
    if hasattr(config, 'log_file') and config.log_file:
        print(f"Logging attivato: i log saranno salvati nel file {config.log_file} con livello {config.log_level.upper()}")
    
    # Avvia l'addestramento
    train(config)


if __name__ == "__main__":
    main() 