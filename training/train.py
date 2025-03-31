#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Training
============
Implementa il training reale ottimizzato per il modello DQN.

Autore: Baratti Federico
Versione: 1.0
"""

import os
import time
import argparse
import numpy as np
import random
import torch
import torch.multiprocessing as mp
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
from torch.utils.tensorboard import SummaryWriter

from backend.snake_game import SnakeGame
from backend.environment import SnakeEnv
from dqn_agent.dqn_agent import DQNAgent
from dqn_agent.config import get_config
from backend.utils import set_seed, get_device, detect_hardware, create_checkpoint_dir, moving_average, timeit

def train_agent(model_complexity="base", checkpoint_path=None, config=None,
               num_episodes=None, grid_size=20, device=None):
    """
    Addestra l'agente DQN nell'ambiente Snake.
    
    Args:
        model_complexity (str): Complessità del modello ('base', 'avanzato', 'complesso', 'perfetto')
        checkpoint_path (str, optional): Percorso del checkpoint da caricare
        config (dict, optional): Configurazione personalizzata
        num_episodes (int, optional): Numero di episodi di training
        grid_size (int): Dimensione della griglia di gioco
        device (torch.device, optional): Dispositivo su cui eseguire il training
        
    Returns:
        DQNAgent: Agente addestrato
    """
    # Ottieni la configurazione
    if config is None:
        config = get_config(complexity=model_complexity)
    
    # Ottieni il dispositivo
    if device is None:
        device = get_device()
    
    # Inizializza l'ambiente
    game = SnakeGame(grid_size=grid_size)
    env = SnakeEnv(game, use_normalized_state=True)
    
    # Ottieni le dimensioni di stato e azione
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Imposta il numero di episodi
    if num_episodes is None:
        num_episodes = config["train"]["episodes"]
    
    # Inizializza l'agente
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Caricamento agente dal checkpoint: {checkpoint_path}")
        agent = DQNAgent.load(checkpoint_path, device=device)
        
        # Verifica compatibilità dell'agente con la nuova dimensione delle azioni
        if agent.action_dim != action_dim:
            print(f"ATTENZIONE: L'agente caricato ha {agent.action_dim} azioni, ma l'ambiente ne ha {action_dim}")
            print("Creazione di un nuovo agente con la dimensione corretta delle azioni")
            agent = DQNAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                complexity=model_complexity,
                config=config
            )
    else:
        print(f"Creazione nuovo agente con complessità: {model_complexity}")
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            complexity=model_complexity,
            config=config
        )
    
    # Setup tensorboard
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path("training") / "logs" / f"{model_complexity}_{current_time}"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    # Parametri di training
    batch_size = config["train"]["batch_size"]
    evaluation_interval = config["train"]["evaluation_interval"]
    save_interval = config["train"]["save_interval"]
    
    print(f"Inizio training per {num_episodes} episodi (batch size: {batch_size})...")
    
    # Statistiche di training
    rewards_history = []
    lengths_history = []
    eval_rewards = []
    eval_episodes = []
    total_steps = agent.total_steps
    
    # Crea una barra di avanzamento
    progress_bar = tqdm(total=num_episodes, desc="Training")
    
    # Loop principale di training
    for episode in range(1, num_episodes + 1):
        # Resetta l'ambiente
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        done = False
        
        # Loop di un episodio
        while not done:
            # Seleziona un'azione
            action = agent.select_action(state)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Esegui un passo di training
            loss = agent.train_step(state, action, next_state, reward, done)
            
            # Aggiorna statistiche
            episode_reward += reward
            episode_length += 1
            episode_loss += loss
            total_steps += 1
            
            # Passa allo stato successivo
            state = next_state
        
        # Aggiorna le statistiche dell'episodio
        agent.end_episode(episode_reward, episode_length)
        
        # Aggiungi alle statistiche
        rewards_history.append(episode_reward)
        lengths_history.append(episode_length)
        
        # Calcola la media delle loss
        avg_loss = episode_loss / max(1, episode_length)
        
        # Aggiorna tensorboard
        writer.add_scalar('training/reward', episode_reward, episode)
        writer.add_scalar('training/episode_length', episode_length, episode)
        writer.add_scalar('training/avg_loss', avg_loss, episode)
        writer.add_scalar('training/epsilon', agent.epsilon, episode)
        
        # Stampa statistiche
        progress_bar.set_postfix({
            'reward': f"{episode_reward:.1f}",
            'length': episode_length,
            'epsilon': f"{agent.epsilon:.3f}"
        })
        progress_bar.update(1)
        
        # Valutazione periodica
        if episode % evaluation_interval == 0:
            eval_reward, eval_length = evaluate_agent(agent, env, episodes=5)
            eval_rewards.append(eval_reward)
            eval_episodes.append(episode)
            
            # Aggiorna tensorboard
            writer.add_scalar('evaluation/reward', eval_reward, episode)
            writer.add_scalar('evaluation/episode_length', eval_length, episode)
            
            # Stampa statistiche di valutazione
            print(f"\nValutazione all'episodio {episode}: Reward medio {eval_reward:.2f}, Lunghezza {eval_length:.2f}")
        
        # Salvataggio periodico
        if episode % save_interval == 0:
            save_path = agent.save(
                path=f"training/checkpoints/dqn_{model_complexity}_ep{episode}.pt",
                additional_info={"episode": episode, "total_steps": total_steps}
            )
            print(f"\nSalvataggio checkpoint all'episodio {episode}: {save_path}")
    
    # Chiudi la barra di avanzamento
    progress_bar.close()
    
    # Salva il modello finale
    final_path = agent.save(
        path=f"training/checkpoints/dqn_{model_complexity}_final.pt",
        additional_info={"episode": num_episodes, "total_steps": total_steps}
    )
    print(f"Training completato. Modello finale salvato in {final_path}")
    
    # Chiudi tensorboard
    writer.close()
    
    # Genera e salva i grafici
    plot_training_results(
        rewards_history=rewards_history,
        lengths_history=lengths_history,
        eval_rewards=eval_rewards,
        eval_episodes=eval_episodes,
        epsilon_history=agent.stats["epsilon_history"],
        model_complexity=model_complexity,
        num_episodes=num_episodes
    )
    
    return agent

@timeit
def evaluate_agent(agent, env, episodes=5):
    """
    Valuta le prestazioni dell'agente in un ambiente.
    
    Args:
        agent (DQNAgent): Agente DQN da valutare
        env: Ambiente di valutazione
        episodes (int): Numero di episodi
        
    Returns:
        tuple: (reward medio, lunghezza media episodio)
    """
    total_rewards = []
    total_lengths = []
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            # Seleziona un'azione (senza esplorazione casuale)
            action = agent.select_action(state, training=False)
            
            # Esegui l'azione
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Aggiorna reward e lunghezza
            episode_reward += reward
            episode_length += 1
            
            # Aggiorna lo stato
            state = next_state
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
    
    # Calcola le medie
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_length = sum(total_lengths) / len(total_lengths)
    
    return avg_reward, avg_length

def plot_training_results(rewards_history, lengths_history, eval_rewards, eval_episodes,
                          epsilon_history, model_complexity, num_episodes):
    """
    Genera e salva i grafici dei risultati di training.
    
    Args:
        rewards_history (list): Storico delle ricompense per episodio
        lengths_history (list): Storico delle lunghezze degli episodi
        eval_rewards (list): Ricompense dalle valutazioni periodiche
        eval_episodes (list): Episodi in cui sono state fatte le valutazioni
        epsilon_history (list): Storico dei valori di epsilon
        model_complexity (str): Complessità del modello
        num_episodes (int): Numero totale di episodi
    """
    plt.figure(figsize=(15, 10))
    
    # Plot delle ricompense durante il training
    plt.subplot(2, 2, 1)
    window = min(100, len(rewards_history) // 10)
    if window > 0:
        smoothed_rewards = moving_average(rewards_history, window)
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
        plt.title(f'Ricompense durante training (Media mobile {window})')
    else:
        plt.plot(range(len(rewards_history)), rewards_history)
        plt.title('Ricompense durante training')
    plt.xlabel('Episodi')
    plt.ylabel('Ricompensa')
    
    # Plot delle lunghezze degli episodi
    plt.subplot(2, 2, 2)
    window = min(100, len(lengths_history) // 10)
    if window > 0:
        smoothed_lengths = moving_average(lengths_history, window)
        plt.plot(range(len(smoothed_lengths)), smoothed_lengths)
        plt.title(f'Lunghezze episodi (Media mobile {window})')
    else:
        plt.plot(range(len(lengths_history)), lengths_history)
        plt.title('Lunghezze episodi')
    plt.xlabel('Episodi')
    plt.ylabel('Lunghezza episodio')
    
    # Plot delle ricompense di valutazione
    plt.subplot(2, 2, 3)
    plt.plot(eval_episodes, eval_rewards, 'ro-')
    plt.title('Ricompense di valutazione')
    plt.xlabel('Episodi')
    plt.ylabel('Ricompensa media')
    
    # Grafico del valore di epsilon
    plt.subplot(2, 2, 4)
    plt.plot(range(len(epsilon_history)), epsilon_history)
    plt.title('Epsilon durante training')
    plt.xlabel('Passi')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    
    # Salva il grafico
    plot_path = Path("training") / "plots"
    plot_path.mkdir(exist_ok=True)
    plt.savefig(plot_path / f"train_{model_complexity}_{num_episodes}.png")
    plt.close()

def train_parallel(args, hw_info):
    """
    Esegue il training parallelo su più processi.
    
    Args:
        args: Argomenti da linea di comando
        hw_info: Informazioni sull'hardware disponibile
    """
    # Configura il numero di worker in base alle GPU disponibili
    if hw_info["gpu_available"]:
        # Usa un worker per GPU
        num_workers = min(args.num_workers, hw_info["gpu_count"])
    else:
        # Usa un worker per due core CPU
        num_workers = min(args.num_workers, max(1, hw_info["cpu_count"] // 2))
    
    print(f"Avvio training parallelo con {num_workers} workers...")
    
    # Inizializza il dizionario condiviso
    manager = mp.Manager()
    return_dict = manager.dict()
    
    # Ottieni la configurazione
    config = get_config(args.complexity)
    
    # Distribuisci gli episodi tra i worker
    episodes_per_worker = args.episodes // num_workers
    
    # Avvia i processi
    processes = []
    for rank in range(num_workers):
        # Assegna una GPU specifica a ciascun worker se disponibile
        if hw_info["gpu_available"] and rank < hw_info["gpu_count"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        p = mp.Process(
            target=train_worker,
            args=(rank, args, config, episodes_per_worker, device, return_dict)
        )
        p.start()
        processes.append(p)
    
    # Attendi il completamento dei processi
    for p in processes:
        p.join()
    
    # Seleziona l'agente con le prestazioni migliori
    best_agent = None
    best_reward = float('-inf')
    
    for rank, (agent, avg_reward) in return_dict.items():
        if avg_reward > best_reward:
            best_agent = agent
            best_reward = avg_reward
    
    # Salva il miglior agente
    if best_agent:
        final_path = best_agent.save(
            path=f"training/checkpoints/dqn_{args.complexity}_best.pt",
            additional_info={"episodes": args.episodes, "reward": best_reward}
        )
        print(f"Training parallelo completato. Miglior modello salvato in {final_path}")
    
    return best_agent

def train_worker(rank, args, config, num_episodes, device, return_dict):
    """
    Worker per il training parallelo.
    
    Args:
        rank (int): Rango del processo
        args: Argomenti da linea di comando
        config: Configurazione
        num_episodes (int): Numero di episodi per questo worker
        device (torch.device): Dispositivo su cui eseguire il training
        return_dict: Dizionario condiviso per restituire l'agente addestrato
    """
    # Configura il seed per la riproducibilità
    set_seed(args.seed + rank)
    
    # Crea ambiente e agente
    checkpoint_path = args.checkpoint if rank == 0 else None
    
    print(f"Worker {rank} su {device}: Training per {num_episodes} episodi...")
    
    # Addestra l'agente
    agent = train_agent(
        model_complexity=args.complexity,
        checkpoint_path=checkpoint_path,
        config=config,
        num_episodes=num_episodes,
        grid_size=args.grid_size,
        device=device
    )
    
    # Valuta l'agente finale
    game = SnakeGame(grid_size=args.grid_size)
    env = SnakeEnv(game, use_normalized_state=True)
    avg_reward, _ = evaluate_agent(agent, env, episodes=10)
    
    # Restituisci l'agente e la sua performance
    return_dict[rank] = (agent, avg_reward)

def main(args):
    """
    Esegue il training dell'agente DQN.
    
    Args:
        args: Argomenti da linea di comando
    """
    print("Avvio training DQN...")
    
    # Imposta il seed per la riproducibilità
    set_seed(args.seed)
    
    # Rileva l'hardware disponibile
    hw_info = detect_hardware()
    print(f"Hardware rilevato: {hw_info['device']}")
    print(f"  - CPU: {hw_info['cpu_count']} cores")
    if hw_info['gpu_available']:
        print(f"  - GPU: {hw_info['gpu_count']} ({', '.join(hw_info['gpu_names'])})")
    
    # Crea la directory per i checkpoint
    create_checkpoint_dir()
    
    # Ottieni la configurazione
    config = get_config(args.complexity)
    
    # Se non è specificato il numero di episodi, usa quello predefinito dalla configurazione
    if args.episodes is None:
        args.episodes = config["train"]["episodes"]
    
    # Configura il numero di worker in base alle CPU disponibili
    args.num_workers = min(args.num_workers, max(1, hw_info["cpu_count"]))
    
    # Esegui il training (parallelo o sequenziale)
    if args.num_workers > 1 and not args.no_mp and (hw_info["gpu_count"] > 1 or not hw_info["gpu_available"]):
        # Training parallelo (ha senso solo con multiple GPU o senza GPU)
        agent = train_parallel(args, hw_info)
    else:
        # Training sequenziale
        print("Avvio training sequenziale...")
        agent = train_agent(
            model_complexity=args.complexity,
            checkpoint_path=args.checkpoint,
            config=config,
            num_episodes=args.episodes,
            grid_size=args.grid_size
        )
    
    print("Training completato!")
    
    return agent

if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Training DQN per Snake')
    
    parser.add_argument('--complexity', type=str, default='base',
                        choices=['base', 'avanzato', 'complesso', 'perfetto'],
                        help='Complessità del modello DQN')
    
    parser.add_argument('--episodes', type=int, default=None,
                        help='Numero di episodi di training')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la generazione casuale')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Percorso del checkpoint da caricare')
    
    parser.add_argument('--grid_size', type=int, default=20,
                        help='Dimensione della griglia di gioco')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Numero di worker per il training parallelo')
    
    parser.add_argument('--no_mp', action='store_true',
                        help='Disabilita il multiprocessing')
    
    args = parser.parse_args()
    
    # Esegui il training
    main(args) 