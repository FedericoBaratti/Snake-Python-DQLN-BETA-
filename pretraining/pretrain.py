#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Preaddestramento
====================
Implementa il preaddestramento sintetico per il modello DQN.

Autore: Federico Baratti
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

from pretraining.synthetic_env import SyntheticSnakeEnv, SyntheticExperienceGenerator
from dqn_agent.dqn_agent import DQNAgent
from dqn_agent.config import get_config
from backend.utils import set_seed, get_device, detect_hardware, create_checkpoint_dir, moving_average

def pretrain_agent(agent, experience_generator, steps, batch_size=32, update_interval=4, 
                  eval_interval=1000, save_interval=5000, plot=True):
    """
    Preaddestramento dell'agente DQN con esperienze sintetiche.
    
    Args:
        agent (DQNAgent): Agente DQN da addestrare
        experience_generator (SyntheticExperienceGenerator): Generatore di esperienze sintetiche
        steps (int): Numero totale di passi di preaddestramento
        batch_size (int): Dimensione del batch
        update_interval (int): Intervallo di aggiornamento del target network
        eval_interval (int): Intervallo di valutazione
        save_interval (int): Intervallo di salvataggio checkpoint
        plot (bool): Se visualizzare i grafici
        
    Returns:
        DQNAgent: Agente preaddestrato
    """
    print(f"Inizio preaddestramento per {steps} passi con batch size {batch_size}...")
    
    # Statistiche di addestramento
    rewards_history = []
    loss_history = []
    eval_rewards = []
    eval_steps = []
    
    # Crea una barra di avanzamento
    progress_bar = tqdm(total=steps, desc="Preaddestramento")
    
    # Preaddestramento per il numero specificato di passi
    step = 0
    while step < steps:
        # Genera un batch di esperienze
        experiences = experience_generator.generate_batch(batch_size)
        
        # Memorizza le esperienze e aggiorna l'agente
        total_loss = 0.0
        for state, action, next_state, reward, done in experiences:
            # Esegui un passo di training
            loss = agent.train_step(state, action, next_state, reward, done)
            total_loss += loss
            
            # Aggiorna il contatore
            step += 1
            progress_bar.update(1)
            
            # Aggiorna statistiche
            rewards_history.append(reward)
            loss_history.append(loss)
            
            # Valutazione periodica
            if step % eval_interval == 0:
                eval_reward, _ = evaluate_agent(agent, SyntheticSnakeEnv())
                eval_rewards.append(eval_reward)
                eval_steps.append(step)
                print(f"\nValutazione al passo {step}: Reward medio {eval_reward:.2f}")
            
            # Salvataggio periodico
            if step % save_interval == 0:
                save_path = agent.save(
                    path=f"training/checkpoints/pretrain_{agent.complexity}_{step}.pt",
                    additional_info={"pretrain_steps": step}
                )
                print(f"\nSalvataggio checkpoint al passo {step}: {save_path}")
            
            # Termina se abbiamo raggiunto il numero di passi
            if step >= steps:
                break
    
    # Chiudi la barra di avanzamento
    progress_bar.close()
    
    # Salva il modello finale
    final_path = agent.save(
        path=f"training/checkpoints/pretrain_{agent.complexity}_final.pt",
        additional_info={"pretrain_steps": steps}
    )
    print(f"Preaddestramento completato. Modello salvato in {final_path}")
    
    # Visualizza le statistiche
    if plot:
        plt.figure(figsize=(15, 10))
        
        # Plot delle ricompense durante il training
        plt.subplot(2, 2, 1)
        window = min(100, len(rewards_history) // 10)
        smoothed_rewards = moving_average(rewards_history, window)
        plt.plot(range(len(smoothed_rewards)), smoothed_rewards)
        plt.title(f'Ricompense durante preaddestramento (Media mobile {window})')
        plt.xlabel('Passi')
        plt.ylabel('Ricompensa')
        
        # Plot della loss
        plt.subplot(2, 2, 2)
        window = min(100, len(loss_history) // 10)
        smoothed_loss = moving_average(loss_history, window)
        plt.plot(range(len(smoothed_loss)), smoothed_loss)
        plt.title(f'Loss durante preaddestramento (Media mobile {window})')
        plt.xlabel('Passi')
        plt.ylabel('Loss')
        
        # Plot delle ricompense di valutazione
        plt.subplot(2, 2, 3)
        plt.plot(eval_steps, eval_rewards, 'ro-')
        plt.title('Ricompense di valutazione')
        plt.xlabel('Passi')
        plt.ylabel('Ricompensa media')
        
        # Grafico del valore di epsilon
        plt.subplot(2, 2, 4)
        plt.plot(range(len(agent.stats["epsilon_history"])), agent.stats["epsilon_history"])
        plt.title('Epsilon durante preaddestramento')
        plt.xlabel('Passi')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        
        # Salva il grafico
        plot_path = Path("training") / "plots"
        plot_path.mkdir(exist_ok=True)
        plt.savefig(plot_path / f"pretrain_{agent.complexity}_{steps}.png")
        plt.close()
    
    return agent

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

def pretrain_worker(rank, args, config, return_dict):
    """
    Worker per il preaddestramento parallelo.
    
    Args:
        rank (int): Rango del processo
        args: Argomenti da linea di comando
        config: Configurazione
        return_dict: Dizionario per restituire l'agente addestrato
    """
    # Configura il seed per la riproducibilità
    set_seed(args.seed + rank)
    
    # Crea il generatore di esperienze
    grid_size = config["pretrain"]["grid_size"]
    experience_generator = SyntheticExperienceGenerator(
        grid_size=grid_size,
        max_steps=200
    )
    
    # Crea l'ambiente sintetico
    env = SyntheticSnakeEnv(grid_size=grid_size)
    
    # Ottieni dimensioni di stato e azione
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # Crea l'agente
    device = get_device()
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        complexity=args.complexity,
        config=config
    )
    
    # Calcola il numero di passi per questo worker
    steps_per_worker = args.steps // args.num_workers
    batch_size = config["pretrain"]["batch_size"]
    
    # Preaddestra l'agente
    print(f"Worker {rank}: Preaddestramento per {steps_per_worker} passi...")
    agent = pretrain_agent(
        agent=agent,
        experience_generator=experience_generator,
        steps=steps_per_worker,
        batch_size=batch_size,
        update_interval=config["target_update"],
        eval_interval=steps_per_worker // 10,
        save_interval=steps_per_worker // 2,
        plot=(rank == 0)  # Solo il primo worker genera grafici
    )
    
    # Restituisci l'agente
    return_dict[rank] = agent

def main(args):
    """
    Esegue il preaddestramento sintetico dell'agente DQN.
    
    Args:
        args: Argomenti da linea di comando
    """
    print("Avvio preaddestramento sintetico...")
    
    # Imposta il seed per la riproducibilità
    set_seed(args.seed)
    
    # Ottieni la configurazione
    config = get_config(args.complexity)
    
    # Rileva l'hardware disponibile
    hw_info = detect_hardware()
    print(f"Hardware rilevato: {hw_info['device']}")
    print(f"  - CPU: {hw_info['cpu_count']} cores")
    if hw_info['gpu_available']:
        print(f"  - GPU: {hw_info['gpu_count']} ({', '.join(hw_info['gpu_names'])})")
    
    # Crea la directory per i checkpoint
    create_checkpoint_dir()
    
    # Configura il numero di worker in base alle CPU disponibili
    args.num_workers = min(args.num_workers, hw_info['cpu_count'])
    
    if args.num_workers > 1 and not args.no_mp:
        print(f"Avvio preaddestramento parallelo con {args.num_workers} workers...")
        # Preaddestramento multi-processo
        
        # Inizializza il dizionario condiviso
        manager = mp.Manager()
        return_dict = manager.dict()
        
        # Avvia i processi
        processes = []
        for rank in range(args.num_workers):
            p = mp.Process(
                target=pretrain_worker,
                args=(rank, args, config, return_dict)
            )
            p.start()
            processes.append(p)
        
        # Attendi il completamento dei processi
        for p in processes:
            p.join()
        
        # Seleziona l'agente con le prestazioni migliori
        best_agent = None
        best_reward = float('-inf')
        
        for rank, agent in return_dict.items():
            if agent.best_reward > best_reward:
                best_agent = agent
                best_reward = agent.best_reward
        
        # Salva il miglior agente
        if best_agent:
            final_path = best_agent.save(
                path=f"training/checkpoints/pretrain_{args.complexity}_best.pt",
                additional_info={"pretrain_steps": args.steps}
            )
            print(f"Preaddestramento parallelo completato. Miglior modello salvato in {final_path}")
    else:
        print("Avvio preaddestramento sequenziale...")
        # Preaddestramento sequenziale
        
        # Crea il generatore di esperienze
        grid_size = config["pretrain"]["grid_size"]
        experience_generator = SyntheticExperienceGenerator(
            grid_size=grid_size,
            max_steps=200
        )
        
        # Crea l'ambiente sintetico
        env = SyntheticSnakeEnv(grid_size=grid_size)
        
        # Ottieni dimensioni di stato e azione
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # Crea l'agente
        device = get_device()
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            complexity=args.complexity,
            config=config
        )
        
        # Preaddestra l'agente
        batch_size = config["pretrain"]["batch_size"]
        agent = pretrain_agent(
            agent=agent,
            experience_generator=experience_generator,
            steps=args.steps,
            batch_size=batch_size,
            update_interval=config["target_update"],
            eval_interval=args.steps // 10,
            save_interval=args.steps // 5,
            plot=True
        )
        
        # Valuta l'agente finale
        final_reward, final_length = evaluate_agent(agent, env, episodes=10)
        print(f"Valutazione finale: Reward medio {final_reward:.2f}, Lunghezza media {final_length:.2f}")
    
    print("Preaddestramento completato!")

if __name__ == "__main__":
    # Argomenti da linea di comando
    parser = argparse.ArgumentParser(description='Preaddestramento sintetico per DQN')
    
    parser.add_argument('--complexity', type=str, default='base',
                        choices=['base', 'avanzato', 'complesso', 'perfetto'],
                        help='Complessità del modello DQN')
    
    parser.add_argument('--steps', type=int, default=None,
                        help='Numero di passi di preaddestramento')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed per la generazione casuale')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Numero di worker per il preaddestramento parallelo')
    
    parser.add_argument('--no_mp', action='store_true',
                        help='Disabilita il multiprocessing')
    
    args = parser.parse_args()
    
    # Se non è specificato il numero di passi, usa quello predefinito dalla configurazione
    if args.steps is None:
        config = get_config(args.complexity)
        args.steps = config["pretrain"]["synthetic_steps"]
    
    # Esegui il preaddestramento
    main(args) 