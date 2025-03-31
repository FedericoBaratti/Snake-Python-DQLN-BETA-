#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Snake-Python-DQLN
=================
Punto d'ingresso principale del progetto Snake con Deep Q-Learning.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import sys
import argparse
import time
from pathlib import Path

def parse_args():
    """Analizza gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Snake-Python-DQLN - Gioco Snake con Deep Q-Learning')
    
    # Modalità di esecuzione
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'web'], default='play',
                        help='Modalità di esecuzione: train, play, web')
    
    # Parametri comuni
    parser.add_argument('--config', type=str, default=None,
                        help='Percorso del file di configurazione')
    
    # Parametri per l'addestramento
    parser.add_argument('--agent', type=str, default='dqn',
                        help='Tipo di agente (dqn, ...)')
    parser.add_argument('--model', type=str, default='dueling',
                        help='Tipo di modello (dqn, dueling, noisy, noisy_dueling)')
    parser.add_argument('--buffer', type=str, default='prioritized',
                        help='Tipo di buffer (standard, prioritized)')
    
    # Parametri per il gioco
    parser.add_argument('--model_path', type=str, default=None,
                        help='Percorso del modello addestrato')
    parser.add_argument('--play_mode', type=str, choices=['ai', 'human', 'compare'], default='ai',
                        help='Modalità di gioco: ai, human, compare')
    
    # Altri parametri
    parser.add_argument('--device', type=str, default='auto',
                        help='Dispositivo (cuda:0, cpu)')
    
    args = parser.parse_args()
    return args

def main():
    """Funzione principale."""
    args = parse_args()
    
    # Modalità di esecuzione
    if args.mode == 'train':
        # Importa solo quando necessario per ridurre l'uso della memoria
        from train import main as train_main
        
        # Prepara gli argomenti per il training
        sys.argv = [sys.argv[0]]
        
        # Aggiungi gli argomenti
        if args.config:
            sys.argv.extend(['--config', args.config])
        if args.agent:
            sys.argv.extend(['--agent', args.agent])
        if args.model:
            sys.argv.extend(['--model', args.model])
        if args.buffer:
            sys.argv.extend(['--buffer', args.buffer])
        if args.device:
            sys.argv.extend(['--device', args.device])
        
        # Avvia il training
        print("Avvio modalità di addestramento...")
        train_main()
    
    elif args.mode == 'play':
        # Importa solo quando necessario
        from play import main as play_main
        
        # Prepara gli argomenti per il gioco
        sys.argv = [sys.argv[0]]
        
        # Aggiungi gli argomenti
        sys.argv.extend(['--mode', args.play_mode])
        
        if args.model_path:
            sys.argv.extend(['--model_path', args.model_path])
        if args.device:
            sys.argv.extend(['--device', args.device])
        
        # Avvia il gioco
        print(f"Avvio modalità di gioco: {args.play_mode}...")
        play_main(sys.argv[1:])
    
    elif args.mode == 'web':
        # Importazione condizionale
        try:
            from run_web import main as web_main
            print("Avvio interfaccia web...")
            web_main()
        except ImportError:
            print("Errore: modulo web non trovato. Esegui 'pip install flask' per installare le dipendenze.")
    
    else:
        print(f"Modalità non valida: {args.mode}")
        print("Usa --mode train|play|web")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgramma interrotto dall'utente.")
        sys.exit(0) 