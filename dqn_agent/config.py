#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Configurazione DQN
=======================
Gestisce le configurazioni per i diversi livelli di complessità del modello DQN.

Autore: Federico Baratti
Versione: 1.0
"""

import os
from backend.utils import detect_hardware

# Configura i livelli di complessità del modello
MODEL_CONFIGS = {
    "base": {
        "description": "Modello base con architettura semplice (3k parametri)",
        "hidden_layers": [64, 32],
        "activation": "ReLU",
        "batch_size": 32,
        "lr": 1e-3,
        "target_update": 10,
        "memory_size": 10000,
        "gamma": 0.9,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 5000,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "min_hardware": {
            "cpu_cores": 1,
            "gpu_memory": 0,  # Funziona bene anche senza GPU
        }
    },
    "avanzato": {
        "description": "Modello avanzato con architettura più profonda (12k parametri)",
        "hidden_layers": [128, 64, 32],
        "activation": "ReLU",
        "batch_size": 64,
        "lr": 5e-4,
        "target_update": 5,
        "memory_size": 50000,
        "gamma": 0.95,
        "eps_start": 1.0,
        "eps_end": 0.01,
        "eps_decay": 10000,
        "optimizer": "Adam",
        "loss": "HuberLoss",
        "min_hardware": {
            "cpu_cores": 2,
            "gpu_memory": 1024,  # ~1GB VRAM consigliato
        }
    },
    "complesso": {
        "description": "Modello complesso con architettura profonda (40k parametri)",
        "hidden_layers": [256, 128, 64, 32],
        "activation": "ReLU",
        "batch_size": 128,
        "lr": 3e-4,
        "target_update": 3,
        "memory_size": 100000,
        "gamma": 0.98,
        "eps_start": 1.0,
        "eps_end": 0.005,
        "eps_decay": 20000,
        "optimizer": "Adam",
        "loss": "HuberLoss",
        "min_hardware": {
            "cpu_cores": 4,
            "gpu_memory": 2048,  # ~2GB VRAM consigliato
        }
    },
    "perfetto": {
        "description": "Modello perfetto con architettura molto profonda (170k parametri)",
        "hidden_layers": [512, 256, 128, 64],
        "activation": "ReLU",
        "batch_size": 256,
        "lr": 1e-4,
        "target_update": 2,
        "memory_size": 200000,
        "gamma": 0.99,
        "eps_start": 1.0,
        "eps_end": 0.001,
        "eps_decay": 30000,
        "optimizer": "Adam",
        "loss": "HuberLoss",
        "min_hardware": {
            "cpu_cores": 8,
            "gpu_memory": 4096,  # ~4GB VRAM consigliato
        }
    }
}

# Configurazioni per il preaddestramento
PRETRAIN_CONFIGS = {
    "base": {
        "episodes": 1000,
        "learning_rate": 1e-3,
        "grid_size": 10,  # Griglia più piccola per pretraining
        "batch_size": 32,
        "synthetic_steps": 50000,
    },
    "avanzato": {
        "episodes": 2000,
        "learning_rate": 5e-4,
        "grid_size": 10,
        "batch_size": 64,
        "synthetic_steps": 100000,
    },
    "complesso": {
        "episodes": 5000,
        "learning_rate": 3e-4,
        "grid_size": 15,
        "batch_size": 128,
        "synthetic_steps": 200000,
    },
    "perfetto": {
        "episodes": 10000,
        "learning_rate": 1e-4,
        "grid_size": 15,
        "batch_size": 256,
        "synthetic_steps": 500000,
    }
}

# Configurazioni per il training reale
TRAIN_CONFIGS = {
    "base": {
        "episodes": 5000,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "evaluation_interval": 100,
        "save_interval": 500,
    },
    "avanzato": {
        "episodes": 10000,
        "learning_rate": 5e-4,
        "batch_size": 64,
        "evaluation_interval": 200,
        "save_interval": 1000,
    },
    "complesso": {
        "episodes": 20000,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "evaluation_interval": 500,
        "save_interval": 2000,
    },
    "perfetto": {
        "episodes": 50000,
        "learning_rate": 1e-4,
        "batch_size": 256,
        "evaluation_interval": 1000,
        "save_interval": 5000,
    }
}


def get_config(complexity="base"):
    """
    Ottiene la configurazione per il livello di complessità specificato.
    
    Args:
        complexity (str): Livello di complessità ('base', 'avanzato', 'complesso', 'perfetto')
        
    Returns:
        dict: Configurazione per il livello specificato
    """
    # Scegli la complessità adeguata se non specificata
    if complexity not in MODEL_CONFIGS:
        print(f"Complessità '{complexity}' non valida. Utilizza 'base', 'avanzato', 'complesso' o 'perfetto'.")
        complexity = "base"
    
    # Ottieni la configurazione di base per il livello specificato
    config = MODEL_CONFIGS[complexity].copy()
    
    # Aggiungi configurazioni di pretraining e training
    config["pretrain"] = PRETRAIN_CONFIGS[complexity]
    config["train"] = TRAIN_CONFIGS[complexity]
    
    # Rileva automaticamente l'hardware e aggiusta la configurazione
    hw_info = detect_hardware()
    config["hardware"] = hw_info
    
    # Regola configurazioni in base all'hardware disponibile
    if hw_info["gpu_available"]:
        # Se ci sono GPU, aumenta il batch size in base al numero di GPU
        config["batch_size"] = min(config["batch_size"] * hw_info["gpu_count"], config["batch_size"] * 4)
        
        # Aggiungi opzioni di parallelizzazione per multi-GPU
        if hw_info["gpu_count"] > 1:
            config["parallel_strategy"] = "DataParallel"
    else:
        # Se non ci sono GPU, mantieni un batch size ragionevole per CPU
        config["batch_size"] = min(config["batch_size"], 64)
    
    # Usa mini-batch più piccoli se memoria limitata
    if "min_hardware" in config and config["min_hardware"]["gpu_memory"] > 0:
        # Logica per adattare i batch in base alla memoria disponibile
        # (Informazione non facilmente disponibile in PyTorch)
        pass
    
    # Configura numero di worker per il dataloader in base alle CPU disponibili
    config["num_workers"] = min(os.cpu_count() or 1, 4)
    
    return config


def list_available_configs():
    """
    Elenca tutte le configurazioni disponibili con dettagli.
    
    Returns:
        list: Lista di configurazioni disponibili con dettagli
    """
    result = []
    for name, config in MODEL_CONFIGS.items():
        result.append({
            "name": name,
            "description": config["description"],
            "parameters": f"{len(config['hidden_layers']) + 1} layers, {config['hidden_layers']}",
            "memory": config["memory_size"],
            "min_hardware": config["min_hardware"]
        })
    return result


if __name__ == "__main__":
    # Testa la configurazione
    import pprint
    
    print("Configurazioni disponibili:")
    configs = list_available_configs()
    for cfg in configs:
        print(f"- {cfg['name']}: {cfg['description']}")
        print(f"  Parametri: {cfg['parameters']}")
        print(f"  Memoria: {cfg['memory']}")
        print(f"  Hardware minimo: CPU: {cfg['min_hardware']['cpu_cores']} cores, GPU: {cfg['min_hardware']['gpu_memory']} MB VRAM")
        print()
    
    print("\nConfigurazione 'base' con hardware rilevato:")
    config = get_config("base")
    pprint.pprint(config["hardware"]) 