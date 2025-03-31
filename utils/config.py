#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Configurazione Centralizzata
=================================
Gestisce le configurazioni per l'intero progetto tramite un sistema centralizzato.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Importa la funzione per rilevare l'hardware
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.hardware_utils import detect_hardware

# Configurazioni predefinite per i modelli
DEFAULT_MODEL_CONFIGS = {
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

# Configurazioni predefinte per l'addestramento
DEFAULT_TRAINING_CONFIGS = {
    "pretrain": {
        "base": {
            "episodes": 1000,
            "learning_rate": 1e-3,
            "grid_size": 10,
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
    },
    "train": {
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
}

# Configurazioni predefinite per l'ambiente di gioco
DEFAULT_ENVIRONMENT_CONFIGS = {
    "default": {
        "grid_size": 20,
        "speed": 10,
        "reward_food": 10.0,
        "reward_death": -10.0,
        "reward_step": -0.01,
        "reward_closer_to_food": 0.1,
        "reward_farther_from_food": -0.1,
        "max_steps_without_food": 100
    },
    "small": {
        "grid_size": 10,
        "speed": 5,
        "reward_food": 5.0,
        "reward_death": -5.0,
        "reward_step": -0.01,
        "reward_closer_to_food": 0.05,
        "reward_farther_from_food": -0.05,
        "max_steps_without_food": 50
    },
    "large": {
        "grid_size": 30,
        "speed": 15,
        "reward_food": 20.0,
        "reward_death": -20.0,
        "reward_step": -0.01,
        "reward_closer_to_food": 0.2,
        "reward_farther_from_food": -0.2,
        "max_steps_without_food": 200
    }
}

# Configurazioni predefinite per l'UI
DEFAULT_UI_CONFIGS = {
    "pygame": {
        "cell_size": 30,
        "fps": 60,
        "colors": {
            "background": [30, 30, 30],
            "grid": [50, 50, 50],
            "snake_head": [0, 255, 0],
            "snake_body": [0, 200, 0],
            "food": [255, 50, 50],
            "text": [255, 255, 255],
            "button": [100, 100, 100],
            "button_hover": [150, 150, 150]
        }
    },
    "web": {
        "host": "0.0.0.0",
        "port": 5000,
        "debug": True,
        "grid_size": 20,
        "update_interval": 100  # millisecondi
    }
}

class ConfigManager:
    """
    Gestore centralizzato delle configurazioni.
    
    Carica, salva e fornisce configurazioni per tutti i componenti 
    del progetto da file e/o default.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inizializza il ConfigManager.
        
        Args:
            config_path (str, optional): Percorso del file di configurazione
        """
        self.config_path = config_path
        self.default_configs = {
            "models": DEFAULT_MODEL_CONFIGS,
            "training": DEFAULT_TRAINING_CONFIGS,
            "environment": DEFAULT_ENVIRONMENT_CONFIGS,
            "ui": DEFAULT_UI_CONFIGS,
        }
        
        # Configurazione attuale
        self.configs = self.default_configs.copy()
        
        # Carica il file di configurazione se specificato
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
            
        # Rileva hardware e adatta le configurazioni
        self._adjust_for_hardware()
    
    def _load_config(self, config_path: str):
        """
        Carica le configurazioni da file.
        
        Args:
            config_path (str): Percorso del file di configurazione
        """
        ext = os.path.splitext(config_path)[1].lower()
        
        try:
            if ext == '.json':
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif ext in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                print(f"Formato file di configurazione non supportato: {ext}")
                return
                
            # Aggiorna le configurazioni con quelle caricate dal file
            self._update_nested_dict(self.configs, user_config)
            print(f"Configurazioni caricate da {config_path}")
        except Exception as e:
            print(f"Errore nel caricamento del file di configurazione: {e}")
    
    def save_config(self, config_path: Optional[str] = None):
        """
        Salva le configurazioni correnti su file.
        
        Args:
            config_path (str, optional): Percorso in cui salvare la configurazione.
                                        Se None, usa il percorso corrente.
        """
        save_path = config_path or self.config_path
        
        if not save_path:
            print("Nessun percorso specificato per salvare la configurazione.")
            return
            
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        ext = os.path.splitext(save_path)[1].lower()
        
        try:
            if ext == '.json':
                with open(save_path, 'w') as f:
                    json.dump(self.configs, f, indent=4)
            elif ext in ['.yaml', '.yml']:
                with open(save_path, 'w') as f:
                    yaml.dump(self.configs, f, default_flow_style=False)
            else:
                print(f"Formato file di configurazione non supportato: {ext}")
                return
                
            print(f"Configurazioni salvate in {save_path}")
        except Exception as e:
            print(f"Errore nel salvataggio del file di configurazione: {e}")
    
    def _update_nested_dict(self, d, u):
        """
        Aggiorna un dizionario annidato con un altro.
        
        Args:
            d (dict): Dizionario da aggiornare
            u (dict): Dizionario con cui aggiornare
            
        Returns:
            dict: Dizionario aggiornato
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    def _adjust_for_hardware(self):
        """Adatta le configurazioni in base all'hardware rilevato."""
        hw_info = detect_hardware()
        
        # Per ogni configurazione del modello
        for model_name, config in self.configs["models"].items():
            # Se ci sono GPU, aumenta il batch size
            if hw_info["gpu_available"]:
                config["batch_size"] = min(
                    config["batch_size"] * hw_info["gpu_count"], 
                    config["batch_size"] * 4
                )
                
                # Multi-GPU
                if hw_info["gpu_count"] > 1:
                    config["parallel_strategy"] = "DataParallel"
            else:
                # Limita batch size su CPU
                config["batch_size"] = min(config["batch_size"], 64)
                
            # Aggiungi informazioni hardware alla configurazione
            config["hardware"] = hw_info
            
            # Configura numero di workers per il dataloader
            config["num_workers"] = min(os.cpu_count() or 1, 4)
    
    def get_model_config(self, complexity: str = "base") -> Dict[str, Any]:
        """
        Ottiene la configurazione per un modello.
        
        Args:
            complexity (str): Livello di complessità del modello
            
        Returns:
            Dict[str, Any]: Configurazione del modello
        """
        if complexity not in self.configs["models"]:
            print(f"Complessità '{complexity}' non valida. Usa 'base', 'avanzato', 'complesso' o 'perfetto'.")
            complexity = "base"
            
        # Clona la configurazione per non modificare l'originale
        config = self.configs["models"][complexity].copy()
        
        # Aggiungi le configurazioni di training
        config["pretrain"] = self.configs["training"]["pretrain"][complexity]
        config["train"] = self.configs["training"]["train"][complexity]
        
        return config
    
    def get_environment_config(self, size: str = "default") -> Dict[str, Any]:
        """
        Ottiene la configurazione per l'ambiente di gioco.
        
        Args:
            size (str): Dimensione dell'ambiente ('small', 'default', 'large')
            
        Returns:
            Dict[str, Any]: Configurazione dell'ambiente
        """
        if size not in self.configs["environment"]:
            print(f"Dimensione '{size}' non valida. Usa 'small', 'default' o 'large'.")
            size = "default"
            
        return self.configs["environment"][size].copy()
    
    def get_ui_config(self, ui_type: str = "pygame") -> Dict[str, Any]:
        """
        Ottiene la configurazione per l'UI.
        
        Args:
            ui_type (str): Tipo di UI ('pygame', 'web')
            
        Returns:
            Dict[str, Any]: Configurazione dell'UI
        """
        if ui_type not in self.configs["ui"]:
            print(f"Tipo UI '{ui_type}' non valido. Usa 'pygame' o 'web'.")
            ui_type = "pygame"
            
        return self.configs["ui"][ui_type].copy()
    
    def update_model_config(self, complexity: str, updates: Dict[str, Any]):
        """
        Aggiorna la configurazione di un modello.
        
        Args:
            complexity (str): Livello di complessità
            updates (Dict[str, Any]): Aggiornamenti da applicare
        """
        if complexity not in self.configs["models"]:
            print(f"Complessità '{complexity}' non valida.")
            return
            
        self._update_nested_dict(self.configs["models"][complexity], updates)
    
    def update_environment_config(self, size: str, updates: Dict[str, Any]):
        """
        Aggiorna la configurazione dell'ambiente di gioco.
        
        Args:
            size (str): Dimensione dell'ambiente
            updates (Dict[str, Any]): Aggiornamenti da applicare
        """
        if size not in self.configs["environment"]:
            print(f"Dimensione '{size}' non valida.")
            return
            
        self._update_nested_dict(self.configs["environment"][size], updates)
    
    def update_ui_config(self, ui_type: str, updates: Dict[str, Any]):
        """
        Aggiorna la configurazione dell'UI.
        
        Args:
            ui_type (str): Tipo di UI
            updates (Dict[str, Any]): Aggiornamenti da applicare
        """
        if ui_type not in self.configs["ui"]:
            print(f"Tipo UI '{ui_type}' non valido.")
            return
            
        self._update_nested_dict(self.configs["ui"][ui_type], updates)

# Singleton per la gestione della configurazione
_config_manager = None

def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """
    Ottiene l'istanza del ConfigManager (singleton).
    
    Args:
        config_path (str, optional): Percorso del file di configurazione
        
    Returns:
        ConfigManager: Istanza del ConfigManager
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
        
    return _config_manager

# Funzioni di utility per ottenere configurazioni specifiche
def get_model_config(complexity: str = "base") -> Dict[str, Any]:
    """
    Ottiene la configurazione per un modello.
    
    Args:
        complexity (str): Livello di complessità del modello
        
    Returns:
        Dict[str, Any]: Configurazione del modello
    """
    return get_config_manager().get_model_config(complexity)

def get_environment_config(size: str = "default") -> Dict[str, Any]:
    """
    Ottiene la configurazione per l'ambiente di gioco.
    
    Args:
        size (str): Dimensione dell'ambiente
        
    Returns:
        Dict[str, Any]: Configurazione dell'ambiente
    """
    return get_config_manager().get_environment_config(size)

def get_ui_config(ui_type: str = "pygame") -> Dict[str, Any]:
    """
    Ottiene la configurazione per l'UI.
    
    Args:
        ui_type (str): Tipo di UI
        
    Returns:
        Dict[str, Any]: Configurazione dell'UI
    """
    return get_config_manager().get_ui_config(ui_type)

# Funzioni per salvare la configurazione attuale
def save_config(config_path: str):
    """
    Salva la configurazione attuale su file.
    
    Args:
        config_path (str): Percorso in cui salvare la configurazione
    """
    get_config_manager().save_config(config_path)


# Codice di test
if __name__ == "__main__":
    import pprint
    
    # Crea una nuova istanza di ConfigManager
    config_mgr = get_config_manager()
    
    print("Configurazione modello 'base':")
    pprint.pprint(config_mgr.get_model_config("base"))
    
    print("\nConfigurazione ambiente 'default':")
    pprint.pprint(config_mgr.get_environment_config("default"))
    
    print("\nConfigurazione UI 'pygame':")
    pprint.pprint(config_mgr.get_ui_config("pygame"))
    
    # Esempio di aggiornamento configurazione
    print("\nAggiornamento configurazione modello 'base'...")
    config_mgr.update_model_config("base", {"batch_size": 64, "lr": 2e-3})
    
    print("\nConfigurazione modello 'base' aggiornata:")
    pprint.pprint(config_mgr.get_model_config("base"))
    
    # Esempio di salvataggio configurazione
    # config_mgr.save_config("config.yaml") 