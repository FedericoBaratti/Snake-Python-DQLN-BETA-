#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Utility Hardware
====================
Funzioni per il rilevamento e la gestione dell'hardware disponibile.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import torch
import platform
import psutil
from typing import Dict, Any

def get_device():
    """
    Determina il dispositivo migliore disponibile (CUDA, MPS, CPU).
    
    Returns:
        torch.device: Dispositivo da utilizzare per training
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")

def detect_hardware() -> Dict[str, Any]:
    """
    Rileva e restituisce informazioni sull'hardware disponibile.
    
    Returns:
        dict: Dizionario con informazioni sull'hardware
    """
    hardware_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "cpu_count": os.cpu_count(),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "ram_total": psutil.virtual_memory().total,
        "ram_available": psutil.virtual_memory().available,
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "device": str(get_device()),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
        "tpu_available": False,  # PyTorch XLA non è facilmente rilevabile
    }
    
    # Aggiungi info sulle GPU
    if hardware_info["gpu_available"]:
        for i in range(hardware_info["gpu_count"]):
            hardware_info["gpu_names"].append(torch.cuda.get_device_name(i))
            
        # Prova a ottenere la memoria disponibile sulla GPU
        try:
            free_memory, total_memory = torch.cuda.mem_get_info(0)
            hardware_info["gpu_memory_total"] = total_memory
            hardware_info["gpu_memory_free"] = free_memory
        except:
            hardware_info["gpu_memory_total"] = 0
            hardware_info["gpu_memory_free"] = 0
    
    return hardware_info

def get_optimal_batch_size(model_size_mb=100, target_device=None):
    """
    Stima una dimensione ottimale del batch in base all'hardware disponibile.
    
    Args:
        model_size_mb (int): Dimensione del modello in MB
        target_device (torch.device, optional): Dispositivo target o None per auto-detect
        
    Returns:
        int: Dimensione del batch consigliata
    """
    device = target_device or get_device()
    
    # Stima una dimensione di batch predefinita in base al dispositivo
    if device.type == 'cuda':
        try:
            # Cerca di ottenere informazioni sulla memoria della GPU
            free_memory, total_memory = torch.cuda.mem_get_info(device.index or 0)
            free_memory_mb = free_memory / (1024 * 1024)
            
            # Usiamo una formula euristica: 20% della memoria libera diviso per la dimensione del modello
            # con un fattore di sicurezza
            batch_size = int((free_memory_mb * 0.2) / model_size_mb)
            
            # Limita il batch size a valori ragionevoli
            return max(min(batch_size, 512), 32)
        except:
            # Se non riusciamo a ottenere informazioni sulla memoria, usiamo valori predefiniti
            return 64
    elif device.type == 'mps':
        # Per Apple Silicon, usiamo un valore più conservativo
        return 32
    else:
        # Per CPU, usiamo un valore basso
        ram_available_mb = psutil.virtual_memory().available / (1024 * 1024)
        
        # Usiamo una formula simile ma più conservativa
        batch_size = int((ram_available_mb * 0.1) / model_size_mb)
        return max(min(batch_size, 128), 16)

def monitor_memory_usage():
    """
    Stampa informazioni sull'utilizzo corrente della memoria.
    """
    print(f"RAM: {psutil.virtual_memory().percent}% in uso")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            try:
                free_memory, total_memory = torch.cuda.mem_get_info(i)
                free_memory_mb = free_memory / (1024 * 1024)
                total_memory_mb = total_memory / (1024 * 1024)
                used_percent = (1 - free_memory / total_memory) * 100
                print(f"GPU {i}: {used_percent:.1f}% in uso ({total_memory_mb - free_memory_mb:.0f}MB / {total_memory_mb:.0f}MB)")
            except:
                print(f"GPU {i}: Impossibile ottenere informazioni sulla memoria")

def set_hardware_optimization():
    """
    Configura le ottimizzazioni hardware per PyTorch.
    """
    # Ottimizzazioni per CUDA
    if torch.cuda.is_available():
        # Abilita ottimizzazioni automatiche
        torch.backends.cudnn.benchmark = True
        
        # Limita memoria pre-allocata (utile in ambienti con più processi)
        # torch.cuda.set_per_process_memory_fraction(0.8)
        
        print("Ottimizzazioni CUDA abilitate")
    
    # Ottimizzazioni per CPU
    if not torch.cuda.is_available():
        # Imposta il numero di thread per operazioni parallele
        torch.set_num_threads(os.cpu_count())
        
        # Imposta il numero di thread per operazioni inter-op
        torch.set_num_interop_threads(max(4, os.cpu_count() // 2))
        
        print(f"Ottimizzazioni CPU abilitate ({os.cpu_count()} threads)")

def enable_mixed_precision(enable=True):
    """
    Abilita o disabilita l'addestramento con precisione mista.
    Funziona solo con GPU compatibili (Tensor Cores).
    
    Args:
        enable (bool): Se abilitare la precisione mista
    
    Returns:
        bool: True se la precisione mista è stata abilitata, False altrimenti
    """
    if not enable:
        return False
        
    # Verifica se CUDA è disponibile
    if not torch.cuda.is_available():
        print("Precisione mista non disponibile: richiede CUDA.")
        return False
        
    # Verifica se il dispositivo supporta la precisione mista
    device_capability = torch.cuda.get_device_capability()
    if device_capability[0] < 7:  # Richiede Volta (SM 7.0) o superiore per Tensor Cores
        print(f"Precisione mista non supportata: la GPU ha compute capability {device_capability[0]}.{device_capability[1]}, richiesto >=7.0")
        return False
    
    # Abilita autocast per precisione mista
    try:
        from torch.cuda.amp import GradScaler
        print("Precisione mista abilitata.")
        return True
    except ImportError:
        print("Precisione mista non disponibile nella versione corrente di PyTorch.")
        return False

# Codice di test
if __name__ == "__main__":
    hw_info = detect_hardware()
    for key, value in hw_info.items():
        print(f"{key}: {value}")
    
    print("\nDispositivo ottimale:", get_device())
    print("Batch size ottimale:", get_optimal_batch_size())
    
    print("\nMonitoraggio memoria:")
    monitor_memory_usage()
    
    print("\nImpostazione ottimizzazioni hardware...")
    set_hardware_optimization() 