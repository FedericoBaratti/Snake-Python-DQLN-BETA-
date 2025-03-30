#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Utility
============
Funzioni helper per il gioco Snake e l'ambiente DQN.

Autore: Federico Baratti
Versione: 1.0
"""

import numpy as np
import time
import os
import torch
import random
from pathlib import Path

def set_seed(seed):
    """
    Imposta tutti i seed per riproducibilità.
    
    Args:
        seed (int): Seed da impostare
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_device():
    """
    Determina il dispositivo migliore disponibile (CUDA, MPS, CPU).
    
    Returns:
        torch.device: Dispositivo da utilizzare per training
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")

def detect_hardware():
    """
    Rileva e restituisce informazioni sull'hardware disponibile.
    
    Returns:
        dict: Dizionario con informazioni sull'hardware
    """
    hardware_info = {
        "cpu_count": os.cpu_count(),
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_names": [],
        "device": str(get_device()),
        "tpu_available": False,  # PyTorch XLA non è facilmente rilevabile
    }
    
    # Aggiungi info sulle GPU
    if hardware_info["gpu_available"]:
        for i in range(hardware_info["gpu_count"]):
            hardware_info["gpu_names"].append(torch.cuda.get_device_name(i))
    
    return hardware_info

def create_checkpoint_dir(path="training/checkpoints"):
    """
    Crea la directory per i checkpoint se non esiste.
    
    Args:
        path (str): Percorso della directory
        
    Returns:
        str: Percorso della directory creata
    """
    checkpoint_dir = Path(path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir)

def calculate_epsilon(start_eps, end_eps, current_step, final_step):
    """
    Calcola epsilon per l'esplorazione usando un decadimento lineare.
    
    Args:
        start_eps (float): Valore iniziale di epsilon
        end_eps (float): Valore finale di epsilon
        current_step (int): Step corrente
        final_step (int): Step finale per il decadimento
        
    Returns:
        float: Valore corrente di epsilon
    """
    if current_step >= final_step:
        return end_eps
    return end_eps + (start_eps - end_eps) * (1 - current_step / final_step)

def human_readable_size(size_bytes):
    """
    Converte dimensioni in byte in formato leggibile.
    
    Args:
        size_bytes (int): Dimensione in byte
        
    Returns:
        str: Dimensione in formato leggibile
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB")
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = np.power(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def timeit(func):
    """
    Decoratore per misurare il tempo di esecuzione di una funzione.
    
    Args:
        func (callable): Funzione da decorare
        
    Returns:
        callable: Funzione decorata
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Funzione {func.__name__} eseguita in {end_time - start_time:.4f} secondi")
        return result
    return wrapper

def moving_average(values, window):
    """
    Calcola la media mobile di un array di valori.
    
    Args:
        values (list): Lista di valori
        window (int): Dimensione della finestra
        
    Returns:
        np.ndarray: Media mobile
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def preprocess_frame(frame, target_size=(84, 84)):
    """
    Preprocessa un frame per l'input alla rete neurale.
    
    Args:
        frame (np.ndarray): Frame da preprocessare
        target_size (tuple): Dimensione target
        
    Returns:
        np.ndarray: Frame preprocessato
    """
    # Converti in scala di grigi se a colori
    if len(frame.shape) == 3:
        frame = np.mean(frame, axis=2).astype(np.float32)
    
    # Ridimensiona se necessario
    if frame.shape != target_size:
        # Implementare ridimensionamento se necessario
        pass
    
    # Normalizza i valori
    frame = frame / 255.0
    
    return frame 