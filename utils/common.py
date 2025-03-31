#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo Utility Comuni
==================
Funzioni di utilità generali per il progetto.

Autore: Federico Baratti
Versione: 2.0
"""

import numpy as np
import time
import os
import torch
import random
from pathlib import Path
from typing import List, Any, Callable

def set_seed(seed: int):
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

def create_checkpoint_dir(path: str = "training/checkpoints") -> str:
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

def calculate_epsilon(start_eps: float, end_eps: float, current_step: int, final_step: int) -> float:
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

def human_readable_size(size_bytes: int) -> str:
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

def timeit(func: Callable) -> Callable:
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

def moving_average(values: List[float], window: int) -> np.ndarray:
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

def preprocess_frame(frame: np.ndarray, target_size: tuple = (84, 84)) -> np.ndarray:
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
        from PIL import Image
        import numpy as np
        
        # Converti a PIL Image
        img = Image.fromarray(frame)
        
        # Ridimensiona
        img = img.resize(target_size, Image.BILINEAR)
        
        # Riconverti in numpy array
        frame = np.array(img).astype(np.float32)
    
    # Normalizza i valori
    frame = frame / 255.0
    
    return frame

def ensure_dir_exists(path: str) -> str:
    """
    Assicura che la directory esista, creandola se necessario.
    
    Args:
        path (str): Percorso della directory
        
    Returns:
        str: Percorso della directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return str(dir_path)

def get_latest_file(directory: str, pattern: str = "*") -> str:
    """
    Ottiene il file più recente in una directory.
    
    Args:
        directory (str): Directory da esaminare
        pattern (str): Pattern per filtrare i file
        
    Returns:
        str: Percorso del file più recente o None se non ce ne sono
    """
    import glob
    
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
        
    return max(files, key=os.path.getmtime)

def extract_model_info_from_filename(filepath: str) -> dict:
    """
    Estrae informazioni sul modello dal nome del file.
    
    Args:
        filepath (str): Percorso del file
        
    Returns:
        dict: Informazioni estratte
    """
    filename = os.path.basename(filepath)
    
    # Informazioni predefinite
    info = {
        "complexity": "base",
        "version": None,
        "timestamp": None,
        "score": None
    }
    
    # Estrai complessità
    if "base" in filename:
        info["complexity"] = "base"
    elif "avanzato" in filename:
        info["complexity"] = "avanzato"
    elif "complesso" in filename:
        info["complexity"] = "complesso"
    elif "perfetto" in filename:
        info["complexity"] = "perfetto"
    
    # Estrai versione se presente (formato v1, v2, ecc.)
    import re
    version_match = re.search(r'v(\d+)', filename)
    if version_match:
        info["version"] = int(version_match.group(1))
    
    # Estrai score se presente (formato s100, s200, ecc.)
    score_match = re.search(r's(\d+)', filename)
    if score_match:
        info["score"] = int(score_match.group(1))
    
    return info

# Codice di test
if __name__ == "__main__":
    # Test delle funzioni
    print("Test set_seed(42)")
    set_seed(42)
    
    print("\nTest create_checkpoint_dir()")
    checkpoint_dir = create_checkpoint_dir("test_checkpoints")
    print(f"Directory creata: {checkpoint_dir}")
    
    print("\nTest calculate_epsilon()")
    print(f"Epsilon: {calculate_epsilon(1.0, 0.01, 500, 1000)}")
    
    print("\nTest human_readable_size()")
    print(f"1024 bytes = {human_readable_size(1024)}")
    print(f"1048576 bytes = {human_readable_size(1048576)}")
    
    print("\nTest moving_average()")
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Media mobile (window=3): {moving_average(values, 3)}")
    
    print("\nTest ensure_dir_exists()")
    test_dir = ensure_dir_exists("test_dir")
    print(f"Directory assicurata: {test_dir}")
    
    print("\nTest extract_model_info_from_filename()")
    info = extract_model_info_from_filename("dqn_avanzato_v2_s150.pt")
    print(f"Informazioni estratte: {info}")
    
    # Pulizia
    import shutil
    if os.path.exists("test_checkpoints"):
        shutil.rmtree("test_checkpoints")
    if os.path.exists("test_dir"):
        shutil.rmtree("test_dir") 