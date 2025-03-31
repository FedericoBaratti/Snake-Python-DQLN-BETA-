#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pacchetto Utility
=============
Funzioni e classi di utilità per il progetto Snake con DQN.

Autore: Federico Baratti
Versione: 2.0
"""

# Importa funzioni e classi principali per renderle disponibili direttamente dal pacchetto
from utils.common import (
    set_seed, 
    create_checkpoint_dir, 
    calculate_epsilon, 
    human_readable_size, 
    timeit,
    moving_average, 
    preprocess_frame,
    ensure_dir_exists,
    get_latest_file,
    extract_model_info_from_filename
)

from utils.hardware_utils import (
    get_device,
    detect_hardware,
    get_optimal_batch_size,
    monitor_memory_usage,
    set_hardware_optimization
)

from utils.config import (
    get_config_manager,
    get_model_config,
    get_environment_config,
    get_ui_config,
    save_config,
    ConfigManager
)

# Esponi le versioni
__version__ = "2.0.0" 