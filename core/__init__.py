#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pacchetto Core
==========
Componenti principali del gioco Snake e dell'ambiente Gym.

Autore: Federico Baratti
Versione: 2.0
"""

# Importa classi principali per renderle disponibili direttamente dal pacchetto
from core.snake_game import (
    SnakeGame,
    Direction,
    RewardSystem
)

from core.environment import (
    SnakeEnv
)

# Versione del pacchetto
__version__ = "2.0.0" 