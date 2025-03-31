#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package Agents
==============
Raccolta di algoritmi di apprendimento per rinforzo.

Autore: Federico Baratti
Versione: 2.0
"""

from .dqn import DQNAgent

# Dizionario degli agenti disponibili
AGENTS = {
    'dqn': DQNAgent,
}

def get_agent(agent_type: str, **kwargs):
    """
    Factory function per creare un agente.
    
    Args:
        agent_type (str): Tipo di agente ('dqn', etc.)
        **kwargs: Parametri specifici per l'agente
        
    Returns:
        Agent: Istanza dell'agente richiesto
        
    Raises:
        ValueError: Se il tipo di agente non è supportato
    """
    agent_type = agent_type.lower()
    if agent_type not in AGENTS:
        raise ValueError(f"Tipo di agente non supportato: {agent_type}. "
                         f"Opzioni disponibili: {list(AGENTS.keys())}")
    
    return AGENTS[agent_type](**kwargs)

__all__ = ['DQNAgent', 'get_agent', 'AGENTS'] 