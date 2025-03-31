#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Esegui Test
===========
Script per eseguire tutti i test del progetto.

Autore: Federico Baratti
Versione: 2.0
"""

import os
import sys
import unittest
import argparse

# Aggiungi il percorso principale al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    """Analizza gli argomenti da linea di comando."""
    parser = argparse.ArgumentParser(description='Esegui i test del progetto Snake-Python-DQLN')
    
    parser.add_argument('--test-type', type=str, choices=['all', 'dqn', 'integration'], 
                        default='all', help='Tipo di test da eseguire')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Mostra output dettagliato')
    
    return parser.parse_args()

def main():
    """Funzione principale."""
    args = parse_args()
    
    # Determina i test da eseguire
    if args.test_type == 'all':
        test_files = ['test_dqn.py', 'test_snake_integration.py']
    elif args.test_type == 'dqn':
        test_files = ['test_dqn.py']
    elif args.test_type == 'integration':
        test_files = ['test_snake_integration.py']
    
    # Stampa informazioni sui test
    print(f"\nEseguo i test: {', '.join(test_files)}")
    print("=" * 50)
    
    # Trova tutti i test
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Aggiungi i test selezionati
    for test_file in test_files:
        test_module = __import__(test_file[:-3])
        suite.addTests(loader.loadTestsFromModule(test_module))
    
    # Esegui i test
    verbosity = 2 if args.verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    # Stampa un riepilogo
    print("\nRiepilogo dei test:")
    print(f"Test eseguiti: {result.testsRun}")
    print(f"Errori: {len(result.errors)}")
    print(f"Fallimenti: {len(result.failures)}")
    
    # Restituisci un codice di errore se ci sono stati fallimenti
    if result.errors or result.failures:
        sys.exit(1)

if __name__ == "__main__":
    main() 