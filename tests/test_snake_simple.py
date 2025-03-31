#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Semplice del Gioco Snake
=============================
Versione estremamente semplificata per testare il gioco Snake

Autore: Claude AI
Versione: 1.0
"""

import pygame
import sys
import time
from backend.snake_game import SnakeGame, Direction

# Colori
NERO = (0, 0, 0)
BIANCO = (255, 255, 255)
VERDE = (0, 255, 0)
ROSSO = (255, 0, 0)
BLU = (0, 0, 255)

def main():
    """Funzione principale per eseguire il test semplice."""
    try:
        # Parametri di gioco
        grid_size = 20
        cell_size = 25
        
        # Inizializza il gioco
        game = SnakeGame(grid_size=grid_size)
        
        # Inizializza pygame
        pygame.init()
        
        # Crea la finestra
        window_width = grid_size * cell_size
        window_height = grid_size * cell_size + 40
        screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Test Snake Semplice")
        
        # Inizializza il font
        font = pygame.font.SysFont('Arial', 20)
        
        # Clock per il controllo degli FPS
        clock = pygame.time.Clock()
        
        # Flag di controllo
        running = True
        
        # Loop principale
        while running:
            # Gestione eventi
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        game.reset()
                    elif event.key == pygame.K_RIGHT:
                        game.step(0)  # Vai dritto
                    elif event.key == pygame.K_DOWN:
                        game.step(1)  # Gira a destra
                    elif event.key == pygame.K_LEFT:
                        game.step(2)  # Gira a sinistra
                    elif event.key == pygame.K_UP:
                        game.step(3)  # Gira indietro
            
            # Pulisci lo schermo
            screen.fill(NERO)
            
            # Disegna la griglia
            for y in range(grid_size):
                for x in range(grid_size):
                    rect = pygame.Rect(
                        x * cell_size,
                        y * cell_size,
                        cell_size,
                        cell_size
                    )
                    pygame.draw.rect(screen, BIANCO, rect, 1)
            
            # Disegna il serpente
            for i, segment in enumerate(game.get_snake_positions()):
                x, y = segment
                rect = pygame.Rect(
                    x * cell_size,
                    y * cell_size,
                    cell_size,
                    cell_size
                )
                color = VERDE if i > 0 else BLU  # Testa blu, corpo verde
                pygame.draw.rect(screen, color, rect)
            
            # Disegna il cibo
            food_x, food_y = game.get_food_position()
            rect = pygame.Rect(
                food_x * cell_size,
                food_y * cell_size,
                cell_size,
                cell_size
            )
            pygame.draw.rect(screen, ROSSO, rect)
            
            # Disegna il punteggio
            score_text = font.render(f"Punteggio: {game.get_score()}", True, BIANCO)
            screen.blit(score_text, (10, grid_size * cell_size + 10))
            
            # Se il gioco è terminato, mostra il messaggio di game over
            if game.done:
                game_over_text = font.render("GAME OVER - Premi R per riavviare", True, ROSSO)
                text_rect = game_over_text.get_rect(center=(window_width // 2, window_height // 2))
                screen.blit(game_over_text, text_rect)
            
            # Aggiorna lo schermo
            pygame.display.flip()
            
            # Limita gli FPS
            clock.tick(10)
        
        # Chiudi pygame
        pygame.quit()
        
    except Exception as e:
        print(f"Errore: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()

if __name__ == "__main__":
    try:
        print("Test Semplice Snake Game")
        print("------------------------")
        print("Controlli:")
        print("  - Frecce: Controllano la direzione del serpente")
        print("  - R: Riavvia")
        print("  - ESC: Esci")
        
        main()
    except Exception as e:
        print(f"Errore critico: {e}")
        import traceback
        traceback.print_exc() 