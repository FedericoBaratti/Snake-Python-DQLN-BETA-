#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Interattivo del Gioco Snake (Versione 2)
=============================================
Script per testare interattivamente il gioco Snake con movimento automatico

Autore: Baratti Federico
Versione: 1.0
"""

import pygame
import sys
import numpy as np
import time
from backend.snake_game import SnakeGame, Direction

# Colori
NERO = (0, 0, 0)
BIANCO = (255, 255, 255)
VERDE = (0, 255, 0)
VERDE_CHIARO = (150, 255, 150)
ROSSO = (255, 0, 0)
BLU = (0, 0, 255)
GRIGIO = (100, 100, 100)

class SnakeGameGUI:
    """Classe per la visualizzazione grafica del gioco Snake."""
    
    def __init__(self, grid_size=20, cell_size=20):
        """
        Inizializza la GUI del gioco Snake.
        
        Args:
            grid_size (int): Dimensione della griglia di gioco
            cell_size (int): Dimensione di una cella in pixel
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.game = SnakeGame(grid_size=grid_size)
        
        # Inizializza pygame
        pygame.init()
        
        # Calcola la dimensione della finestra
        window_width = grid_size * cell_size + 200  # Spazio extra per info
        window_height = grid_size * cell_size + 60  # Spazio extra per il punteggio
        
        # Crea la finestra
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Test Interattivo Snake Game v2")
        
        # Inizializza il timer
        self.clock = pygame.time.Clock()
        
        # Inizializza il font
        self.font = pygame.font.SysFont('Arial', 20)
        self.small_font = pygame.font.SysFont('Arial', 14)
        
        # Flag per il gioco
        self.running = True
        self.paused = False
        self.auto_mode = False
        self.fps = 10
        self.game_over_time = None
        
        print("GUI inizializzata con successo")
    
    def handle_events(self):
        """Gestisce gli eventi di pygame."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_p:
                    self.paused = not self.paused
                    print(f"Gioco {'in pausa' if self.paused else 'ripreso'}")
                elif event.key == pygame.K_r:
                    print("Reset del gioco")
                    self.game.reset()
                    self.game_over_time = None
                elif event.key == pygame.K_a:
                    self.auto_mode = not self.auto_mode
                    print(f"Modalità auto {'attivata' if self.auto_mode else 'disattivata'}")
                elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                    self.fps = min(30, self.fps + 1)
                    print(f"Velocità: {self.fps} FPS")
                elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                    self.fps = max(1, self.fps - 1)
                    print(f"Velocità: {self.fps} FPS")
                elif not self.paused and not self.auto_mode and self.game_over_time is None:
                    # Gestione delle direzioni
                    if event.key == pygame.K_RIGHT:
                        self.step(0)  # Vai dritto
                    elif event.key == pygame.K_DOWN:
                        self.step(1)  # Gira a destra
                    elif event.key == pygame.K_LEFT:
                        self.step(2)  # Gira a sinistra
                    elif event.key == pygame.K_UP:
                        self.step(3)  # Vai indietro
    
    def step(self, action):
        """
        Esegue un'azione nel gioco.
        
        Args:
            action (int): Azione da eseguire
        """
        try:
            state, reward, done, info = self.game.step(action)
            
            if done:
                print(f"Game Over! Punteggio: {self.game.get_score()}")
                self.game_over_time = time.time()
                
            return done
        except Exception as e:
            print(f"Errore durante l'esecuzione dell'azione: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def auto_play(self):
        """Implementa una logica semplice per il gioco automatico."""
        if self.paused or self.game_over_time is not None:
            return
        
        # Ottieni posizione della testa e del cibo
        head_x, head_y = self.game.snake[0]
        food_x, food_y = self.game.food
        
        # Direzione corrente
        current_dir = self.game.direction
        
        # Decisione di base: cerca di muoversi verso il cibo
        if current_dir == Direction.RIGHT:
            if food_y < head_y:
                action = 2  # Gira a sinistra (su)
            elif food_y > head_y:
                action = 1  # Gira a destra (giù)
            elif food_x > head_x:
                action = 0  # Vai dritto
            else:
                action = 1  # Gira a destra (evita di tornare indietro)
        elif current_dir == Direction.LEFT:
            if food_y < head_y:
                action = 1  # Gira a destra (su)
            elif food_y > head_y:
                action = 2  # Gira a sinistra (giù)
            elif food_x < head_x:
                action = 0  # Vai dritto
            else:
                action = 1  # Gira a destra (evita di tornare indietro)
        elif current_dir == Direction.UP:
            if food_x < head_x:
                action = 2  # Gira a sinistra (sinistra)
            elif food_x > head_x:
                action = 1  # Gira a destra (destra)
            elif food_y < head_y:
                action = 0  # Vai dritto
            else:
                action = 1  # Gira a destra (evita di tornare indietro)
        elif current_dir == Direction.DOWN:
            if food_x < head_x:
                action = 1  # Gira a destra (sinistra)
            elif food_x > head_x:
                action = 2  # Gira a sinistra (destra)
            elif food_y > head_y:
                action = 0  # Vai dritto
            else:
                action = 1  # Gira a destra (evita di tornare indietro)
        
        # Evita collisioni prevedendo la prossima posizione
        # (Logica semplificata, può ancora portare a collisioni)
        
        # Esegui l'azione
        self.step(action)
    
    def draw(self):
        """Disegna lo stato del gioco."""
        # Pulisci lo schermo
        self.screen.fill(NERO)
        
        # Disegna la griglia
        grid_width = self.grid_size * self.cell_size
        grid_height = self.grid_size * self.cell_size
        grid_rect = pygame.Rect(0, 0, grid_width, grid_height)
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                rect = pygame.Rect(
                    x * self.cell_size,
                    y * self.cell_size,
                    self.cell_size,
                    self.cell_size
                )
                pygame.draw.rect(self.screen, GRIGIO, rect, 1)
        
        # Disegna il serpente
        for i, segment in enumerate(self.game.get_snake_positions()):
            x, y = segment
            rect = pygame.Rect(
                x * self.cell_size,
                y * self.cell_size,
                self.cell_size,
                self.cell_size
            )
            color = VERDE if i > 0 else BLU  # Testa blu, corpo verde
            pygame.draw.rect(self.screen, color, rect)
            
            # Disegna un cerchio più piccolo all'interno per un effetto migliore
            inner_rect = pygame.Rect(
                x * self.cell_size + 2,
                y * self.cell_size + 2,
                self.cell_size - 4,
                self.cell_size - 4
            )
            inner_color = VERDE_CHIARO if i > 0 else BLU
            pygame.draw.rect(self.screen, inner_color, inner_rect)
        
        # Disegna il cibo
        food_x, food_y = self.game.get_food_position()
        rect = pygame.Rect(
            food_x * self.cell_size,
            food_y * self.cell_size,
            self.cell_size,
            self.cell_size
        )
        pygame.draw.rect(self.screen, ROSSO, rect)
        
        # Area informazioni
        info_rect = pygame.Rect(grid_width + 10, 0, 190, grid_height)
        pygame.draw.rect(self.screen, GRIGIO, info_rect, 2)
        
        # Disegna il punteggio e altre informazioni
        texts = [
            f"Punteggio: {self.game.get_score()}",
            f"Dimensione: {len(self.game.snake)}",
            f"FPS: {self.fps}",
            f"Modalità: {'Auto' if self.auto_mode else 'Manuale'}",
            "",
            "Controlli:",
            "Frecce: Movimento",
            "P: Pausa/Riprendi",
            "R: Reset",
            "A: Auto/Manuale",
            "+/-: Velocità",
            "ESC: Esci"
        ]
        
        for i, text in enumerate(texts):
            text_surf = self.small_font.render(text, True, BIANCO)
            self.screen.blit(text_surf, (grid_width + 20, 20 + i * 22))
        
        # Disegna il punteggio principale in basso
        score_text = self.font.render(f"Punteggio: {self.game.get_score()}", True, BIANCO)
        self.screen.blit(score_text, (10, grid_height + 20))
        
        # Disegna lo stato di pausa se il gioco è in pausa
        if self.paused:
            pause_text = self.font.render("PAUSA", True, BIANCO)
            text_rect = pause_text.get_rect(center=(grid_width // 2, grid_height // 2))
            self.screen.blit(pause_text, text_rect)
        
        # Disegna il messaggio di game over
        if self.game_over_time is not None:
            game_over_text = self.font.render("GAME OVER", True, ROSSO)
            text_rect = game_over_text.get_rect(center=(grid_width // 2, grid_height // 2 - 20))
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = self.small_font.render("Premi R per ricominciare", True, BIANCO)
            restart_rect = restart_text.get_rect(center=(grid_width // 2, grid_height // 2 + 20))
            self.screen.blit(restart_text, restart_rect)
            
            # Reset automatico dopo 3 secondi
            if time.time() - self.game_over_time > 3:
                self.game.reset()
                self.game_over_time = None
        
        # Aggiorna lo schermo
        pygame.display.flip()
    
    def run(self):
        """Esegue il loop principale del gioco."""
        auto_step_time = 0
        
        print("Avvio del loop principale...")
        
        try:
            while self.running:
                self.handle_events()
                
                # Modalità automatica
                if self.auto_mode and not self.paused and self.game_over_time is None:
                    current_time = time.time()
                    if current_time - auto_step_time > 1.0/self.fps:
                        self.auto_play()
                        auto_step_time = current_time
                
                self.draw()
                self.clock.tick(max(30, self.fps * 2))  # Mantieni l'interfaccia reattiva
            
            pygame.quit()
            print("Pygame terminato con successo")
            sys.exit()
        except Exception as e:
            print(f"ERRORE nel loop principale: {e}")
            import traceback
            traceback.print_exc()
            pygame.quit()
            sys.exit(1)

if __name__ == "__main__":
    try:
        print("Test Interattivo Snake Game v2")
        print("------------------------------")
        print("Controlli:")
        print("  - Frecce: Controllano la direzione del serpente")
        print("  - P: Pausa/Riprendi")
        print("  - R: Riavvia")
        print("  - A: Modalità Auto/Manuale")
        print("  - +/-: Aumenta/Diminuisci velocità")
        print("  - ESC: Esci")
        
        # Avvia il gioco
        print("Avvio del gioco...")
        game_gui = SnakeGameGUI(grid_size=20, cell_size=25)
        print("Istanza GUI creata con successo.")
        game_gui.run()
    except Exception as e:
        print(f"ERRORE FATALE: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 