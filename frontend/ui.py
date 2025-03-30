#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modulo UI
=======
Implementa l'interfaccia grafica del gioco Snake utilizzando Pygame.

Autore: Federico Baratti
Versione: 1.0
"""

import pygame
import sys
import time
import numpy as np
from pygame.locals import *

# Colori per l'interfaccia
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
BRIGHT_GREEN = (0, 255, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
BACKGROUND = (30, 30, 30)
GRID_COLOR = (50, 50, 50)
FOOD_COLOR = (255, 50, 50)
SNAKE_HEAD_COLOR = (0, 255, 0)
SNAKE_BODY_COLOR = (0, 200, 0)

class GameUI:
    """
    Interfaccia grafica del gioco Snake utilizzando Pygame.
    
    Gestisce:
    - Visualizzazione della griglia di gioco
    - Input dell'utente
    - Rendering del serpente e del cibo
    - Visualizzazione di punteggio e informazioni aggiuntive
    - Controllo modalità autoplay
    """
    
    def __init__(self, game, cell_size=30, speed=10, autoplay_controller=None):
        """
        Inizializza l'interfaccia grafica.
        
        Args:
            game: Istanza della classe SnakeGame
            cell_size (int): Dimensione in pixel di ogni cella della griglia
            speed (int): Velocità iniziale del gioco (1-20)
            autoplay_controller: Controller per la modalità autoplay (opzionale)
        """
        # Inizializza pygame
        pygame.init()
        pygame.font.init()
        
        # Salva i riferimenti
        self.game = game
        self.grid_size = game.get_grid_size()
        self.cell_size = cell_size
        self.speed = min(max(speed, 1), 20)  # Limita la velocità tra 1 e 20
        self.autoplay_controller = autoplay_controller
        
        # Dimensioni della finestra
        sidebar_width = 300
        self.window_width = self.grid_size * self.cell_size + sidebar_width
        self.window_height = self.grid_size * self.cell_size
        
        # Impostazioni UI
        self.score_font = pygame.font.SysFont('Arial', 28, bold=True)
        self.info_font = pygame.font.SysFont('Arial', 20)
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        
        # Crea la finestra
        self.window = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption('Snake con Deep Q-Learning')
        
        # Impostazioni del gioco
        self.is_running = True
        self.is_paused = False
        self.clock = pygame.time.Clock()
        self.frame_rate = 60
        
        # Flag per la modalità autoplay
        self.autoplay_enabled = self.autoplay_controller is not None
        self.toggle_autoplay = False  # Per passare tra modalità manuale e autoplay
        
        # Mappatura azioni direzioni
        self.direction_map = {
            K_UP: 2,     # Su -> gira a sinistra se direzione è destra
            K_DOWN: 1,   # Giù -> gira a destra se direzione è destra
            K_LEFT: 2,   # Sinistra -> gira a sinistra se direzione è giù
            K_RIGHT: 1,  # Destra -> gira a destra se direzione è giù
            K_w: 2,      # W -> gira a sinistra se direzione è destra
            K_s: 1,      # S -> gira a destra se direzione è destra
            K_a: 2,      # A -> gira a sinistra se direzione è giù
            K_d: 1,      # D -> gira a destra se direzione è giù
        }
        
        # Memorizza l'ultima azione e timestamp
        self.last_action = 0  # Inizialmente vai dritto
        self.last_update_time = time.time()
        
        # Memorizza il punteggio più alto
        self.high_score = 0
        
        # Informazioni sulla modalità corrente
        self.mode_info = "Modalità manuale" if not self.autoplay_enabled else "Modalità autoplay"
    
    def toggle_autoplay_mode(self):
        """Attiva/disattiva la modalità autoplay se il controller è disponibile."""
        if self.autoplay_controller:
            self.toggle_autoplay = not self.toggle_autoplay
            self.mode_info = "Modalità autoplay" if self.toggle_autoplay else "Modalità manuale"
    
    def handle_input(self):
        """
        Gestisce l'input dell'utente (tastiera/eventi).
        
        Returns:
            int: Azione da eseguire (0: dritto, 1: destra, 2: sinistra)
        """
        action = self.last_action  # Default: mantieni l'ultima azione
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.is_running = False
                return action
            
            elif event.type == KEYDOWN:
                # Gestione tasti per le azioni
                if event.key in self.direction_map:
                    if not self.toggle_autoplay:  # Solo se non in autoplay
                        action = self.direction_map[event.key]
                
                # Gestione altri tasti
                elif event.key == K_ESCAPE:
                    self.is_running = False
                elif event.key == K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == K_r:
                    self.game.reset()
                    action = 0  # Reset dell'azione
                elif event.key == K_t:
                    self.toggle_autoplay_mode()
                elif event.key in [K_PLUS, K_EQUALS]:
                    self.speed = min(self.speed + 1, 20)
                elif event.key in [K_MINUS, K_UNDERSCORE]:
                    self.speed = max(self.speed - 1, 1)
        
        return action
    
    def get_next_action(self):
        """
        Determina la prossima azione in base alla modalità (manuale/autoplay).
        
        Returns:
            int: Azione da eseguire (0: dritto, 1: destra, 2: sinistra)
        """
        # Gestisci input utente
        action = self.handle_input()
        
        # In modalità autoplay, usa il controller per determinare l'azione
        if self.toggle_autoplay and self.autoplay_controller:
            action = self.autoplay_controller.get_action()
        
        return action
    
    def draw_grid(self):
        """Disegna la griglia di gioco."""
        for x in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.window, 
                GRID_COLOR, 
                (x, 0), 
                (x, self.grid_size * self.cell_size)
            )
        
        for y in range(0, self.grid_size * self.cell_size, self.cell_size):
            pygame.draw.line(
                self.window, 
                GRID_COLOR, 
                (0, y), 
                (self.grid_size * self.cell_size, y)
            )
    
    def draw_food(self):
        """Disegna il cibo."""
        food_position = self.game.get_food_position()
        x, y = food_position
        
        food_rect = pygame.Rect(
            x * self.cell_size, 
            y * self.cell_size,
            self.cell_size, 
            self.cell_size
        )
        
        pygame.draw.rect(self.window, FOOD_COLOR, food_rect)
        
        # Disegna un cerchio all'interno per renderlo più appetibile
        pygame.draw.circle(
            self.window,
            (255, 255, 255),
            (x * self.cell_size + self.cell_size // 2, y * self.cell_size + self.cell_size // 2),
            self.cell_size // 4
        )
    
    def draw_snake(self):
        """Disegna il serpente."""
        snake_positions = self.game.get_snake_positions()
        
        # Disegna prima il corpo
        for i, (x, y) in enumerate(snake_positions):
            if i == 0:
                continue  # Salta la testa per ora
                
            snake_rect = pygame.Rect(
                x * self.cell_size, 
                y * self.cell_size,
                self.cell_size, 
                self.cell_size
            )
            
            pygame.draw.rect(self.window, SNAKE_BODY_COLOR, snake_rect)
            
            # Disegna un bordo più scuro
            pygame.draw.rect(
                self.window, 
                (0, 150, 0), 
                snake_rect, 
                1
            )
        
        # Disegna la testa del serpente (se esiste)
        if snake_positions:
            head_x, head_y = snake_positions[0]
            head_rect = pygame.Rect(
                head_x * self.cell_size, 
                head_y * self.cell_size,
                self.cell_size, 
                self.cell_size
            )
            
            pygame.draw.rect(self.window, SNAKE_HEAD_COLOR, head_rect)
            
            # Disegna gli occhi
            eye_radius = self.cell_size // 8
            eye_offset = self.cell_size // 3
            
            # Posizione degli occhi in base alla direzione
            left_eye = (head_x * self.cell_size + eye_offset, head_y * self.cell_size + eye_offset)
            right_eye = (head_x * self.cell_size + self.cell_size - eye_offset, head_y * self.cell_size + eye_offset)
            
            pygame.draw.circle(self.window, BLACK, left_eye, eye_radius)
            pygame.draw.circle(self.window, BLACK, right_eye, eye_radius)
    
    def draw_sidebar(self):
        """Disegna la barra laterale con informazioni e punteggio."""
        sidebar_rect = pygame.Rect(
            self.grid_size * self.cell_size, 
            0,
            self.window_width - (self.grid_size * self.cell_size), 
            self.window_height
        )
        
        # Sfondo barra laterale
        pygame.draw.rect(self.window, (40, 40, 45), sidebar_rect)
        
        # Titolo
        title_surface = self.title_font.render("Snake Game", True, WHITE)
        self.window.blit(
            title_surface, 
            (self.grid_size * self.cell_size + 20, 20)
        )
        
        # Punteggio corrente
        score_text = f"Punteggio: {self.game.get_score()}"
        score_surface = self.score_font.render(score_text, True, WHITE)
        self.window.blit(
            score_surface, 
            (self.grid_size * self.cell_size + 20, 80)
        )
        
        # Punteggio massimo
        high_score_text = f"Record: {self.high_score}"
        high_score_surface = self.score_font.render(high_score_text, True, WHITE)
        self.window.blit(
            high_score_surface, 
            (self.grid_size * self.cell_size + 20, 120)
        )
        
        # Modalità
        mode_surface = self.info_font.render(self.mode_info, True, BRIGHT_GREEN if self.toggle_autoplay else WHITE)
        self.window.blit(
            mode_surface, 
            (self.grid_size * self.cell_size + 20, 170)
        )
        
        # Velocità
        speed_text = f"Velocità: {self.speed}"
        speed_surface = self.info_font.render(speed_text, True, WHITE)
        self.window.blit(
            speed_surface, 
            (self.grid_size * self.cell_size + 20, 200)
        )
        
        # Comandi
        commands = [
            "Comandi:",
            "↑↓←→ o WASD: Muovi",
            "SPAZIO: Pausa",
            "R: Ricomincia",
            "T: Attiva/disattiva autoplay",
            "+/-: Cambia velocità",
            "ESC: Esci"
        ]
        
        y_offset = 250
        for command in commands:
            command_surface = self.info_font.render(command, True, GRAY if "Comandi:" in command else WHITE)
            self.window.blit(
                command_surface, 
                (self.grid_size * self.cell_size + 20, y_offset)
            )
            y_offset += 30
    
    def update_game(self):
        """
        Aggiorna lo stato del gioco in base all'input e alla modalità.
        """
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        
        # Determina l'intervallo di aggiornamento in base alla velocità
        update_interval = 1.0 / self.speed
        
        if not self.is_paused and time_since_last_update >= update_interval:
            # Determina la prossima azione
            action = self.get_next_action()
            
            # Esegui l'azione nel gioco
            _, _, done, _, _ = self.game.step(action)
            
            # Aggiorna il punteggio massimo
            self.high_score = max(self.high_score, self.game.get_score())
            
            # Memorizza l'ultima azione e aggiorna il timestamp
            self.last_action = action
            self.last_update_time = current_time
            
            # Se il gioco è terminato e non in modalità autoplay, mostra messaggio e reimposta
            if done and not self.toggle_autoplay:
                self.show_game_over()
                self.game.reset()
            # Se in modalità autoplay e terminato, reimposta immediatamente
            elif done and self.toggle_autoplay:
                self.game.reset()
    
    def show_game_over(self):
        """Mostra il messaggio di Game Over."""
        game_over_font = pygame.font.SysFont('Arial', 50, bold=True)
        game_over_surface = game_over_font.render("GAME OVER", True, RED)
        
        # Centra il messaggio
        game_over_rect = game_over_surface.get_rect(
            center=(self.grid_size * self.cell_size // 2, self.window_height // 2)
        )
        
        # Disegna il messaggio
        self.window.blit(game_over_surface, game_over_rect)
        pygame.display.update()
        
        # Pausa per qualche secondo
        pygame.time.wait(2000)
    
    def render(self):
        """Renderizza lo stato corrente del gioco."""
        # Sfondo
        self.window.fill(BACKGROUND)
        
        # Disegna gli elementi di gioco
        self.draw_grid()
        self.draw_food()
        self.draw_snake()
        self.draw_sidebar()
        
        # Aggiorna il display
        pygame.display.update()
    
    def run(self):
        """Esegue il loop principale del gioco."""
        try:
            while self.is_running:
                # Gestione degli input e aggiornamento del gioco
                self.update_game()
                
                # Renderizzazione
                self.render()
                
                # Limita il frame rate
                self.clock.tick(self.frame_rate)
        
        except KeyboardInterrupt:
            print("Gioco interrotto dall'utente.")
        finally:
            # Pulizia e uscita
            pygame.quit()
            sys.exit()


if __name__ == "__main__":
    # Test dell'interfaccia grafica da sola
    from backend.snake_game import SnakeGame
    
    game = SnakeGame(grid_size=20)
    ui = GameUI(game, speed=5)
    ui.run() 