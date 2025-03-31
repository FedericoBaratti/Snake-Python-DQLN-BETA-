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
import os
import glob

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
    - Selezione del modello da caricare
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
            K_b: 3,      # B -> gira indietro (180 gradi)
            K_BACKSPACE: 3  # Backspace -> gira indietro (180 gradi)
        }
        
        # Memorizza l'ultima azione e timestamp
        self.last_action = 0  # Inizialmente vai dritto
        self.last_update_time = time.time()
        
        # Memorizza il punteggio più alto
        self.high_score = 0
        
        # Informazioni sulla modalità corrente
        self.mode_info = "Modalità manuale" if not self.autoplay_enabled else "Modalità autoplay"
        
        # Lista dei checkpoint disponibili
        self.available_checkpoints = []
        self.current_checkpoint_idx = 0
        self.refresh_checkpoint_list()
        
        # Flag per mostrare la finestra di selezione del modello
        self.show_model_selector = False
    
    def refresh_checkpoint_list(self):
        """Aggiorna la lista dei checkpoint disponibili nel sistema."""
        checkpoint_dir = os.path.join("training", "checkpoints")
        if os.path.exists(checkpoint_dir):
            # Trova tutti i file .pt (checkpoint PyTorch)
            self.available_checkpoints = sorted(
                [f for f in glob.glob(os.path.join(checkpoint_dir, "*.pt")) 
                 if "_memory" not in f],  # Esclude i file di memoria
                key=os.path.getmtime,  # Ordina per data di modifica
                reverse=True  # Più recenti prima
            )
            
            # Aggiungi anche l'opzione "Nessun checkpoint" all'inizio
            self.available_checkpoints = ["Nessun checkpoint"] + self.available_checkpoints
            
            # Resetta l'indice del checkpoint corrente
            self.current_checkpoint_idx = 0
        else:
            # Se la directory non esiste, crea una lista con solo "Nessun checkpoint"
            self.available_checkpoints = ["Nessun checkpoint"]
            self.current_checkpoint_idx = 0
    
    def toggle_model_selector(self):
        """Attiva/disattiva la finestra di selezione del modello."""
        self.show_model_selector = not self.show_model_selector
        if self.show_model_selector:
            # Aggiorna la lista dei checkpoint quando apri il selettore
            self.refresh_checkpoint_list()
    
    def change_selected_model(self, direction):
        """
        Cambia il modello selezionato nella lista.
        
        Args:
            direction (int): 1 per avanzare, -1 per tornare indietro
        """
        if self.available_checkpoints:
            self.current_checkpoint_idx = (self.current_checkpoint_idx + direction) % len(self.available_checkpoints)
    
    def get_current_model_path(self):
        """
        Restituisce il percorso del modello attualmente selezionato.
        
        Returns:
            str: Percorso del checkpoint o None se "Nessun checkpoint" è selezionato
        """
        if not self.available_checkpoints or self.current_checkpoint_idx == 0:
            return None
        return self.available_checkpoints[self.current_checkpoint_idx]
    
    def load_selected_model(self):
        """
        Carica il modello selezionato nell'autoplay controller.
        
        Returns:
            bool: True se il caricamento è avvenuto con successo, False altrimenti
        """
        if not self.autoplay_controller:
            return False
            
        checkpoint_path = self.get_current_model_path()
        if checkpoint_path and os.path.exists(checkpoint_path):
            # Ottieni la complessità del modello dal nome del file
            model_complexity = "base"  # Default
            for complexity in ["base", "avanzato", "complesso", "perfetto"]:
                if complexity in os.path.basename(checkpoint_path):
                    model_complexity = complexity
                    break
            
            try:
                # Reinizializza il controller autoplay con il nuovo modello
                from autoplay.autoplay import AutoplayController
                from backend.environment import SnakeEnv
                
                # Ricrea l'ambiente
                env = SnakeEnv(self.game)
                
                # Crea un nuovo controller con il checkpoint selezionato
                self.autoplay_controller = AutoplayController(
                    env=env,
                    model_complexity=model_complexity,
                    checkpoint_path=checkpoint_path
                )
                
                # Aggiorna le informazioni di modalità
                model_name = os.path.basename(checkpoint_path).replace('.pt', '')
                self.mode_info = f"Autoplay: {model_name}"
                
                # Attiva la modalità autoplay
                self.toggle_autoplay = True
                
                # Chiudi il selettore
                self.show_model_selector = False
                
                return True
            except Exception as e:
                print(f"Errore nel caricamento del modello: {e}")
                return False
        return False
    
    def toggle_autoplay_mode(self):
        """Attiva/disattiva la modalità autoplay se il controller è disponibile."""
        if self.autoplay_controller:
            self.toggle_autoplay = not self.toggle_autoplay
            self.mode_info = "Modalità autoplay" if self.toggle_autoplay else "Modalità manuale"
    
    def handle_input(self):
        """
        Gestisce l'input dell'utente (tastiera/eventi).
        
        Returns:
            int: Azione da eseguire (0: dritto, 1: destra, 2: sinistra, 3: indietro)
        """
        action = self.last_action  # Default: mantieni l'ultima azione
        
        for event in pygame.event.get():
            if event.type == QUIT:
                self.is_running = False
                return action
            
            elif event.type == KEYDOWN:
                # Se il selettore di modelli è attivo, gestisci i suoi controlli
                if self.show_model_selector:
                    if event.key == K_UP:
                        self.change_selected_model(-1)
                    elif event.key == K_DOWN:
                        self.change_selected_model(1)
                    elif event.key == K_RETURN:
                        self.load_selected_model()
                    elif event.key == K_ESCAPE:
                        self.show_model_selector = False
                    return action  # Non processare altri input durante la selezione
                
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
                elif event.key == K_m:
                    self.toggle_model_selector()
                elif event.key in [K_PLUS, K_EQUALS]:
                    self.speed = min(self.speed + 1, 20)
                elif event.key in [K_MINUS, K_UNDERSCORE]:
                    self.speed = max(self.speed - 1, 1)
        
        return action
    
    def get_next_action(self):
        """
        Determina la prossima azione in base alla modalità (manuale/autoplay).
        
        Returns:
            int: Azione da eseguire (0: dritto, 1: destra, 2: sinistra, 3: indietro)
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
            "M: Selezione modello",
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
    
    def draw_model_selector(self):
        """Disegna la finestra di selezione del modello."""
        if not self.show_model_selector:
            return
        
        # Dimensioni della finestra di selezione
        selector_width = min(self.window_width - 100, 600)
        selector_height = min(self.window_height - 100, 400)
        
        # Posizione centrale della finestra
        selector_x = (self.window_width - selector_width) // 2
        selector_y = (self.window_height - selector_height) // 2
        
        # Disegna lo sfondo semi-trasparente
        overlay = pygame.Surface((self.window_width, self.window_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))  # Nero semi-trasparente
        self.window.blit(overlay, (0, 0))
        
        # Disegna il rettangolo della finestra
        selector_rect = pygame.Rect(selector_x, selector_y, selector_width, selector_height)
        pygame.draw.rect(self.window, (60, 60, 70), selector_rect)
        pygame.draw.rect(self.window, WHITE, selector_rect, 2)  # Bordo
        
        # Titolo
        title_surface = self.title_font.render("Seleziona Modello", True, WHITE)
        title_rect = title_surface.get_rect(center=(selector_x + selector_width // 2, selector_y + 30))
        self.window.blit(title_surface, title_rect)
        
        # Istruzioni
        instructions = self.info_font.render("Usa ↑↓ per navigare, INVIO per selezionare, ESC per annullare", True, GRAY)
        instructions_rect = instructions.get_rect(center=(selector_x + selector_width // 2, selector_y + selector_height - 30))
        self.window.blit(instructions, instructions_rect)
        
        # Lista dei checkpoint
        if not self.available_checkpoints:
            msg = self.info_font.render("Nessun checkpoint disponibile", True, RED)
            msg_rect = msg.get_rect(center=(selector_x + selector_width // 2, selector_y + selector_height // 2))
            self.window.blit(msg, msg_rect)
        else:
            # Calcola quanti elementi possiamo mostrare
            item_height = 40
            visible_items = min(8, len(self.available_checkpoints))
            
            # Calcola l'offset per centrare la lista
            list_y = selector_y + 80
            
            # Determina l'intervallo di indici da visualizzare
            start_idx = max(0, self.current_checkpoint_idx - (visible_items // 2))
            end_idx = min(start_idx + visible_items, len(self.available_checkpoints))
            
            # Ajusta l'inizio se non abbiamo abbastanza elementi alla fine
            if end_idx - start_idx < visible_items:
                start_idx = max(0, end_idx - visible_items)
            
            # Disegna gli elementi visibili
            for i in range(start_idx, end_idx):
                checkpoint = self.available_checkpoints[i]
                
                # Determina il nome da visualizzare
                if checkpoint == "Nessun checkpoint":
                    display_name = checkpoint
                else:
                    display_name = os.path.basename(checkpoint)
                
                # Evidenzia l'elemento selezionato
                text_color = BRIGHT_GREEN if i == self.current_checkpoint_idx else WHITE
                
                # Disegna lo sfondo per l'elemento selezionato
                item_y = list_y + (i - start_idx) * item_height
                if i == self.current_checkpoint_idx:
                    item_rect = pygame.Rect(selector_x + 10, item_y - 5, selector_width - 20, item_height)
                    pygame.draw.rect(self.window, (80, 80, 100), item_rect)
                    pygame.draw.rect(self.window, (100, 100, 140), item_rect, 1)
                
                # Disegna il testo
                item_surface = self.info_font.render(display_name, True, text_color)
                self.window.blit(item_surface, (selector_x + 20, item_y))
    
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
        
        # Disegna la finestra di selezione del modello se attiva
        self.draw_model_selector()
        
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