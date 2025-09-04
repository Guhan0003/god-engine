# main.py
import pygame
import sys
from engine.board import Board
from engine.search import find_best_move
from gui.gui import GUI

class Game:
    """
    Manages the main game loop, game state, and user input.
    """
    def __init__(self, screen, player_color='w', ai_depth=3):
        self.screen = screen
        self.board = Board()
        self.gui = GUI()
        self.player_color = player_color
        self.ai_depth = ai_depth
        
        self.selected_sq = None
        self.legal_moves_for_selected = []

    def run(self):
        """The main game loop with the correct update/draw order."""
        clock = pygame.time.Clock()
        running = True
        while running:
            # --- 1. Event Handling ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if self.board.side_to_move == self.player_color and not self._is_game_over():
                        self._handle_click(event.pos)

            # --- 2. Game Logic (AI Move) ---
            if self.board.side_to_move != self.player_color and not self._is_game_over():
                pygame.time.wait(100) # Small delay to make AI move visible
                self._make_ai_move()

            # --- 3. Drawing ---
            self.gui.draw_gamestate(self.screen, self.board, self.selected_sq, self.legal_moves_for_selected)
            pygame.display.flip()
            
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

    def _handle_click(self, pos):
        """Handles a mouse click, either selecting a piece or making a move."""
        clicked_sq = self._pos_to_square(pos)
        
        if self.selected_sq is not None and clicked_sq in self.legal_moves_for_selected:
            move = next((m for m in self.board.generate_legal_moves(self.player_color) 
                         if m.from_sq == self.selected_sq and m.to_sq == clicked_sq), None)
            if move:
                print(f"Player move: {move}") 
                self.board.make_move(move)
                self.selected_sq = None
                self.legal_moves_for_selected = []
        else:
            piece = self.board.get_piece(clicked_sq)
            is_players_piece = piece != '.' and (piece.isupper() if self.player_color == 'w' else piece.islower())
            
            if is_players_piece:
                self.selected_sq = clicked_sq
                legal_moves = self.board.generate_legal_moves(self.player_color)
                self.legal_moves_for_selected = [m.to_sq for m in legal_moves if m.from_sq == clicked_sq]
            else:
                self.selected_sq = None
                self.legal_moves_for_selected = []

    def _make_ai_move(self):
        """Finds and makes the best move for the AI using the search algorithm."""
        print("AI is thinking...")
        score, ai_move = find_best_move(self.board, self.ai_depth)
        
        if ai_move:
            print(f"AI move: {ai_move}")
            print(f"AI plays move. Eval: {score/100.0}")
            self.board.make_move(ai_move)
        else:
            print("AI has no moves. Game over.")

    def _pos_to_square(self, pos):
        """Converts a Pygame screen position to a board square index (0-63)."""
        file = pos[0] // self.gui.square_size
        rank = 7 - (pos[1] // self.gui.square_size)
        return rank * 8 + file

    def _is_game_over(self):
        """Checks if the game has ended."""
        return not self.board.generate_legal_moves(self.board.side_to_move)

if __name__ == "__main__":
    WIDTH, HEIGHT = 640, 640
    
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("God Engine")
    
    game = Game(screen, player_color='w', ai_depth=3)
    game.run()