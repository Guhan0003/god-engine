# gui/gui.py
import pygame
import os

class GUI:
    """
    Manages all visual components and drawing for the chess game.
    """
    def __init__(self):
        self.square_size = 640 // 8
        self.colors = [(240, 217, 181), (181, 136, 99)]
        self.images = self._load_images()
        self.font = pygame.font.SysFont("consolas", 32, bold=True)

    def _load_images(self):
        """
        Loads piece images from the assets directory.
        The dictionary keys created here ('P', 'p', 'N', 'n', etc.) are the
        exact characters the engine uses, which is crucial for correct drawing.
        """
        images = {}
        # These are the base piece types, used to find the asset files.
        asset_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']

        for piece in asset_pieces:
            # The key for white pieces is the uppercase character (e.g., 'P').
            white_key = piece.upper()
            # The key for black pieces is the lowercase character (e.g., 'p').
            black_key = piece.lower()

            # The path always uses the uppercase character, per your file names.
            white_path = os.path.join("assets", f"w{piece.upper()}.png")
            black_path = os.path.join("assets", f"b{piece.upper()}.png")

            images[white_key] = pygame.transform.scale(
                pygame.image.load(white_path).convert_alpha(), 
                (self.square_size, self.square_size)
            )
            images[black_key] = pygame.transform.scale(
                pygame.image.load(black_path).convert_alpha(), 
                (self.square_size, self.square_size)
            )
            
        print("Piece images loaded successfully.")
        return images

    def draw_gamestate(self, screen, board, selected_sq, legal_moves_for_selected):
        """Draws the entire game state by calling helper methods."""
        self._draw_board(screen)
        self._draw_highlights(screen, selected_sq, legal_moves_for_selected)
        self._draw_pieces(screen, board)
        self._draw_game_over_text(screen, board)

    def _draw_board(self, screen):
        """Draws the checkerboard squares."""
        for r in range(8):
            for c in range(8):
                color = self.colors[(r + c) % 2]
                pygame.draw.rect(screen, color, (c * self.square_size, r * self.square_size, self.square_size, self.square_size))

    def _draw_pieces(self, screen, board):
        """
        Draws the pieces on the board. This function now correctly uses
        the piece character directly as the key to find the image.
        """
        for sq in range(64):
            piece = board.get_piece(sq) # Gets the character, e.g., 'P', 'b', 'k'
            if piece != '.':
                # The 'piece' character is the direct key for the self.images dictionary.
                rank, file = divmod(sq, 8)
                # The (7 - rank) correctly flips the board vertically for Pygame's coordinate system.
                screen.blit(self.images[piece], (file * self.square_size, (7 - rank) * self.square_size))

    def _draw_highlights(self, screen, selected_sq, legal_moves):
        """Draws highlights for the selected piece and its legal moves."""
        if selected_sq is None:
            return

        rank, file = divmod(selected_sq, 8)
        pygame.draw.rect(screen, (255, 255, 0), (file * self.square_size, (7 - rank) * self.square_size, self.square_size, self.square_size), 4)

        s = pygame.Surface((self.square_size, self.square_size), pygame.SRCALPHA)
        s.fill((0, 150, 0, 90))
        for to_sq in legal_moves:
            rank, file = divmod(to_sq, 8)
            screen.blit(s, (file * self.square_size, (7 - rank) * self.square_size))
            
    def _draw_game_over_text(self, screen, board):
        """Checks for and displays checkmate or stalemate messages."""
        if not board.generate_legal_moves(board.side_to_move):
            in_check = board.is_in_check(board.side_to_move)
            message = "Checkmate!" if in_check else "Stalemate!"
            
            text = self.font.render(message, True, (200, 20, 20))
            text_rect = text.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))
            
            bg_rect = text_rect.inflate(20, 20)
            pygame.draw.rect(screen, (240, 217, 181), bg_rect)
            pygame.draw.rect(screen, (0, 0, 0), bg_rect, 2)
            
            screen.blit(text, text_rect)