import pygame
import sys
import os
from engine.board import Board
from engine.move import Move
from engine.search import minimax

# Pygame initialization
pygame.init()
  
# Screen settings
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("God Engine Chess")
clock = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
HIGHLIGHT_COLOR = (0, 255, 0)
MOVE_INDICATOR_COLOR = (100, 255, 100)

# Load piece images
PIECE_IMAGES = {}
def load_images():
    for color in ['w', 'b']:
        for piece in ['P', 'R', 'N', 'B', 'Q', 'K']:
            name = f"{color}{piece}"
            path = os.path.join("assets", f"{name}.png")
            PIECE_IMAGES[name] = pygame.transform.scale(
                pygame.image.load(path), (SQUARE_SIZE, SQUARE_SIZE)
            )

# Drawing function
def draw_board(win, board, selected_square=None, legal_moves=[]):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            rect = pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE)
            pygame.draw.rect(screen, color, rect)

            index = (7 - row) * 8 + col  # Flip to make white at bottom

            # Highlight selected square
            if selected_square == index:
                pygame.draw.rect(screen, HIGHLIGHT_COLOR, rect, 4)

            # Highlight legal moves
            if index in legal_moves:
                pygame.draw.circle(
                    screen,
                    MOVE_INDICATOR_COLOR,
                    (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2),
                    10
                )

            # Draw piece
            piece = board.get_piece(index)
            if piece != '.':
                key = ('w' if piece.isupper() else 'b') + piece.upper()
                if key in PIECE_IMAGES:
                    screen.blit(PIECE_IMAGES[key], rect.topleft)

# Main game loop
def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))  # ✅ THIS LINE CREATES `win`
    pygame.display.set_caption("God Engine Chess")
    clock = pygame.time.Clock()

    board = Board()
    load_images()
    player_color = 'w'
    selected = None
    selected_legal_moves = []
    move_history = []

    running = True

    while running:
        draw_board(win, board, selected, selected_legal_moves)
        pygame.display.flip()
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and board.side_to_move == player_color:
                x, y = pygame.mouse.get_pos()
                col = x // SQUARE_SIZE
                row = y // SQUARE_SIZE
                clicked_square = (7 - row) * 8 + col

                if selected is not None and clicked_square in selected_legal_moves:
                    move = next(
                        (m for m in board.generate_legal_moves(board.side_to_move)
                        if m.from_sq == selected and m.to_sq == clicked_square),
                        None
                    )
                    if move:
                        board.make_move(move)
                        move_history.append(move)
                        selected = None
                        selected_legal_moves = []

                elif board.get_piece(clicked_square) != '.' and (
                    (player_color == 'w' and board.get_piece(clicked_square).isupper()) or
                    (player_color == 'b' and board.get_piece(clicked_square).islower())
                ):
                    selected = clicked_square
                    selected_legal_moves = [
                        m.to_sq for m in board.generate_legal_moves(board.side_to_move)
                        if m.from_sq == selected
                    ]
                else:
                    selected = None
                    selected_legal_moves = []

        # AI move
        if board.side_to_move != player_color:
            _, ai_move = minimax(board, 2, float('-inf'), float('inf'), board.side_to_move == 'w')
            if ai_move:
                board.make_move(ai_move)
                move_history.append(ai_move)
                selected = None
                selected_legal_moves = []

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
