import pygame
import sys
import os
from engine.board import Board
from engine.search import minimax

# Constants
WIDTH, HEIGHT = 640, 640
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

# Piece images (you must have these!)
PIECE_IMAGES = {}

def load_images():
    pieces = ['P', 'R', 'N', 'B', 'Q', 'K', 'p', 'r', 'n', 'b', 'q', 'k']
    for piece in pieces:
        PIECE_IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(f"assets/{piece}.png"), (SQUARE_SIZE, SQUARE_SIZE)
        )

def load_piece_images():
    pieces = ['P', 'R', 'N', 'B', 'Q', 'K']
    images = {}

    for color in ['w', 'b']:
        for piece in pieces:
            name = f"{color}{piece}"
            image_path = os.path.join("assets", f"{name}.png")
            images[name] = pygame.image.load(image_path)

    return images

def draw_board(win, board):
    for row in range(ROWS):
        for col in range(COLS):
            color = WHITE if (row + col) % 2 == 0 else BROWN
            pygame.draw.rect(win, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

            piece = board.squares[row][col]
            if piece != '.':
                win.blit(PIECE_IMAGES[piece], (col * SQUARE_SIZE, row * SQUARE_SIZE))

def main():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("God Engine Chess")

    board = Board()
    load_images()

    running = True
    selected = None

    while running:
        draw_board(win, board)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                row = y // SQUARE_SIZE
                col = x // SQUARE_SIZE
                index = row * 8 + col

                if selected is None:
                    selected = index
                else:
                    from engine.move import Move
                    piece = board.get_piece(selected)
                    move = Move(selected, index, piece)
                    board.make_move(move)
                    selected = None

                    # AI Move
                    _, best_move = minimax(board, 2, float('-inf'), float('inf'), False)
                    if best_move:
                        board.make_move(best_move)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
