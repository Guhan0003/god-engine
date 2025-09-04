# engine/zobrist.py
import random

# Initialize the Zobrist key table
# We need a unique random number for each piece on each square.
PIECE_KEYS = [[random.getrandbits(64) for _ in range(64)] for _ in range(12)]

# A key to XOR if it's black's turn to move
BLACK_TO_MOVE_KEY = random.getrandbits(64)

# Keys for each of the four castling rights
CASTLING_KEYS = [random.getrandbits(64) for _ in range(4)]

# Keys for each possible en passant file (columns a-h)
EN_PASSANT_KEYS = [random.getrandbits(64) for _ in range(8)]

# Map pieces to their index in the PIECE_KEYS table (0-11)
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}