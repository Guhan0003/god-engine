# engine/eval.py

# Piece values for evaluation
PIECE_VALUES = {
    'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': 0
}

# Evaluate the board from White's perspective
def evaluate(board):
    score = 0
    for i in range(64):
        piece = board.get_piece(i)
        if piece in PIECE_VALUES:
            score += PIECE_VALUES[piece]
    return score
