# engine/eval.py

# Use int.bit_count if available (Python 3.10+), otherwise a fallback.
_popcount = getattr(int, 'bit_count', lambda x: bin(x).count('1'))

# Standard piece values. The king is given a very high value to ensure
# the AI never considers a move that would lead to its capture.
PIECE_VALUES = {
    'P': 100,  'N': 320,  'B': 330,  'R': 500,  'Q': 900,  'K': 20000,
    'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': -20000
}

def evaluate(board) -> int:
    """
    Calculates the material balance of the board from white's perspective.
    A positive score favors white, a negative score favors black.
    """
    score = 0
    # Iterate through each piece's bitboard for maximum efficiency,
    # multiplying the piece value by the number of pieces found.
    for piece, bb in board.bitboards.items():
        score += PIECE_VALUES[piece] * _popcount(bb)
    
    # Add a bonus for the side to move, which encourages activity.
    score += 10 if board.side_to_move == 'w' else -10
    
    return score