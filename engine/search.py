# engine/search.py
from engine.eval import evaluate
from dataclasses import dataclass

# Constants for TT entry flags
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

@dataclass
class TT_Entry:
    """Stores data for a single entry in the Transposition Table."""
    depth: int
    score: int
    flag: int
    best_move: object = None # Can be a Move object

class TranspositionTable:
    """A simple dictionary-based transposition table."""
    def __init__(self):
        self.table = {}

    def get(self, zobrist_hash: int):
        return self.table.get(zobrist_hash)

    def put(self, zobrist_hash: int, depth: int, score: int, flag: int, best_move=None):
        self.table[zobrist_hash] = TT_Entry(depth, score, flag, best_move)

# --- AI Search ---

# Global TT instance
tt = TranspositionTable()

def find_best_move(board, depth: int):
    """Public entry point to find the best move for the current position."""
    is_white_turn = board.side_to_move == 'w'
    negamax(board, depth, -float('inf'), float('inf'), is_white_turn)
    
    # Retrieve the best move from the TT for the root position
    root_entry = tt.get(board.zobrist_hash)
    if root_entry:
        return root_entry.score, root_entry.best_move
    return 0, None # Should not happen if search is run

def negamax(board, depth: int, alpha: float, beta: float, is_white_turn: bool):
    """
    A negamax search algorithm with alpha-beta pruning and transposition table integration.
    """
    alpha_orig = alpha
    
    # 1. Transposition Table Lookup
    tt_entry = tt.get(board.zobrist_hash)
    if tt_entry and tt_entry.depth >= depth:
        if tt_entry.flag == TT_EXACT:
            return tt_entry.score
        elif tt_entry.flag == TT_LOWERBOUND:
            alpha = max(alpha, tt_entry.score)
        elif tt_entry.flag == TT_UPPERBOUND:
            beta = min(beta, tt_entry.score)
        
        if alpha >= beta:
            return tt_entry.score

    # 2. Base Case
    if depth == 0:
        score = evaluate(board)
        return score if is_white_turn else -score

    # 3. Recursive Search
    best_move = None
    max_eval = -float('inf')
    moves = board.generate_legal_moves(board.side_to_move)
    
    if not moves: # Checkmate or stalemate
        return -20000 if board.is_in_check(board.side_to_move) else 0

    for move in moves:
        board.make_move(move)
        eval = -negamax(board, depth - 1, -beta, -alpha, not is_white_turn)
        board.undo_move()
        
        if eval > max_eval:
            max_eval = eval
            best_move = move
        
        alpha = max(alpha, eval)
        if alpha >= beta:
            break # Pruning

    # 4. Transposition Table Store
    flag = TT_EXACT
    if max_eval <= alpha_orig:
        flag = TT_UPPERBOUND
    elif max_eval >= beta:
        flag = TT_LOWERBOUND
        
    tt.put(board.zobrist_hash, depth, max_eval, flag, best_move)
    
    return max_eval