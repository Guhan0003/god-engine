from engine.eval import evaluate
from engine.move import Move

def minimax(board, depth, alpha, beta, maximizing_player):
    # Performs minimax search with alpha-beta pruning

    if depth == 0:
        return evaluate(board), None  # Return evaluation at leaf node

    best_move = None

    def generate_all_moves(board, side):
        # Generates all legal Move objects for given side (w/b)
        legal_moves = board.generate_legal_moves(side)
        move_list = []

        for move in legal_moves:
            from_sq = move.from_sq
            to_sq = move.to_sq
            promotion = move.promotion
            piece = board.get_piece(from_sq)
            captured = board.get_piece(to_sq)

            # Ignore if same-side piece is on target square
            if captured == '.' or (side == 'w' and captured.isupper()) or (side == 'b' and captured.islower()):
                captured = None

            move_list.append(Move(from_sq, to_sq, piece, captured=captured, promotion=promotion))

        return move_list

    if maximizing_player:
        max_eval = float('-inf')  # Best score for white
        legal_moves = generate_all_moves(board, 'w')

        for move in legal_moves:
            board.make_move(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)  # Recurse as black
            board.undo_move()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)  # Update alpha for pruning
            if beta <= alpha:
                break  # Beta cutoff

        return max_eval, best_move

    else:
        min_eval = float('inf')  # Best score for black
        legal_moves = generate_all_moves(board, 'b')

        for move in legal_moves:
            board.make_move(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)  # Recurse as white
            board.undo_move()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)  # Update beta for pruning
            if beta <= alpha:
                break  # Alpha cutoff

        return min_eval, best_move
