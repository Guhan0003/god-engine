from engine.move import Move

class Board:
    def __init__(self):
        self.init_bitboards()
        self.side_to_move = 'w'
        self.en_passant = None
        self.move_stack = []

    def init_bitboards(self):
        self.bitboards = {
            'P': 0x000000000000FF00,
            'N': 0x0000000000000042,
            'B': 0x0000000000000024,
            'R': 0x0000000000000081,
            'Q': 0x0000000000000008,
            'K': 0x0000000000000010,
            'p': 0x00FF000000000000,
            'n': 0x4200000000000000,
            'b': 0x2400000000000000,
            'r': 0x8100000000000000,
            'q': 0x0800000000000000,
            'k': 0x1000000000000000,
        }

    def get_piece(self, square):
        for piece, bb in self.bitboards.items():
            if (bb >> square) & 1:
                return piece
        return '.'

    def get_occupied(self):
        return sum(self.bitboards.values())

    def make_move(self, move):
        from_mask = 1 << move.from_sq
        to_mask = 1 << move.to_sq

        if move.promotion == 'ep':
            self.bitboards[move.piece] &= ~from_mask
            self.bitboards[move.piece] |= to_mask
            capture_sq = move.to_sq + (8 if move.piece.isupper() else -8)
            captured_piece = 'p' if move.piece.isupper() else 'P'
            self.bitboards[captured_piece] &= ~(1 << capture_sq)
            move.captured = captured_piece
        else:
            if move.captured:
                self.bitboards[move.captured] &= ~to_mask
            self.bitboards[move.piece] &= ~from_mask
            if move.promotion:
                self.bitboards[move.promotion] |= to_mask
            else:
                self.bitboards[move.piece] |= to_mask

        if move.piece.lower() == 'p' and abs(move.from_sq - move.to_sq) == 16:
            self.en_passant = (move.from_sq + move.to_sq) // 2
        else:
            self.en_passant = None

        self.move_stack.append(move)
        self.side_to_move = 'b' if self.side_to_move == 'w' else 'w'

    def undo_move(self):
        if not self.move_stack:
            return
        move = self.move_stack.pop()
        from_mask = 1 << move.from_sq
        to_mask = 1 << move.to_sq

        self.en_passant = None

        if move.promotion == 'ep':
            self.bitboards[move.piece] &= ~to_mask
            self.bitboards[move.piece] |= from_mask
            captured_sq = move.to_sq + (8 if move.piece.isupper() else -8)
            self.bitboards[move.captured] |= (1 << captured_sq)
        else:
            if move.promotion:
                self.bitboards[move.promotion] &= ~to_mask
                self.bitboards[move.piece] |= from_mask
            else:
                self.bitboards[move.piece] &= ~to_mask
                self.bitboards[move.piece] |= from_mask
            if move.captured:
                self.bitboards[move.captured] |= to_mask

        self.side_to_move = 'b' if self.side_to_move == 'w' else 'w'

    def is_in_check(self, color):
        king_sq = (self.bitboards['K' if color == 'w' else 'k']).bit_length() - 1
        for move in self.generate_pseudo_legal_moves('b' if color == 'w' else 'w'):
            if move[1] == king_sq:
                return True
        return False

    def generate_legal_moves(self, color):
        legal = []
        for from_sq, to_sq, *promo in self.generate_pseudo_legal_moves(color):
            piece = self.get_piece(from_sq)
            if piece == '.':
                continue
            move = Move(from_sq, to_sq, piece, promo[0] if promo else None)
            self.make_move(move)
            if not self.is_in_check(color):
                legal.append(move)
            self.undo_move()
        return legal

    def generate_pseudo_legal_moves(self, color):
        return (
            self.generate_pawn_moves(color) +
            self.generate_knight_moves(color) +
            self.generate_bishop_moves(color) +
            self.generate_rook_moves(color) +
            self.generate_queen_moves(color) +
            self.generate_king_moves(color)
        )

    def generate_pawn_moves(self, color):
        symbol = 'P' if color == 'w' else 'p'
        pawns = self.bitboards[symbol]
        empty = ~self.get_occupied() & 0xFFFFFFFFFFFFFFFF
        moves = []

        forward = -8 if color == 'w' else 8
        start_rank = 6 if color == 'w' else 1
        promotion_rank = 0 if color == 'w' else 7

        direction = lambda s: s << 8 if color == 'w' else s >> 8
        first_rank_mask = 0x000000000000FF00 if color == 'w' else 0x00FF000000000000

        # Single push
        single_push = direction(pawns) & empty
        for to in range(64):
            if (single_push >> to) & 1:
                from_sq = to + forward
                if to // 8 == promotion_rank:
                    for promo in ('Q', 'R', 'B', 'N') if color == 'w' else ('q', 'r', 'b', 'n'):
                        moves.append((from_sq, to, promo))
                else:
                    moves.append((from_sq, to))

        # Double push
        pawns_on_start = pawns & first_rank_mask
        first_step = direction(pawns_on_start) & empty
        second_step = direction(first_step) & empty
        for to in range(64):
            if (second_step >> to) & 1:
                from_sq = to + 2 * forward
                moves.append((from_sq, to))

        # Captures
        left_mask = 0xFEFEFEFEFEFEFEFE
        right_mask = 0x7F7F7F7F7F7F7F7F
        enemies = sum(bb for k, bb in self.bitboards.items() if (k.islower() if color == 'w' else k.isupper()))

        # Diagonal captures
        cap_left = ((pawns >> 9) & left_mask) if color == 'w' else ((pawns << 7) & left_mask)
        cap_right = ((pawns >> 7) & right_mask) if color == 'w' else ((pawns << 9) & right_mask)

        for to in range(64):
            for src_offset, cap_bb in [(+9, cap_left), (+7, cap_right)] if color == 'w' else [(-7, cap_left), (-9, cap_right)]:
                if (cap_bb & (1 << to)) and ((1 << to) & enemies):
                    from_sq = to + src_offset
                    if to // 8 == promotion_rank:
                        for promo in ('Q', 'R', 'B', 'N') if color == 'w' else ('q', 'r', 'b', 'n'):
                            moves.append((from_sq, to, promo))
                    else:
                        moves.append((from_sq, to))

        # En passant
        if self.en_passant is not None:
            ep_sq = self.en_passant
            ep_mask = 1 << ep_sq
            rank = ep_sq // 8
            if color == 'w':
                if ((pawns >> 9) & left_mask) & ep_mask:
                    moves.append((ep_sq + 9, ep_sq, 'ep'))
                if ((pawns >> 7) & right_mask) & ep_mask:
                    moves.append((ep_sq + 7, ep_sq, 'ep'))
            else:
                if ((pawns << 7) & left_mask) & ep_mask:
                    moves.append((ep_sq - 7, ep_sq, 'ep'))
                if ((pawns << 9) & right_mask) & ep_mask:
                    moves.append((ep_sq - 9, ep_sq, 'ep'))

        return moves

    def generate_knight_moves(self, color):
        knights = self.bitboards['N' if color == 'w' else 'n']
        own = sum(bb for k, bb in self.bitboards.items() if (k.isupper() if color == 'w' else k.islower()))
        moves = []
        offsets = [+17, +15, +10, +6, -6, -10, -15, -17]
        for i in range(64):
            if (knights >> i) & 1:
                rank, file = divmod(i, 8)
                for o in offsets:
                    to = i + o
                    if 0 <= to < 64:
                        tr, tf = divmod(to, 8)
                        if sorted([abs(tr - rank), abs(tf - file)]) == [1, 2]:
                            if not ((own >> to) & 1):
                                moves.append((i, to))
        return moves

    def generate_king_moves(self, color):
        king = self.bitboards['K' if color == 'w' else 'k']
        own = sum(bb for k, bb in self.bitboards.items() if (k.isupper() if color == 'w' else k.islower()))
        moves = []
        directions = [+1, -1, +8, -8, +9, -9, +7, -7]
        for i in range(64):
            if (king >> i) & 1:
                for d in directions:
                    to = i + d
                    if 0 <= to < 64:
                        if abs((i % 8) - (to % 8)) <= 1:
                            if not ((own >> to) & 1):
                                moves.append((i, to))
        return moves

    def sliding_moves(self, bb, directions, own, enemy):
        moves = []
        for from_sq in range(64):
            if (bb >> from_sq) & 1:
                for d in directions:
                    to_sq = from_sq
                    while True:
                        to_sq += d
                        if not (0 <= to_sq < 64):
                            break
                        if abs((from_sq % 8) - (to_sq % 8)) > 7 and d in [+7, +9, -7, -9]:
                            break
                        if (own >> to_sq) & 1:
                            break
                        moves.append((from_sq, to_sq))
                        if (enemy >> to_sq) & 1:
                            break
        return moves

    def generate_bishop_moves(self, color):
        bb = self.bitboards['B' if color == 'w' else 'b']
        own = sum(bb for k, bb in self.bitboards.items() if (k.isupper() if color == 'w' else k.islower()))
        enemy = sum(bb for k, bb in self.bitboards.items() if (k.islower() if color == 'w' else k.isupper()))
        return self.sliding_moves(bb, [+9, -9, +7, -7], own, enemy)

    def generate_rook_moves(self, color):
        bb = self.bitboards['R' if color == 'w' else 'r']
        own = sum(bb for k, bb in self.bitboards.items() if (k.isupper() if color == 'w' else k.islower()))
        enemy = sum(bb for k, bb in self.bitboards.items() if (k.islower() if color == 'w' else k.isupper()))
        return self.sliding_moves(bb, [+8, -8, +1, -1], own, enemy)

    def generate_queen_moves(self, color):
        bb = self.bitboards['Q' if color == 'w' else 'q']
        own = sum(bb for k, bb in self.bitboards.items() if (k.isupper() if color == 'w' else k.islower()))
        enemy = sum(bb for k, bb in self.bitboards.items() if (k.islower() if color == 'w' else k.isupper()))
        return self.sliding_moves(bb, [+1, -1, +8, -8, +9, -9, +7, -7], own, enemy)
