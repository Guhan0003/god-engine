# engine/board.py
from engine.move import Move
from engine import zobrist

_popcount = getattr(int, 'bit_count', lambda x: bin(x).count('1'))

NOT_A_FILE = 0xFEFEFEFEFEFEFEFE
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F

class Board:
    def __init__(self):
        self.bitboards = {}
        self.side_to_move = 'w'
        self.en_passant = None
        self.move_stack = []
        self.zobrist_hash = 0
        self.init_bitboards()

    def init_bitboards(self):
        self.bitboards = {
            'P': 0x000000000000FF00, 'N': 0x0000000000000042, 'B': 0x0000000000000024,
            'R': 0x0000000000000081, 'Q': 0x0000000000000008, 'K': 0x0000000000000010,
            'p': 0x00FF000000000000, 'n': 0x4200000000000000, 'b': 0x2400000000000000,
            'r': 0x8100000000000000, 'q': 0x0800000000000000, 'k': 0x1000000000000000,
        }
        self.compute_initial_zobrist_hash()

    def get_piece(self, square: int) -> str:
        for piece, bb in self.bitboards.items():
            if (bb >> square) & 1:
                return piece
        return '.'

    def get_occupied(self) -> int:
        occ = 0
        for bb in self.bitboards.values(): occ |= bb
        return occ

    def get_color_bb(self, color: str) -> int:
        bb = 0
        pieces = ('P','N','B','R','Q','K') if color == 'w' else ('p','n','b','r','q','k')
        for piece in pieces: bb |= self.bitboards[piece]
        return bb

    def make_move(self, move: Move) -> None:
        move.prev_en_passant = self.en_passant
        from_mask, to_mask = 1 << move.from_sq, 1 << move.to_sq

        if move.promotion == 'ep':
            self.bitboards[move.piece] &= ~from_mask
            self.bitboards[move.piece] |= to_mask
            captured_sq = move.to_sq + (8 if move.piece == 'P' else -8)
            self.bitboards[move.captured] &= ~(1 << captured_sq)
        else:
            if move.captured: self.bitboards[move.captured] &= ~to_mask
            self.bitboards[move.piece] &= ~from_mask
            if move.promotion: self.bitboards[move.promotion] |= to_mask
            else: self.bitboards[move.piece] |= to_mask

        self.en_passant = (move.from_sq + move.to_sq) // 2 if move.piece.lower() == 'p' and abs(move.to_sq - move.from_sq) == 16 else None
        
        self._update_hash_on_make(move)
        self.move_stack.append(move)
        self.side_to_move = 'b' if self.side_to_move == 'w' else 'w'
        self.zobrist_hash ^= zobrist.BLACK_TO_MOVE_KEY

    def undo_move(self) -> None:
        move = self.move_stack.pop()
        
        self.zobrist_hash ^= zobrist.BLACK_TO_MOVE_KEY
        self.side_to_move = 'b' if self.side_to_move == 'w' else 'w'
        self._update_hash_on_undo(move)
        
        self.en_passant = move.prev_en_passant
        from_mask, to_mask = 1 << move.from_sq, 1 << move.to_sq

        if move.promotion == 'ep':
            self.bitboards[move.piece] &= ~to_mask; self.bitboards[move.piece] |= from_mask
            captured_sq = move.to_sq + (8 if move.piece == 'P' else -8)
            self.bitboards[move.captured] |= (1 << captured_sq)
        else:
            if move.promotion: self.bitboards[move.promotion] &= ~to_mask
            else: self.bitboards[move.piece] &= ~to_mask
            self.bitboards[move.piece] |= from_mask
            if move.captured: self.bitboards[move.captured] |= to_mask

    def is_in_check(self, color: str) -> bool:
        king_bb = self.bitboards['K' if color == 'w' else 'k']
        if not king_bb: return True
        king_sq = _popcount(king_bb - 1)
        
        enemy_moves = self.generate_pseudo_legal_moves('b' if color == 'w' else 'w')
        return any(move.to_sq == king_sq for move in enemy_moves)

    def generate_legal_moves(self, color: str) -> list[Move]:
        legal = []
        for move in self.generate_pseudo_legal_moves(color):
            self.make_move(move)
            if not self.is_in_check(color):
                legal.append(move)
            self.undo_move()
        return legal

    def generate_pseudo_legal_moves(self, color: str) -> list[Move]:
        return (
            self.generate_pawn_moves(color)   + self.generate_knight_moves(color) +
            self.generate_bishop_moves(color) + self.generate_rook_moves(color)   +
            self.generate_queen_moves(color)  + self.generate_king_moves(color)
        )
    
    def _split_bb(self, bb: int):
        while bb:
            lsb = bb & -bb; yield lsb; bb &= bb - 1

    def generate_pawn_moves(self, color: str) -> list[Move]:
        moves = []
        occupied, promotions = self.get_occupied(), ('Q','R','B','N') if color == 'w' else ('q','r','b','n')
        if color == 'w':
            pawns, piece, enemy_bb = self.bitboards['P'], 'P', self.get_color_bb('b')
            pushes = (pawns << 8) & ~occupied
            for to_bb in self._split_bb(pushes):
                to_sq = _popcount(to_bb - 1); from_sq = to_sq - 8
                if to_sq // 8 == 7:
                    for promo in promotions: moves.append(Move(from_sq, to_sq, piece, promotion=promo))
                else: moves.append(Move(from_sq, to_sq, piece))
            
            double_pushes = (((pawns & 0xFF00) << 8) & ~occupied) << 8 & ~occupied
            for to_bb in self._split_bb(double_pushes):
                to_sq = _popcount(to_bb - 1); moves.append(Move(to_sq - 16, to_sq, piece))

            for cap_bb, shift in [((pawns & NOT_A_FILE) << 7, 7), ((pawns & NOT_H_FILE) << 9, 9)]:
                targets = cap_bb & enemy_bb
                for to_bb in self._split_bb(targets):
                    to_sq = _popcount(to_bb - 1); from_sq, captured = to_sq - shift, self.get_piece(to_sq)
                    if to_sq // 8 == 7:
                        for promo in promotions: moves.append(Move(from_sq, to_sq, piece, promotion=promo, captured=captured))
                    else: moves.append(Move(from_sq, to_sq, piece, captured=captured))
            
            if self.en_passant is not None:
                ep_mask = 1 << self.en_passant
                attackers = (((pawns & NOT_A_FILE) << 7) | ((pawns & NOT_H_FILE) << 9))
                if attackers & ep_mask:
                    from_sq = _popcount((((ep_mask >> 7) & NOT_H_FILE) | ((ep_mask >> 9) & NOT_A_FILE)) & pawns - 1)
                    moves.append(Move(from_sq, self.en_passant, 'P', promotion='ep', captured='p'))
        else: # Black moves
            pawns, piece, enemy_bb = self.bitboards['p'], 'p', self.get_color_bb('w')
            pushes = (pawns >> 8) & ~occupied
            for to_bb in self._split_bb(pushes):
                to_sq = _popcount(to_bb-1); from_sq = to_sq + 8
                if to_sq // 8 == 0:
                    for promo in promotions: moves.append(Move(from_sq, to_sq, piece, promotion=promo))
                else: moves.append(Move(from_sq, to_sq, piece))
            
            double_pushes = (((pawns & 0xFF000000000000) >> 8) & ~occupied) >> 8 & ~occupied
            for to_bb in self._split_bb(double_pushes):
                to_sq = _popcount(to_bb-1); moves.append(Move(to_sq + 16, to_sq, piece))

            for cap_bb, shift in [((pawns & NOT_H_FILE) >> 7, -7), ((pawns & NOT_A_FILE) >> 9, -9)]:
                targets = cap_bb & enemy_bb
                for to_bb in self._split_bb(targets):
                    to_sq = _popcount(to_bb-1); from_sq, captured = to_sq - shift, self.get_piece(to_sq)
                    if to_sq // 8 == 0:
                        for promo in promotions: moves.append(Move(from_sq, to_sq, piece, promotion=promo, captured=captured))
                    else: moves.append(Move(from_sq, to_sq, piece, captured=captured))

            if self.en_passant is not None:
                ep_mask = 1 << self.en_passant
                attackers = (((pawns & NOT_H_FILE) >> 7) | ((pawns & NOT_A_FILE) >> 9))
                if attackers & ep_mask:
                    from_sq = _popcount((((ep_mask << 7) & NOT_A_FILE) | ((ep_mask << 9) & NOT_H_FILE)) & pawns - 1)
                    moves.append(Move(from_sq, self.en_passant, 'p', promotion='ep', captured='P'))
        return moves

    def generate_knight_moves(self, color: str) -> list[Move]:
        moves = []
        piece = 'N' if color == 'w' else 'n'
        knights_bb, own_bb = self.bitboards[piece], self.get_color_bb(color)
        offsets = (17, 15, 10, 6, -6, -10, -15, -17)
        for from_bb in self._split_bb(knights_bb):
            from_sq = _popcount(from_bb - 1)
            for off in offsets:
                to_sq = from_sq + off
                if 0 <= to_sq < 64 and abs((from_sq % 8) - (to_sq % 8)) <= 2:
                    if not ((own_bb >> to_sq) & 1):
                        captured = self.get_piece(to_sq)
                        moves.append(Move(from_sq, to_sq, piece, captured=captured if captured != '.' else None))
        return moves

    def generate_king_moves(self, color: str) -> list[Move]:
        moves = []
        piece = 'K' if color == 'w' else 'k'
        king_bb, own_bb = self.bitboards[piece], self.get_color_bb(color)
        if not king_bb: return []
        
        from_sq = _popcount(king_bb - 1)
        dirs = (1, -1, 8, -8, 9, -9, 7, -7)
        for d in dirs:
            to_sq = from_sq + d
            if 0 <= to_sq < 64 and abs((from_sq % 8) - (to_sq % 8)) <= 1:
                if not ((own_bb >> to_sq) & 1):
                    captured = self.get_piece(to_sq)
                    moves.append(Move(from_sq, to_sq, piece, captured=captured if captured != '.' else None))
        return moves

    def sliding_moves(self, piece_bb: int, piece_char: str, dirs: tuple[int, ...], own: int, enemy: int) -> list[Move]:
        moves = []
        for from_bb in self._split_bb(piece_bb):
            from_sq = _popcount(from_bb - 1)
            for step in dirs:
                current_sq = from_sq
                while True:
                    next_sq = current_sq + step
                    if not (0 <= next_sq < 64): break
                    if abs((next_sq % 8) - (current_sq % 8)) > 1 and step not in (8, -8): break # Universal wrap check
                    
                    if (own >> next_sq) & 1: break
                    
                    captured = self.get_piece(next_sq)
                    moves.append(Move(from_sq, next_sq, piece_char, captured=captured if captured != '.' else None))
                    
                    if (enemy >> next_sq) & 1: break
                    current_sq = next_sq
        return moves

    def generate_bishop_moves(self, color: str) -> list[Move]:
        piece = 'B' if color == 'w' else 'b'
        return self.sliding_moves(self.bitboards[piece], piece, (7, -7, 9, -9), self.get_color_bb(color), self.get_color_bb('b' if color == 'w' else 'w'))

    def generate_rook_moves(self, color: str) -> list[Move]:
        piece = 'R' if color == 'w' else 'r'
        return self.sliding_moves(self.bitboards[piece], piece, (1, -1, 8, -8), self.get_color_bb(color), self.get_color_bb('b' if color == 'w' else 'w'))

    def generate_queen_moves(self, color: str) -> list[Move]:
        piece = 'Q' if color == 'w' else 'q'
        return self.sliding_moves(self.bitboards[piece], piece, (1, -1, 8, -8, 7, -7, 9, -9), self.get_color_bb(color), self.get_color_bb('b' if color == 'w' else 'w'))

    def compute_initial_zobrist_hash(self):
        h = 0
        for piece_char, bb in self.bitboards.items():
            piece_idx = zobrist.PIECE_MAP[piece_char]
            for piece_bb in self._split_bb(bb):
                sq = _popcount(piece_bb - 1)
                h ^= zobrist.PIECE_KEYS[piece_idx][sq]
        self.zobrist_hash = h
        
    def _update_hash_on_make(self, move: Move):
        h = self.zobrist_hash
        p_char, from_sq, to_sq = move.piece, move.from_sq, move.to_sq
        p_idx = zobrist.PIECE_MAP[p_char]
        h ^= zobrist.PIECE_KEYS[p_idx][from_sq]
        if move.promotion == 'ep':
            cap_char = 'p' if p_char == 'P' else 'P'
            cap_sq = to_sq + (8 if p_char == 'P' else -8)
            h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[cap_char]][cap_sq]
            h ^= zobrist.PIECE_KEYS[p_idx][to_sq]
        elif move.promotion:
            if move.captured: h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.captured]][to_sq]
            h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.promotion]][to_sq]
        else:
            if move.captured: h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.captured]][to_sq]
            h ^= zobrist.PIECE_KEYS[p_idx][to_sq]
        if move.prev_en_passant is not None: h ^= zobrist.EN_PASSANT_KEYS[move.prev_en_passant % 8]
        if self.en_passant is not None: h ^= zobrist.EN_PASSANT_KEYS[self.en_passant % 8]
        self.zobrist_hash = h

    def _update_hash_on_undo(self, move: Move):
        # Temporarily flip side_to_move to correctly XOR the turn key out
        self.zobrist_hash ^= zobrist.BLACK_TO_MOVE_KEY
        h = self.zobrist_hash
        p_char, from_sq, to_sq = move.piece, move.from_sq, move.to_sq
        p_idx = zobrist.PIECE_MAP[p_char]
        if self.en_passant is not None: h ^= zobrist.EN_PASSANT_KEYS[self.en_passant % 8]
        if move.prev_en_passant is not None: h ^= zobrist.EN_PASSANT_KEYS[move.prev_en_passant % 8]
        if move.promotion == 'ep':
            cap_char = 'p' if p_char == 'P' else 'P'
            cap_sq = to_sq + (8 if p_char == 'P' else -8)
            h ^= zobrist.PIECE_KEYS[p_idx][to_sq]
            h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[cap_char]][cap_sq]
            h ^= zobrist.PIECE_KEYS[p_idx][from_sq]
        elif move.promotion:
            h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.promotion]][to_sq]
            if move.captured: h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.captured]][to_sq]
            h ^= zobrist.PIECE_KEYS[p_idx][from_sq]
        else:
            h ^= zobrist.PIECE_KEYS[p_idx][to_sq]
            if move.captured: h ^= zobrist.PIECE_KEYS[zobrist.PIECE_MAP[move.captured]][to_sq]
            h ^= zobrist.PIECE_KEYS[p_idx][from_sq]
        self.zobrist_hash = h