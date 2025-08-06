# engine/move.py

class Move:
    def __init__(self, from_sq, to_sq, piece, captured=None, promotion=None):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.piece = piece
        self.captured = captured
        self.promotion = promotion

    def __repr__(self):
        return f"{self.piece}: {self.from_sq} -> {self.to_sq}" + \
               (f" = {self.promotion}" if self.promotion else "")

