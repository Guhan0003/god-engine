# engine/move.py
"""
Defines the Move data structure for the chess engine.

This module contains a single dataclass, Move, which is used throughout the
engine to represent a single chess move. It holds all necessary information
to make, undo, and describe a move.
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Move:
    """
    Represents a single move in a chess game.

    Using a dataclass automatically generates essential methods like __init__,
    __repr__, and __eq__, making the code cleaner and less error-prone.

    Attributes:
        from_sq (int): The starting square index (0-63).
        to_sq (int): The destination square index (0-63).
        piece (str): The piece being moved (e.g., 'P', 'n').
        promotion (Optional[str]): The piece to promote to (e.g., 'Q'), if any.
                                   Also used as a sentinel 'ep' for en passant.
        captured (Optional[str]): The piece being captured, if any.
        prev_en_passant (Optional[int]): The en passant square *before* this
                                          move was made. This is critical for
                                          correctly undoing moves.
    """
    from_sq: int
    to_sq: int
    piece: str
    promotion: Optional[str] = None
    captured: Optional[str] = None
    prev_en_passant: Optional[int] = None

    def __str__(self) -> str:
        """Provides a simple, human-readable string representation for debugging."""
        # This custom __str__ provides a cleaner output than the default dataclass __repr__.
        promo_str = f"={self.promotion}" if self.promotion and self.promotion != 'ep' else ""
        capture_str = f"x{self.captured}" if self.captured else ""
        return f"Move({self.piece}{_square_to_coord(self.from_sq)}{capture_str}{_square_to_coord(self.to_sq)}{promo_str})"

# --- Helper function for pretty printing ---

def _square_to_coord(square: int) -> str:
    """Converts a square index (0-63) to algebraic notation (e.g., 'a1', 'h8')."""
    file = "abcdefgh"[square % 8]
    rank = "12345678"[square // 8]
    return f"{file}{rank}"