# nn_eval.py
import torch
import torch.nn as nn
from engine.board import Board # Assuming 'engine' is in the python path

# Use int.bit_count for performance
_popcount = getattr(int, 'bit_count', lambda x: bin(x).count('1'))

# A mapping from piece characters to their corresponding "plane" in the input tensor.
PIECE_TO_PLANE = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
INPUT_FEATURES = (12 * 64) + 1 # 768 for pieces + 1 for side to move

class NN_Evaluator(nn.Module):
    """
    A simple feed-forward neural network for evaluating a chess position.
    The evaluation is from the perspective of the side to move.
    """
    def __init__(self, input_size: int = INPUT_FEATURES, hidden_size: int = 256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh() # Tanh squashes the output to a range of [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def evaluate_with_nn(board: Board, model: NN_Evaluator) -> float:
    """
    Takes a board state and a model, and returns a scalar evaluation.
    Uses torch.no_grad() for fast inference.
    """
    # Convert the board to a tensor and add a batch dimension (required by PyTorch models)
    input_tensor = board_to_tensor(board).unsqueeze(0)
    
    with torch.no_grad():
        # The output is scaled to be in centipawns, similar to traditional engines
        # A value of 1.0 from Tanh might correspond to ~500 centipawns (a rook's value)
        return model(input_tensor).item() * 500

def board_to_tensor(board: Board) -> torch.Tensor:
    """
    Converts a Board object into a 1D tensor for the neural network.
    
    The tensor has INPUT_FEATURES (769) elements, structured as follows:
    - 768 features: 12 planes of 64 squares each, one for each piece type.
                    A '1' indicates the presence of that piece on that square.
    - 1 feature:   The side to move (1.0 for white, 0.0 for black).
    """
    # Start with a zeroed tensor
    tensor = torch.zeros(INPUT_FEATURES, dtype=torch.float32)

    # Populate piece planes
    for piece, plane_idx in PIECE_TO_PLANE.items():
        bb = board.bitboards[piece]
        offset = plane_idx * 64
        
        # Efficiently iterate through the set bits of the bitboard
        while bb > 0:
            sq = _popcount(bb & -bb) - 1
            tensor[offset + sq] = 1.0
            bb &= bb - 1 # Clear the least significant bit

    # Populate side to move
    if board.side_to_move == 'w':
        tensor[768] = 1.0
    
    return tensor