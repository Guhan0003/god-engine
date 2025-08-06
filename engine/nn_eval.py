import torch
import torch.nn as nn

class NN_Evaluator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(773, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)

def evaluate_with_nn(board, model):
    input_tensor = board_to_tensor(board)
    with torch.no_grad():
        return model(input_tensor).item()

def board_to_tensor(board):
    # Convert bitboards to 773-length input tensor
    return torch.zeros(773)
