import torch
from torch import Tensor, nn


class Residual(nn.Module):
    def __init__(self, sublayer, dimension, dropout):
        super().__init__()
        self.sublayer = sublayer
        # Norms over the feature dimensions.
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors):
        # Input tensors passed to sublayer first, then dropout applied.
        # Output is added to original input tensor element-wise.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


def feed_forward(dim_input, dim_ff):
    return nn.Sequential(
        nn.Linear(dim_input, dim_ff),
        nn.ReLU(),
        nn.Linear(dim_ff, dim_input)
    )


def position_encoding(seq_len, dim_model, device=torch.device("cpu")):
    # Creates a seq of ints from 0 to (seq_len-1)
    # Reshapes the tensor to (1, seq_len, 1). Position indices for words in sequence.
    # equivalent to "pos" in the paper.
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)

    # dim_model is the hidden size. (1, 1, dim_model)
    # Equivalent to "d" in the paper. Size of word embeddings.
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)

    # Paper formula: phase = (pos / (1000 ** (2i/d)))
    phase = pos / (1e4 ** torch.div(dim, dim_model, rounding_mode="floor"))

    # True: torch.sin(), False: torch.cos()
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))
