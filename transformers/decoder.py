import torch
from torch import Tensor, nn

from multi_head_attention import MultiHeadAttention
from utils import Residual, feed_forward, position_encoding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention_1 = Residual(
            MultiHeadAttention(
                num_heads, dim_model, dim_q, dim_k
            ),
            dimension=dim_model,
            dropout=dropout
        )
        self.attention_2 = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_ff),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, target, memory):
        # Computing attention scores between elements of target and itself.
        target = self.attention_1(target, target, target)
        # Keys and Values come from output of encoder. (Encoder/Decoder attention)
        target = self.attention_2(target, memory, memory)
        # Results are passed through ff layer.
        return self.feed_forward(target)

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers=6, dim_model=512, num_heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim_model, num_heads, dim_ff, dropout
                ) for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, target, memory):
        seq_len, dimension = target.size(1), target.size(2)

        # Positional encoding added to output embedding.
        target += position_encoding(seq_len, dimension)
        for layer in self.layers:
            target = layer(target, memory)

        # After decoder block, linear -> softmax
        return torch.softmax(self.linear(target), dim=-1)