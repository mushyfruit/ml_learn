from torch import Tensor, nn
from utils import Residual, feed_forward, position_encoding
from multi_head_attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model=512, num_heads=6, dim_ff=2048, dropout=0.1):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)

        self.attention = Residual(
            MultiHeadAttention(
                num_heads, dim_model, dim_q, dim_k
            ),
            dimension=dim_model,
            dropout=dropout
        )

        self.feed_forward = Residual(
            feed_forward(
                dim_model, dim_ff
            ),
            dimension=dim_model,
            dropout=dropout
        )

    def forward(self, src):
        src = self.attention(src, src, src)
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, dim_model=512, num_heads=8, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(dim_model, num_heads, dim_ff, dropout)
                for x in range(num_layers)
            ]
        )

    def forward(self, src):
        # src -> (batch_size, seq_len, dim_model)
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)

        # Applies each TransformerEncoderLayer to source sequence.
        for layer in self.layers:
            src = layer(src)

        # Returns the transformed source sequence.
        return src
