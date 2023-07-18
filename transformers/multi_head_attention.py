import torch
import torch.nn.functional as f
from torch import Tensor, nn


def scaled_dot_product_attention(query, key, value):
    # bmm() = batch matrix multiplication (bnm) (bmp) -> (bnp)
    # q/k: (batch, num_queries (seq_len), d_k)
    temp = query.bmm(key.transpose(1, 2))
    # Divide by sqrt of d_k.
    scale = query.size(-1) ** 0.5
    # With dim=-1, ensure softmax runs on d_k for individual query
    # Else, we would run over all queries.
    softmax = f.softmax(temp/scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        # Assume input dim_in and transform to dim_q
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query, key, value):
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_in, dim_q, dim_k):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        # Takes input after we've concatenated the multi-heads together.
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query, key, value):
        # Concat together all the different heads together.
        return self.linear(
            torch.cat(
                # Return output tensors from each attention head.
                # Concat along feature dimensions via torch.cat()
                # Linear layer reduces dim to dim_in
                # h() -> automatically calls the forward() func
                [h(query, key, value) for h in self.heads], dim=-1
            )
        )

