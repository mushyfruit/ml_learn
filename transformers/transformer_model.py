from torch import Tensor, nn

from decoder import TransformerDecoder
from encoder import TransformerEncoder

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6, dim_model=512, num_heads=6,
                 dim_ff=2048, dropout=0.1):
        super().__init__()
        self.encoder = TransformerEncoder(num_layers=num_encoder_layers,
                                          dim_model=dim_model,
                                          num_heads=num_heads,
                                          dim_ff=dim_ff,
                                          dropout=dropout)
        self.decoder = TransformerDecoder(num_layers=num_decoder_layers,
                                          dim_model=dim_model,
                                          num_heads=num_heads,
                                          dim_ff=dim_ff,
                                          dropout=dropout)

    def forward(self, source, target):
        # Teacher forcing with the target sequence.
        return self.decoder(target, self.encoder(source))