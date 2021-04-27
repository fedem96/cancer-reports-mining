import math

import torch
import torch.nn as nn

from layers.positional_encoding import PositionalEncoding
from models.nn.modular_base import ModularBase


class Transformer(ModularBase):

    def __init__(self, vocab_size, embedding_dim, dim_feedforward, dropout, num_heads, n_layers, encode_position, net_seed=None, *args, **kwargs):

        if net_seed is not None:
            torch.manual_seed(net_seed)

        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout)
        modules = {
            "word_embedding": nn.Embedding(vocab_size, embedding_dim),
            "transformer_encoder": nn.TransformerEncoder(encoder_layer, n_layers)
        }
        super(Transformer, self).__init__(modules, embedding_dim, "transformer", *args, **kwargs)
        self.encode_position = encode_position
        if encode_position:
            self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.emb_dim_sqrt = math.sqrt(embedding_dim)

    def extract_features(self, x):
        # x.shape: (num_reports, num_tokens)
        batch_embs = self.word_embedding(x) * self.emb_dim_sqrt
        # batch_embs.shape: (num_reports, num_tokens, num_features)
        mask = x == 0
        if self.encode_position:
            batch_embs = self.pos_encoder(batch_embs)
        # required shape by transformer_encoder: (sequence_length, batch_size, num_features)
        return self.transformer_encoder(batch_embs.permute(1, 0, 2), src_key_padding_mask=mask).permute(1,0,2)  # returned shape: (num_reports, num_tokens, num_features)


if __name__ == "__main__":
    def main():
        encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        src = torch.rand(11, 34, 512)
        out = encoder_layer(src)

        print(src.shape)
        print(out.shape)

        decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        memory = torch.rand(10, 32, 512)
        tgt = torch.rand(20, 32, 512)
        out = decoder_layer(tgt, memory)

        print(memory.shape)
        print(tgt.shape)
        print(out.shape)
    main()
