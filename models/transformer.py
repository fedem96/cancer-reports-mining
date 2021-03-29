import math

import torch
import torch.nn as nn

from layers.positional_encoding import PositionalEncoding
from models.modular_base import ModularBase


class Transformer(ModularBase):

    def __init__(self, vocab_size, embedding_dim, deep_features, dropout, num_heads, n_layers, net_seed=None, *args, **kwargs):

        if net_seed is not None:
            torch.manual_seed(net_seed)

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, deep_features, dropout)
        modules = {
            "word_embedding": nn.Embedding(vocab_size, embedding_dim),
            "transformer_encoder": nn.TransformerEncoder(encoder_layers, n_layers)
        }
        super(Transformer, self).__init__(modules, deep_features, "transformer", *args, **kwargs)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        self.emb_dim_sqrt = math.sqrt(embedding_dim)

    def extract_features(self, x):
        x = x.long()
        x[x < 0] += 65536
        batch_embs = self.word_embedding(x) * self.emb_dim_sqrt
        mask = x == 0
        #batch_embs = self.pos_encoder(batch_embs)  # TODO: check and add position encoding
        batch_embs = self.transformer_encoder(batch_embs.permute(1, 0, 2), src_key_padding_mask=mask)
        return batch_embs.max(dim=0).values

    def extract_groups_features(self, records_list):
        records_sizes = [len(rec) for rec in records_list]
        records = torch.cat(records_list, dim=0).long()

        records = (self.word_embedding(records) * self.emb_dim_sqrt).permute(1,0,2)
        mask = torch.cat([rec == 0 for rec in records_list], dim=0).bool()

        #records = self.pos_encoder(records)  # TODO: check and add position encoding
        records = self.transformer_encoder(records, src_key_padding_mask=mask)
        records = records.split(records_sizes, 1)
        return [record_embeddings.max(dim=0).values for record_embeddings in records]

    def forward(self, x):
        super(Transformer, self).forward(x)


if __name__ == "__main__":

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
