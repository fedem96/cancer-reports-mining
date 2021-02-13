import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from layers.positional_encoding import PositionalEncoding
from models.modular_base import ModularBase


class Transformer(ModularBase):

    def __init__(self, vocab_size, embedding_dim, deep_features, dropout, num_heads, n_layers, net_seed=None):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print("device:", device)

        if net_seed is not None:
            torch.manual_seed(net_seed)

        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, deep_features, dropout)
        # self.model.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers).to(self.model.current_device())
        # self.model.transformer_encoder = nn.TransformerEncoderLayer(embedding_dim, num_heads, deep_features, dropout).to(self.model.current_device())
        modules = {
            "word_embedding": nn.Embedding(vocab_size, embedding_dim),
            "transformer_encoder": nn.TransformerEncoder(encoder_layers, n_layers)
        }
        # self.model = ModularBase(modules, deep_features, "transformer", directory)
        super(Transformer, self).__init__(modules, deep_features, "transformer")
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # self.model.extract_features = self.extract_features
        # self.init_weights()


    # def _generate_square_subsequent_mask(self, sz):
    #     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    #     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    #     return mask

    # def init_weights(self):
    #     initrange = 0.1
    #     self.encoder.weight.data.uniform_(-initrange, initrange)
    #     self.decoder.bias.data.zero_()
    #     self.decoder.weight.data.uniform_(-initrange, initrange)

    # def forward(self, src):
    #     if self.src_mask is None or self.src_mask.size(0) != len(src):
    #         device = src.device
    #         mask = self._generate_square_subsequent_mask(len(src)).to(device)
    #         self.src_mask = mask
    #
    #     src = self.encoder(src) * math.sqrt(self.ninp)
    #     src = self.pos_encoder(src)
    #     output = self.transformer_encoder(src, self.src_mask)
    #     output = self.decoder(output)
    #     return output

    # def __getattr__(self, *args):
    #     return self.model.__getattribute__(*args)

    def extract_features(self, x): # this is the version with padding
        batch_size = len(x)

        sizes = [len(t) for t in x]
        # target = torch.zeros(batch_size, 1, self.model.deep_features).to(self.model.current_device())
        emb_dim_sqrt = math.sqrt(self.word_embedding.embedding_dim)
        batch_embs = pad_sequence((self.word_embedding(torch.cat(x, dim=0).long()) * emb_dim_sqrt).split(sizes), batch_first=True)
        mask = batch_embs.sum(dim=2) == 0

        #batch_embs = self.pos_encoder(batch_embs)  # TODO: check and add position encoding
        batch_embs = self.transformer_encoder(batch_embs.permute(1, 0, 2), src_key_padding_mask=mask)
        return batch_embs.max(dim=0).values
    
    def forward(self, x):
        super(Transformer, self).forward(x)

    # def extract_features(self, x):
    #     # batch_size = len(x)
    #
    #     sizes = [len(t) for t in x]
    #     x = torch.cat(x).long()
    #
    #     emb_dim_sqrt = math.sqrt(self.model.emb.embedding_dim)
    #     batch_embs = (self.model.emb(x) * emb_dim_sqrt).split(sizes)
    #     out = []
    #
    #     target = torch.zeros(1, 1, self.model.deep_features).to(self.model.current_device())
    #
    #     for words_embs in batch_embs:
    #         #mask = self._generate_square_subsequent_mask(len(words_embs)).to(self.model.current_device())
    #         embs = self.pos_encoder(words_embs.unsqueeze(0))
    #
    #         #embs = self.transformer_encoder(embs, mask)
    #         embs = self.transformer_encoder(embs)
    #
    #         #num_outputs = len(self.model.classifiers) + len(self.model.regressors)
    #         sequence_emb = self.transformer_decoder(target, embs)
    #         out.append(sequence_emb.squeeze())
    #     #out = [self.transformer_decoder(target, self.transformer_encoder(self.pos_encoder(words_emb.unsqueeze(0)))).squeeze() for words_emb in batch_embs]
    #
    #     return torch.stack(out)


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
