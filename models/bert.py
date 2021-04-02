import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

from models.modular_base import ModularBase


class Bert(ModularBase):

    def __init__(self, vocab_size, embedding_dim, dropout, num_heads, n_layers, net_seed=None, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)

        if net_seed is not None:
            torch.manual_seed(net_seed)

        modules = {
            "bert": BertModel(BertConfig(vocab_size=vocab_size, hidden_size=embedding_dim,
                                         num_attention_heads=num_heads, num_hidden_layers=n_layers,
                                         hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout,
                                         type_vocab_size=1), add_pooling_layer=False),
            "dropout": nn.Dropout(dropout)
        }

        super(Bert, self).__init__(modules, embedding_dim, "bert", *args, **kwargs)

    def extract_features(self, x):
        x = self.bert(x, attention_mask=x != 0).last_hidden_state
        return self.dropout(x)
