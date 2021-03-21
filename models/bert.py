import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
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
                                         type_vocab_size=1)),
            "dropout": nn.Dropout(dropout)
        }

        super(Bert, self).__init__(modules, embedding_dim, "bert", *args, **kwargs)

    def extract_features(self, x):
        input_ids = pad_sequence(x, batch_first=True)
        return self.dropout(self.bert(input_ids, attention_mask=input_ids != 0)[1])
