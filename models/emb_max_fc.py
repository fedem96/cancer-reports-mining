import torch
import torch.nn.functional as F
from torch import nn

from models.modular_base import ModularBase


class EmbMaxLin(ModularBase):
    def __init__(self, vocab_size, embedding_dim, num_filters, deep_features, net_seed=None, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)

        if net_seed is not None:
            torch.manual_seed(net_seed)

        if tuple != type(num_filters) != list:
            num_filters = [num_filters]
        layers = []
        input_size = [embedding_dim] + num_filters
        for n in range(len(num_filters)):
            layers.append(nn.Conv1d(input_size[n], num_filters[n], 3, padding=1).to(device))
            layers.append(nn.ReLU())

        modules = {
            "word_embedding": nn.Embedding(vocab_size, embedding_dim),
            "convs": nn.Sequential(*layers),
            "fc": nn.Linear(num_filters[-1], deep_features).to(device)
        }

        super(EmbMaxLin, self).__init__(modules, deep_features, "embmaxlin", *args, **kwargs)

    def extract_features(self, x):
        deep_words = self.word_embedding(x)
        deep_reports = self.convs(deep_words.permute(0,2,1)).max(dim=2).values
        return F.relu(self.fc(F.relu(deep_reports)))
