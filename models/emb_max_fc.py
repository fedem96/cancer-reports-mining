import torch
import torch.nn.functional as F
from torch import nn

from models.modular_base import ModularBase


class EmbMaxLin:
    def __init__(self, vocab_size, embedding_dim, num_filters, deep_features, directory=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)
        self.model = ModularBase(vocab_size, embedding_dim, deep_features, directory).to(device)
        self.model.extract_features = self.extract_features

        if tuple != type(num_filters) != list:
            num_filters = [num_filters]
        layers = []
        input_size = [embedding_dim] + num_filters
        for n in range(len(num_filters)):
            layers.append(nn.Conv1d(input_size[n], num_filters[n], 3, padding=1).to(self.model.current_device()))
            layers.append(nn.ReLU())
        # self.convs = [nn.Conv1d(embedding_dim, n, 3, padding=1).to(self.model.current_device()) for n in num_filters]
        self.model.convs = nn.Sequential(*layers)
        self.model.fc = nn.Linear(num_filters[-1], deep_features).to(self.model.current_device())

    def __getattr__(self, *args):
        return self.model.__getattribute__(*args)

    def extract_features(self, x):
        # batch_size = len(x)

        sizes = [len(t) for t in x]
        x = torch.cat(x)

        batch_embs = self.model.emb(x).split(sizes)
        # out = []
        #
        # for words_embs in batch_embs:
        #     # for conv in self.convs:
        #     #     words_embs = F.relu(conv(words_embs.transpose(0, 1).unsqueeze(0)).squeeze(0))
        #     sequence_emb = torch.max(self.convs(words_embs.transpose(0, 1).unsqueeze(0)).squeeze(0), 1)
        #     out.append(sequence_emb.values)
        #
        # return F.relu(torch.stack(out))

        ### padded version: requires more memory (~3-5x) and it's overall slower (~2-30x),
        ### due to slower (~3-50x) backprop (even if forward is faster ~3-5x) [tried with 5 variables to predict].
        ### With bigger batch_size is more worse
        ### Maybe we can leverage this implementation by making batches of sequences of similar length
        padded = nn.utils.rnn.pad_sequence(batch_embs)
        out = self.model.convs(padded.permute([1, 2, 0])).max(dim=2).values
        # return out
        return F.relu(self.model.fc(F.relu(out)))
