import torch
import torch.nn.functional as F
from torch import nn

from models.modular_base import ModularBase


class EmbMaxLin:
    def __init__(self, vocab_size, embedding_dim, num_filters, deep_features):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = ModularBase(vocab_size, embedding_dim, deep_features).to(device)
        self.model.extract_features = self.extract_features

        self.conv = nn.Conv1d(embedding_dim, num_filters, 3, padding=1).to(self.model.current_device())

    def __getattr__(self, *args):
        return self.model.__getattribute__(*args)

    def extract_features(self, x):
        # batch_size = len(x)

        sizes = [len(t) for t in x]
        x = torch.cat(x)

        batch_embs = self.model.emb(x).split(sizes)
        out = []

        conv = self.conv
        for words_embs in batch_embs:
            sequence_emb = torch.max(conv(words_embs.transpose(0, 1).unsqueeze(0)).squeeze(0), 1)
            out.append(sequence_emb.values)

        return F.relu(torch.stack(out))

        ### padded version: requires more memory (~3-5x) and it's overall slower (~2-30x),
        ### due to slower (~3-50x) backprop (even if forward is faster ~3-5x) [tried with 5 variables to predict].
        ### With bigger batch_size is more worse
        ### Maybe we can leverage this implementation by making batches of sequences of similar length
        # padded = nn.utils.rnn.pad_sequence(batch_embs)
        # out = self.conv(padded.permute([1, 2, 0])).max(dim=2).values
        # return F.relu(out)
