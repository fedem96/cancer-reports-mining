import torch
import torch.nn.functional as F
from torch import nn

from models.modular_base import ModularBase


class EmbMaxLin:
    def __init__(self, vocab_size, embedding_dim, num_filters, deep_features, reduce=None, directory=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)
        self.model = ModularBase(vocab_size, embedding_dim, deep_features, reduce, directory).to(device)
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

        # records_sizes = [len(record) for record in x]
        # reports_lengths = [len(report) for record in x for report in record]
        # x_tensor = torch.cat([torch.cat(record) for record in x])

        reports_lengths = [len(t) for t in x]
        x_tensor = torch.cat(x)

        deep_words = self.model.emb(x_tensor)
        reports_list = deep_words.split(reports_lengths)
        # deep_reports = []
        #
        # for words_embs in reports_list:
        #     # for conv in self.convs:
        #     #     words_embs = F.relu(conv(words_embs.transpose(0, 1).unsqueeze(0)).squeeze(0))
        #     sequence_emb = torch.max(self.convs(words_embs.transpose(0, 1).unsqueeze(0)).squeeze(0), 1)
        #     deep_reports.append(sequence_emb.values)
        #
        # return F.relu(torch.stack(deep_reports))

        reports = nn.utils.rnn.pad_sequence(reports_list, batch_first=True).permute([0, 2, 1])
        deep_reports = self.model.convs(reports).max(dim=2).values
        return F.relu(self.model.fc(F.relu(deep_reports)))

        # records_list = deep_reports.split(records_sizes)
        # records = nn.utils.rnn.pad_sequence(records_list, batch_first=True).permute([0, 2, 1])
        # deep_records = records.max(dim=2).values
        # return deep_records
