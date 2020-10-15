from abc import abstractmethod
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from utils.metrics_logger import MetricsLogger
from utils.utilities import Chronometer


class ModularBase(nn.Module):
    def __init__(self, vocab_size, embedding_dim, deep_features):
        super(ModularBase, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)

        self.deep_features = deep_features

        self.classifiers = nn.ModuleDict({})
        self.regressors = nn.ModuleDict({})

        self.ce = nn.CrossEntropyLoss()
        self.l2 = nn.MSELoss()
        self.logger = MetricsLogger(terminal='table', history_size=10)

    def current_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        features = self.extract_features(x)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        return {**classes, **regressions}

    @abstractmethod
    def extract_features(self, x):
        pass

    def fit(self, train_data, train_labels, num_epochs, batch_size, val_data=None, val_labels=None):
        train_data = [torch.tensor(t, device=self.current_device()) for t in train_data]
        if val_data is not None:
            val_data = [torch.tensor(t, device=self.current_device()) for t in val_data]
        self.optimizer = Adam(self.parameters(), lr=10 ** -3)

        # TODO: handle hyperparameters
        perm = np.random.permutation(len(train_data))
        train_data = [train_data[d] for d in perm]
        train_labels = train_labels.iloc[perm].reset_index(drop=True)

        num_batches = len(train_data) // batch_size
        logger = self.logger
        for epoch in range(num_epochs):
            start = timer()
            perm = np.random.permutation(len(train_data))
            # perm = sorted(range(len(train_data)), key=lambda n: train_data[n].shape[0])
            train_data = [train_data[d] for d in perm]
            train_labels = train_labels.iloc[perm].reset_index(drop=True)
            logger.prepare(epoch, {"Loss": min, "Accuracy": max, "MAE": min})
            for b in range(num_batches):
                batch = train_data[b * batch_size: (b + 1) * batch_size]
                batch_labels = train_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                losses, accuracies, maes = self.train_step(batch, batch_labels)
                logger.accumulate_train({"Loss": losses, "Accuracy": accuracies, "MAE": maes}, num_batches)
            if val_data is not None:
                losses, accuracies, maes = self.train_step(val_data, val_labels, False)
                logger.accumulate_val({"Loss": losses, "Accuracy": accuracies, "MAE": maes})
            logger.log()
            print("time", (timer()-start))
        return logger

    def train_step(self, data, labels, training=True):
        if training:
            self.optimizer.zero_grad()
        torch.set_grad_enabled(training)

        predictions = self(data)
        losses = {}
        accuracies = {}
        maes = {}
        device = self.current_device()

        total_loss = torch.tensor([0.0], device=self.current_device())

        for var in self.classifiers:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = predictions[var][mask]
            grth = torch.tensor(labels[var][mask].to_list(), dtype=torch.long, device=device, requires_grad=False)
            loss = self.ce(preds, grth)
            losses[var] = loss.item()
            total_loss += loss
            # accuracies[var] = (torch.argmax(preds, axis=1) == grth).float().mean().item()
            accuracies[var] = (torch.argmax(preds.detach(), axis=1) == grth).float().mean().item()

        for var in self.regressors:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = predictions[var][mask].squeeze(1)
            grth = torch.tensor(labels[var][mask].to_list(), device=device, requires_grad=False)
            loss = self.l2(preds, grth)
            losses[var] = loss.item()
            total_loss += loss
            # maes[var] = (preds - grth).abs().mean().item()
            maes[var] = (preds.detach() - grth).abs().mean().item()

        if training:
            total_loss.backward()
            self.optimizer.step()
        torch.set_grad_enabled(False)
        return losses, accuracies, maes

    def add_classification(self, task_name, num_classes):
        self.classifiers[task_name] = nn.Linear(self.deep_features, num_classes).to(self.current_device())

    def add_regression(self, task_name):
        self.regressors[task_name] = nn.Linear(self.deep_features, 1).to(self.current_device())

    def infer(self, x):
        with torch.no_grad():
            if type(x) == list or type(x) == tuple:
                x = [torch.tensor(example, device=self.current_device()) for example in x]
                predictions = self(x)
            else:
                x = torch.tensor(x, device=self.current_device())
                predictions = self([x])
        return predictions


if __name__ == "__main__":
    def main():
        import pandas as pd
        # from models.emb_max_fc import EmbMaxLin
        # model = EmbMaxLin(1021, 256, 256, 256)
        from models.transformer import Transformer
        model = Transformer(1021, 256, 8, 256, 256, 0.1)
        model.add_classification("sede", 5)
        model.add_classification("morfologia", 3)
        model.add_classification("stadio", 5)
        model.add_regression("dimensioni")

        a = torch.randint(0, 1020, [35])
        b = torch.randint(0, 1020, [40])
        c = torch.randint(0, 1020, [12])
        k = torch.randint(0, 1020, [17])
        u = torch.randint(0, 1020, [4])
        v = torch.randint(0, 1020, [21])
        x = torch.randint(0, 1020, [12])
        y = torch.randint(0, 1020, [7])
        z = torch.randint(0, 1020, [20])
        l = [a, b, c, k, u, v, x, y, z]
        labels = pd.DataFrame([[3,None,3,17.5], [4,2,0,None], [3,1,2,17.5], [3,0,None,17.5], [4,0,4,None], [2,1,3,10.3], [0,1,1,0.4], [3,2,None,4.5], [1,0,4,2.1]], columns=["sede", "morfologia", "stadio", "dimensioni"])

        model.fit(l, labels, 100, 2)
        print("finish")

    main()
