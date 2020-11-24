from abc import abstractmethod
import os
from timeit import default_timer as timer

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD

from utils.metrics_logger import MetricsLogger, MacroF1Score, CohenKappaScore
from utils.utilities import Chronometer


class ModularBase(nn.Module):
    def __init__(self, vocab_size, embedding_dim, deep_features, directory=None):
        super(ModularBase, self).__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim)

        self.deep_features = deep_features

        self.classifiers = nn.ModuleDict({})
        self.regressors = nn.ModuleDict({})

        # self.ce = nn.CrossEntropyLoss()
        # self.l2 = nn.MSELoss()
        self.losses = {}
        self.dumb_baseline_train_accuracy = {}
        self.dumb_baseline_val_accuracy = {}
        self.num_classes = {}
        if directory is not None and not os.path.exists(directory):
            os.makedirs(directory)
        tb_dir = directory and os.path.join(directory, "logs")
        self.logger = MetricsLogger(terminal='table', tensorboard_dir=tb_dir, history_size=10)

    def current_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        with Chronometer("ef"): features = self.extract_features(x)
        with Chronometer("c"): classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        with Chronometer("r"): regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        return {"features": features, **classes, **regressions}

    @abstractmethod
    def extract_features(self, x):
        pass

    def update_losses(self, train_labels, val_labels):
        self.losses = {}
        self.dumb_baseline_train_accuracy = {}
        self.dumb_baseline_val_accuracy = {}
        self.num_classes = {}
        for var in self.classifiers:
            classes_occurrences = train_labels[var].value_counts().sort_index().values
            self.num_classes[var] = len(classes_occurrences)
            assert self.num_classes[var] == max(train_labels[var].dropna().unique()) + 1
            classes_weights = sum(classes_occurrences) / classes_occurrences
            classes_weights = torch.from_numpy(classes_weights).float().to(self.current_device())
            self.losses[var] = nn.CrossEntropyLoss(classes_weights)
            self.dumb_baseline_train_accuracy[var] = max(classes_occurrences) / sum(classes_occurrences)
            y_val = val_labels[var].dropna()
            self.dumb_baseline_val_accuracy[var] = (y_val == np.argmax(classes_occurrences)).sum() / len(y_val)
        for var in self.regressors:
            self.losses[var] = nn.MSELoss()  # TODO: multiply by a weight

    def fit(self, train_data, train_labels, val_data=None, val_labels=None, **hyperparams):
        self.update_losses(train_labels, val_labels)
        # train_size, val_size = 8192, 8192
        # train_data, train_labels = train_data[:train_size], train_labels.iloc[:train_size]
        # val_data, val_labels = val_data[:val_size], val_labels.iloc[:val_size]

        batch_size = hyperparams["batch_size"]
        self.activation_penalty = hyperparams["activation_penalty"]
        if "data_seed" in hyperparams:
            np.random.seed(hyperparams["data_seed"])

        train_data = [torch.tensor(t, device=self.current_device()) for t in train_data]
        if val_data is not None:
            val_data = [torch.tensor(t, device=self.current_device()) for t in val_data]
        self.optimizer = Adam(self.parameters(), lr=hyperparams["learning_rate"])

        num_batches, num_val_batches = len(train_data) // batch_size, len(val_data) // batch_size
        logger = self.logger
        metrics = {"Loss": min, "Accuracy": max, "MAE": min, "M-F1": MacroF1Score, "CKS": CohenKappaScore, "DBCS": max}
        for epoch in range(hyperparams["max_epochs"]):
            logger.prepare(epoch, metrics)
            perm = np.random.permutation(len(train_data))
            # perm = sorted(range(len(train_data)), key=lambda n: train_data[n].shape[0])
            train_data = [train_data[d] for d in perm]
            train_labels = train_labels.iloc[perm].reset_index(drop=True)

            for b in range(num_batches):
                batch = train_data[b * batch_size: (b + 1) * batch_size]
                batch_labels = train_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                batch_metrics = self.train_step(batch, batch_labels)
                logger.accumulate_train(batch_metrics, num_batches)

            if val_data is not None:
                for b in range(num_val_batches):
                    batch = val_data[b * batch_size: (b + 1) * batch_size]
                    batch_labels = val_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                    batch_metrics = self.train_step(batch, batch_labels, False)
                    logger.accumulate_val(batch_metrics, num_val_batches)

            logger.log()

        return logger.close()

    def train_step(self, data, labels, training=True):
        if training:
            self.optimizer.zero_grad()
        torch.set_grad_enabled(training)

        forwarded = self(data)
        losses = {}
        accuracies = {}
        maes = {}
        macro_f1 = {}
        cks = {}
        dbcs = {}
        device = self.current_device()

        total_loss = torch.tensor([0.0], device=self.current_device())

        for var in self.classifiers:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask]
            grth = torch.tensor(labels[var][mask].to_list(), dtype=torch.long, device=device, requires_grad=False)
            loss = self.losses[var](preds, grth)# / np.log2(self.num_classes[var])
            losses[var] = loss.item()
            total_loss += loss
            preds_classes = torch.argmax(preds.detach(), axis=1)
            accuracies[var] = (preds_classes == grth).float().mean().item()
            macro_f1[var] = [preds_classes, grth]
            cks[var] = [preds_classes, grth]
            dumb_baseline_accuracy = self.dumb_baseline_train_accuracy[var] if training else self.dumb_baseline_val_accuracy[var]
            dbcs[var] = (accuracies[var] - dumb_baseline_accuracy) / (1 - dumb_baseline_accuracy)

        for var in self.regressors:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask].squeeze(1)
            grth = torch.tensor(labels[var][mask].to_list(), device=device, requires_grad=False)
            loss = self.losses[var](preds, grth)
            losses[var] = loss.item()
            total_loss += loss
            maes[var] = (preds.detach() - grth).abs().mean().item()

        total_loss += self.activation_penalty * forwarded["features"].abs().sum()
        if training:
            total_loss.backward()
            self.optimizer.step()
        torch.set_grad_enabled(False)
        return {"Loss": losses, "Accuracy": accuracies, "MAE": maes, "M-F1": macro_f1, "CKS": cks, "DBCS": dbcs}

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
