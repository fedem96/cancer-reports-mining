import os
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from utils.metrics_logger import MetricsLogger, MacroF1Score, CohenKappaScore


class ModularBase(nn.Module):
    def __init__(self, modules_dict, deep_features, model_name=None, directory=None):
        super(ModularBase, self).__init__()
        self.net = nn.ModuleDict(OrderedDict({
            **modules_dict,
            "classifiers": nn.ModuleDict({}),
            "regressors": nn.ModuleDict({})
        }))

        self.deep_features = deep_features

        self.classifiers = self.net['classifiers']
        self.regressors = self.net['regressors']

        for module_name in modules_dict:
            if hasattr(self, module_name):
                raise NameError("Name {} unavailable".format(module_name))
            setattr(self, module_name, modules_dict[module_name])

        # self.ce = nn.CrossEntropyLoss()
        # self.l2 = nn.MSELoss()
        self.losses = {}
        self.dumb_baseline_train_accuracy = {}
        self.dumb_baseline_val_accuracy = {}
        self.num_classes = {}

        self.reduce_type = "data"
        self.features_reducers = None
        self.logits_reducer = None
        self.forward = self._forward_simple

        if directory is not None and not os.path.exists(directory):
            os.makedirs(directory)
        tb_dir = directory and os.path.join(directory, "logs")
        self.logger = MetricsLogger(terminal='table', tensorboard_dir=tb_dir, aim_name=model_name, history_size=10)
        self.optimizer = None

    def set_reduce_method(self, reduce_type, reduce_mode=None):
        if type(reduce_mode) != list and type(reduce_mode) != tuple:
            reduce_mode = [reduce_mode]
        if reduce_type is None or reduce_type == "data":
            reduce_type = "data"
            self.forward = self._forward_simple
        elif reduce_type == "features":
            reduce_dict = {"max": lambda *args, **kwargs: torch.max(*args, **kwargs).values, "mean": torch.mean,
                           "median": lambda *args, **kwargs: torch.median(*args, **kwargs).values, "prod": torch.prod,
                           "std": torch.std, "sum": torch.sum}
            self.features_reducers = [reduce_dict[rm] for rm in reduce_mode]
            self.forward = self._forward_reducing_features
        elif reduce_type == "logits":
            reduce_dict = {
                "mean": lambda *args, **kwargs: torch.mean(*args, **kwargs)}
            assert len(reduce_mode) == 1
            self.logits_reducer = reduce_dict[reduce_mode[0]]
            self.forward = self._forward_reducing_logits
        elif reduce_type == "eval":
            def _argmax(*args, **kwargs):
                num_classes = args[0].shape[1]
                return torch.nn.functional.one_hot(args[0].argmax(dim=1).mode(keepdim=True).values, num_classes)
            reduce_dict = {
                "argmax": _argmax
            }
            assert len(reduce_mode) == 1
            self.logits_reducer = reduce_dict[reduce_mode[0]]
            self.forward = self._forward_simple
        else:
            raise ValueError("Invalid reduce type: " + reduce_type)
        self.reduce_type = reduce_type

    def current_device(self):
        return next(self.parameters()).device

    def forward(self, x):
        pass

    def _forward_simple(self, x):
        features = self.extract_features(x)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        return {"features": features, **classes, **regressions}

    def _forward_reducing_features(self, x):
        features_list = []
        for group in x:
            group_all_features = self.extract_features(group)
            group_features = torch.cat([reducer(group_all_features, dim=0) for reducer in self.features_reducers])
            features_list.append(group_features)
        features = torch.stack(features_list)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        return {"features": features, **classes, **regressions}

    def _forward_reducing_logits(self, x):
        classes = {var: torch.zeros((0, self.classifiers[var].out_features), device=self.current_device()) for var, classifier in self.classifiers.items()}
        regressions = {var: torch.zeros((0, self.regressor[var].out_features), device=self.current_device()) for var, regressor in self.regressors.items()}
        reduce_fn = self.logits_reducer
        for group in x:
            group_all_features = self.extract_features(group)
            group_classes = {var: reduce_fn(classifier(group_all_features), dim=0, keepdim=True) for var, classifier in self.classifiers.items()}
            classes = {var: torch.cat([classes[var], group_classes[var]]) for var in classes}
        return {**classes, **regressions}

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

    def fit(self, train_data, train_labels, val_data=None, val_labels=None, info={}, **hyperparams):
        logger = self.logger
        try:
            self.update_losses(train_labels, val_labels)
            # train_size, val_size = 8192, 8192
            # train_data, train_labels = train_data[:train_size], train_labels.iloc[:train_size]
            # val_data, val_labels = val_data[:val_size], val_labels.iloc[:val_size]

            batch_size = hyperparams["batch_size"]
            self.activation_penalty = hyperparams["activation_penalty"]

            if self.reduce_type is None or self.reduce_type == "data":
                train_data = [torch.tensor(report, device=self.current_device()) for report in train_data]
                if val_data is not None:
                    val_data = [torch.tensor(report, device=self.current_device()) for report in val_data]
            else:
                if self.reduce_type == "eval":
                    train_data = [torch.tensor(report, device=self.current_device()) for report in train_data]
                else:
                    train_data = [[torch.tensor(report, device=self.current_device()) for report in record] for record in train_data]
                if val_data is not None:
                    val_data = [[torch.tensor(report, device=self.current_device()) for report in record] for record in val_data]

            self.optimizer = Adam(self.parameters(), lr=hyperparams["learning_rate"])

            num_batches, num_val_batches = len(train_data) // batch_size, len(val_data) // batch_size
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
        finally:
            logger.log_hyper_parameters_and_best_metrics(info, hyperparams)
            closed_logger = logger.close()

        return closed_logger

    def train_step(self, data, labels, training=True):

        if training:
            self.optimizer.zero_grad()
            self.train()
            torch.set_grad_enabled(True)
            forwarded = self(data)
        else:
            self.eval()
            torch.set_grad_enabled(False)
            if self.reduce_type == "eval":
                forwarded = self._forward_reducing_logits(data)
            else:
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
            losses[var] = loss.detach().item()
            total_loss += loss
            preds_classes = torch.argmax(preds.detach(), axis=1)
            accuracies[var] = (preds_classes == grth).float().mean().detach().item()
            macro_f1[var] = [preds_classes.cpu().numpy(), grth.cpu().numpy()]
            cks[var] = [preds_classes.cpu().numpy(), grth.cpu().numpy()]
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

        if "features" in forwarded and self.activation_penalty != 0:
            total_loss += self.activation_penalty * forwarded["features"].abs().sum()
        if training:
            total_loss.backward()  # TODO: address loss scale
            self.optimizer.step()
        torch.set_grad_enabled(False)
        return {"Loss": losses, "Accuracy": accuracies, "MAE": maes, "M-F1": macro_f1, "CKS": cks, "DBCS": dbcs}

    def add_classification(self, task_name, num_classes):
        self.classifiers[task_name] = nn.Linear(self.deep_features, num_classes).to(self.current_device())

    def add_regression(self, task_name):
        self.regressors[task_name] = nn.Linear(self.deep_features, 1).to(self.current_device())

    @DeprecationWarning
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
