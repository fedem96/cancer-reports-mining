from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

from callbacks.base_callback import Callback
from metrics.accuracy import Accuracy
from metrics.average import Average
from metrics.cks import CohenKappaScore
from metrics.dbcs import DumbBaselineComparisonScore
from metrics.mae import MeanAverageError
from metrics.metrics import Metrics
from metrics.mf1 import MacroF1Score


class ModularBase(nn.Module, ABC):
    def __init__(self, modules_dict, deep_features, model_name, preprocessor, tokenizer, token_codec, labels_codec):
        super(ModularBase, self).__init__()
        self.__name__ = model_name
        self.net = nn.ModuleDict(OrderedDict({
            **modules_dict,
            "classifiers": nn.ModuleDict({}),
            "regressors": nn.ModuleDict({})
        }))

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.token_codec = token_codec
        self.labels_codec = labels_codec

        self.deep_features = deep_features

        self.classifiers = self.net['classifiers']
        self.regressors = self.net['regressors']

        for module_name in modules_dict:
            if hasattr(self, module_name):
                raise NameError("Name {} unavailable".format(module_name))
            setattr(self, module_name, modules_dict[module_name])

        self.losses = {}
        self.dumb_baseline_train_accuracy = {}
        self.dumb_baseline_val_accuracy = {}
        self.num_classes = {}

        self.reduce_type = "data"
        self.features_reducers = None
        self.logits_reducer = None
        self.forward = self._forward_simple

        self.optimizer = None

        self.classification_metrics = ["Accuracy", "Accuracy", "M-F1", "CKS", "DBCS"]
        self.regression_metrics = ["MAE"]

    def encode_report(self, report):
        return self.token_codec.encode(self.tokenizer.tokenize(self.preprocessor.preprocess(report)))

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
        self.info = {}
        self.hyperparameters = {}

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
        all_features = []
        features_list = []
        for group in x:
            group_all_features = self.extract_features(group)
            all_features.append(group_all_features)
            group_features = torch.cat([reducer(group_all_features, dim=0) for reducer in self.features_reducers])
            features_list.append(group_features)
        features = torch.stack(features_list)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        return {"all_features": all_features, "features": features, **classes, **regressions}

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

    def fit(self, train_data, train_labels, val_data=None, val_labels=None, info={}, callbacks: List[Callback]=[], **hyperparams):
        self.info = info
        self.hyperparameters = hyperparams
        [c.on_fit_start(self) for c in callbacks]
        try:
            self.update_losses(train_labels, val_labels)
            # train_size, val_size = 1024, 1024
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
            train_metrics = Metrics({**self.create_losses(), **self.create_classification_metrics(True), **self.create_regression_metrics(True)})
            val_metrics = Metrics({**self.create_losses(), **self.create_classification_metrics(False), **self.create_regression_metrics(False)})
            for epoch in range(hyperparams["max_epochs"]):
                [c.on_epoch_start(self, epoch) for c in callbacks]
                train_metrics.reset(), val_metrics.reset()
                perm = np.random.permutation(len(train_data))
                # perm = sorted(range(len(train_data)), key=lambda n: train_data[n].shape[0])
                train_data = [train_data[d] for d in perm]
                train_labels = train_labels.iloc[perm].reset_index(drop=True)

                [c.on_train_epoch_start(self, epoch) for c in callbacks]
                for b in range(num_batches):
                    [c.on_train_batch_start(self) for c in callbacks]
                    batch = train_data[b * batch_size: (b + 1) * batch_size]
                    batch_labels = train_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                    self.train_step(batch, batch_labels, num_batches, metrics=train_metrics.metrics)
                    [c.on_train_batch_end(self, train_metrics.metrics) for c in callbacks]
                [c.on_train_epoch_end(self, epoch, train_metrics.metrics) for c in callbacks]

                if val_data is not None:
                    [c.on_validation_epoch_start(self, epoch) for c in callbacks]
                    for b in range(num_val_batches):
                        [c.on_validation_batch_start(self) for c in callbacks]
                        batch = val_data[b * batch_size: (b + 1) * batch_size]
                        batch_labels = val_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                        self.train_step(batch, batch_labels, num_val_batches, metrics=val_metrics.metrics, training=False)
                        [c.on_validation_batch_end(self, val_metrics.metrics) for c in callbacks]
                    [c.on_validation_epoch_end(self, epoch, val_metrics.metrics) for c in callbacks]

                [c.on_epoch_end(self, epoch) for c in callbacks]
        finally:
            [c.on_fit_end(self) for c in callbacks]

    def train_step(self, data, labels, num_batches, metrics={}, training=True):

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
        device = self.current_device()

        # total_loss = torch.tensor([0.0], device=self.current_device())
        # total_loss = 0
        total_gradient_norm = 0
        gradient_norms = {}

        for var in self.classifiers:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask]
            grth = torch.tensor(labels[var][mask].to_list(), dtype=torch.long, device=device, requires_grad=False)
            loss = self.losses[var](preds, grth) / len(grth)
            losses[var] = loss.detach().item()
            # total_loss += loss
            if training:
                loss.backward(retain_graph=True)
                gradient_norm = self.grad_norm() - total_gradient_norm
                total_gradient_norm += gradient_norm
                gradient_norms[var] = gradient_norm
                metrics["GradNorm"][var].update(gradient_norm, num_batches)
            
            grth = grth.cpu().numpy()
            preds_classes = torch.argmax(preds.detach(), dim=1).cpu().numpy()

            metrics["Loss"][var].update(losses[var], num_batches)
            for metric_name in self.classification_metrics:
                metrics[metric_name][var].update(preds_classes, grth)

        for var in self.regressors:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask].squeeze(1)
            grth = torch.tensor(labels[var][mask].to_list(), device=device, requires_grad=False)
            loss = self.losses[var](preds, grth) / len(grth)
            losses[var] = loss.detach().item()
            # total_loss += loss
            if training:
                loss.backward(retain_graph=True)
                gradient_norm = self.grad_norm() - total_gradient_norm
                total_gradient_norm += gradient_norm
                gradient_norms[var] = gradient_norm
                metrics["GradNorm"][var].update(gradient_norm, num_batches)

            metrics["Loss"][var].update(losses[var], num_batches)
            for metric_name in self.regression_metrics.keys():
                metrics[metric_name][var].update(preds.cpu().numpy(), grth.cpu().numpy())

        if "features" in forwarded and self.activation_penalty != 0:
            # total_loss += self.activation_penalty * forwarded["features"].abs().sum()
            regularization_loss = self.activation_penalty * forwarded["features"].abs().sum()
            if training:
                regularization_loss.backward()
                gradient_norm = self.grad_norm() - total_gradient_norm
                total_gradient_norm += gradient_norm
                gradient_norms["features_regularization_l1"] = gradient_norm
        if training:
            # total_loss.backward()  # TODO: address loss scale
            self.optimizer.step()
        return None

    def add_classification(self, task_name, num_classes):
        self.classifiers[task_name] = nn.Linear(self.deep_features, num_classes).to(self.current_device())

    def add_regression(self, task_name):
        self.regressors[task_name] = nn.Linear(self.deep_features, 1).to(self.current_device())

    def __str__(self):
        str_dict = {}
        str_dict["name"] = {self.model_name}
        str_dict["tot_parameters"] = sum([p.numel() for p in self.parameters()])
        str_dict["parameters"] = {}
        for parameter_name, parameter in self.named_parameters():
            str_dict["parameters"][parameter_name] = parameter.numel()
        return str(str_dict)

    def create_losses(self):
        return {
            "Loss": {var: Average(min) for var in list(self.classifiers.keys()) + list(self.regressors.keys())}
        }

    def create_classification_metrics(self, training: bool):
        dbcs = self.dumb_baseline_train_accuracy if training else self.dumb_baseline_val_accuracy
        metrics = {
            "Accuracy": {var: Accuracy() for var in self.classifiers},
            "M-F1": {var: MacroF1Score() for var in self.classifiers},
            "CKS": {var: CohenKappaScore() for var in self.classifiers},
            "DBCS": {var: DumbBaselineComparisonScore(dbcs[var]) for var in self.classifiers}
        }
        if training:
            metrics["GradNorm"] = {var: Average(max) for var in self.classifiers}
        return metrics

    def create_regression_metrics(self, training):
        metrics = {
            "MAE": {var: MeanAverageError() for var in self.regressors}
        }
        if training:
            metrics["GradNorm"] = {var: Average(max) for var in self.classifiers}
        return metrics

    def grad_norm(self):
        return sum(p.grad.norm() for p in list(self.parameters()) if p.grad is not None).item()

    # def infer(self, x):
    #     with torch.no_grad():
    #         if type(x) == list or type(x) == tuple:
    #             x = [torch.tensor(example, device=self.current_device()) for example in x]
    #             predictions = self(x)
    #         else:
    #             x = torch.tensor(x, device=self.current_device())
    #             predictions = self([x])
    #     return predictions


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
