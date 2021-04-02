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
    def __init__(self, modules_dict, deep_features, model_name, preprocessor, tokenizer, token_codec, labels_codec, *args, **kwargs):
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

        self.optimizer = None

        self.classification_metrics = ["Accuracy", "Accuracy", "M-F1", "CKS", "DBCS"]
        self.regression_metrics = ["MAE"]

        self.tokens_pooler = None
        self.reports_pooler = None
        self.predictions_pooler = {}

    def encode_report(self, report):
        return self.token_codec.encode(self.tokenizer.tokenize(self.preprocessor.preprocess(report)))

    def set_tokens_pooling_method(self, pool_mode: str, **pool_args):
        if pool_mode is None:
            print("set_tokens_pooling_method skipped: pool_mode is None")
        elif pool_mode == "max":
            self.tokens_pooler = lambda x: x.max(dim=2, keepdim=True).values
        else:
            raise ValueError("Unknown tokens pooling mode: " + pool_mode)

    def set_reports_pooling_method(self, pool_mode: str, **pool_args):
        if pool_mode is None:
            print("set_reports_pooling_method skipped: pool_mode is None")
        elif pool_mode == "max":
            self.reports_pooler = lambda x: x.max(dim=1, keepdim=True).values
        else:
            raise ValueError("Unknown reports pooling mode: " + str(pool_mode))

    def set_predictions_pooling_method(self, pool_mode: str, **pool_args):  # TODO: test
        if pool_mode == "argmax":
            self.predictions_pooler[pool_args['var']] = lambda x: torch.nn.functional.one_hot(x.argmax(dim=3).mode(keepdim=True).values, x.shape[3])
        if pool_mode == "mean":
            self.predictions_pooler[pool_args['var']] = lambda x: x.mean(dim=3, keepdim=True)
        else:
            raise ValueError("Unknown predictions pooling mode: " + pool_mode)

    def current_device(self):
        return next(self.parameters()).device

    def pre_feature_extraction(self, x):
        x = x.flatten(start_dim=0, end_dim=1)
        x = x.long()
        x[x < 0] += 65536
        not_padded = (x.sum(axis=1) != 0)
        return x[not_padded], not_padded

    @abstractmethod
    def extract_features(self, x):
        pass

    def post_feature_extraction(self, x, not_padded, x_orig_shape):
        out = torch.zeros((len(not_padded), x_orig_shape[2], x.shape[-1]), device=self.current_device())
        out[not_padded] = x
        return out.reshape((*x_orig_shape, -1))

    def forward(self, x, pool_tokens=True, pool_reports=True, pool_predictions=False, explain=False):
        x_orig_shape = x.shape
        # x.shape: (num_records, num_reports, num_tokens)
        x, not_padded = self.pre_feature_extraction(x)
        # x.shape: (tot_non_padding_reports, num_tokens)
        features = self.extract_features(x)
        # x.shape: (tot_non_padding_reports, num_tokens, num_features)
        features = self.post_feature_extraction(features, not_padded, x_orig_shape)
        # features.shape: (num_records, num_reports, num_tokens, num_features)
        if pool_tokens:
            features = self.tokens_pooler(features)
            # features.shape: (num_records, num_reports, 1, num_features)
            if pool_reports:
                features = self.reports_pooler(features)
                # features.shape: (num_records, 1, 1, num_features)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        # classifier(features).shape: (num_records, num_reports, num_tokens, num_classes)
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        # regressor(features).shape: (num_records, num_reports, num_tokens, 1)
        if pool_predictions:
            classes.update({var: self.predictions_pooler[var](classes[var]) for var in self.classifiers if var in self.predictions_pooler})
            regressions.update({var: self.predictions_pooler[var](regressions[var]) for var in self.regressors if var in self.predictions_pooler})
        out = {"features": features, **classes, **regressions}
        if explain:
            out.update({}) # TODO: finish
        return out

    def init_losses(self, train_labels, val_labels):
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
            self.init_losses(train_labels, val_labels)
            # train_size, val_size = 1024, 1024
            # train_data, train_labels = train_data[:train_size], train_labels.iloc[:train_size]
            # val_data, val_labels = val_data[:val_size], val_labels.iloc[:val_size]

            batch_size = hyperparams["batch_size"]
            self.activation_penalty = hyperparams["activation_penalty"]

            train_data = torch.tensor(train_data.astype(np.int16), device=self.current_device())
            val_data = torch.tensor(val_data.astype(np.int16), device=self.current_device())
            # torch does not have uint16 dtype: be aware that indices >= 32768 will be stored as negative numbers

            self.optimizer = Adam(self.parameters(), lr=hyperparams["learning_rate"])

            num_batches, num_val_batches = len(train_data) // batch_size, len(val_data) // batch_size
            train_metrics = Metrics({**self.create_losses_metrics(), **self.create_classifications_metrics(True), **self.create_regressions_metrics(), **self.create_grad_norm_metrics()})
            val_metrics = Metrics({**self.create_losses_metrics(), **self.create_classifications_metrics(False), **self.create_regressions_metrics()})
            for epoch in range(hyperparams["max_epochs"]):
                [c.on_epoch_start(self, epoch) for c in callbacks]
                train_metrics.reset(), val_metrics.reset()

                # shuffle dataset
                perm = np.random.permutation(len(train_data))
                train_data = train_data[perm]
                train_labels = train_labels.iloc[perm].reset_index(drop=True)

                # train epoch
                [c.on_train_epoch_start(self, epoch) for c in callbacks]
                for b in range(num_batches):
                    [c.on_train_batch_start(self) for c in callbacks]
                    batch = train_data[b * batch_size: (b + 1) * batch_size]
                    batch_labels = train_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                    self.train_step(batch, batch_labels, num_batches, metrics=train_metrics.metrics)
                    [c.on_train_batch_end(self, train_metrics.metrics) for c in callbacks]
                [c.on_train_epoch_end(self, epoch, train_metrics.metrics) for c in callbacks]

                if val_data is not None:
                    # validation epoch
                    [c.on_validation_epoch_start(self, epoch) for c in callbacks]
                    for b in range(num_val_batches):
                        [c.on_validation_batch_start(self) for c in callbacks]
                        batch = val_data[b * batch_size: (b + 1) * batch_size]
                        batch_labels = val_labels.iloc[b * batch_size: (b + 1) * batch_size].reset_index()
                        self.validation_step(batch, batch_labels, num_val_batches, metrics=val_metrics.metrics)
                        [c.on_validation_batch_end(self, val_metrics.metrics) for c in callbacks]
                    [c.on_validation_epoch_end(self, epoch, val_metrics.metrics) for c in callbacks]

                [c.on_epoch_end(self, epoch) for c in callbacks]
        finally:
            [c.on_fit_end(self) for c in callbacks]

    def train_step(self, data, labels, num_batches, metrics={}):
        self.optimizer.zero_grad()
        self.train()
        torch.set_grad_enabled(True)
        self.step(data, labels, num_batches, metrics, True)
        self.optimizer.step()

    def validation_step(self, data, labels, num_batches, metrics={}):
        self.eval()
        torch.set_grad_enabled(False)
        self.step(data, labels, num_batches, metrics, False)

    def step(self, data, labels, num_batches, metrics={}, training=True):
        forwarded = self.forward(data, pool_tokens=self.tokens_pooler is not None, pool_reports=self.reports_pooler is not None)

        losses = {}
        device = self.current_device()

        total_gradient_norm = 0
        gradient_norms = {} # TODO: removable, it is used only during debug

        for var in self.classifiers:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask].squeeze(2).squeeze(1) # TODO: mask indexing is slow
            grth = torch.tensor(labels[var][mask].to_list(), dtype=torch.long, device=device, requires_grad=False)
            loss = self.losses[var](preds, grth) / len(grth)
            losses[var] = loss.detach().item()

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
            preds = forwarded[var][mask].squeeze(3).squeeze(2).squeeze(1)
            grth = torch.tensor(labels[var][mask].to_list(), device=device, requires_grad=False)
            loss = self.losses[var](preds, grth) / len(grth)
            losses[var] = loss.detach().item()

            if training:
                loss.backward(retain_graph=True)
                gradient_norm = self.grad_norm() - total_gradient_norm
                total_gradient_norm += gradient_norm
                gradient_norms[var] = gradient_norm
                metrics["GradNorm"][var].update(gradient_norm, num_batches)

            metrics["Loss"][var].update(losses[var], num_batches)
            for metric_name in self.regression_metrics:
                metrics[metric_name][var].update(preds.cpu().detach().numpy(), grth.cpu().numpy())

        if "features" in forwarded and self.activation_penalty != 0:
            regularization_loss = self.activation_penalty * forwarded["features"].abs().sum()
            if training:
                regularization_loss.backward()
                gradient_norm = self.grad_norm() - total_gradient_norm
                total_gradient_norm += gradient_norm
                gradient_norms["features_regularization_l1"] = gradient_norm

    def add_classification(self, task_name, num_classes):
        self.classifiers[task_name] = nn.Linear(self.deep_features, num_classes).to(self.current_device())

    def add_regression(self, task_name):
        self.regressors[task_name] = nn.Sequential(
            nn.Linear(self.deep_features, 1).to(self.current_device()),
            nn.Sigmoid()
        )

    def __str__(self):
        str_dict = {}
        str_dict["name"] = {self.model_name}
        str_dict["tot_parameters"] = sum([p.numel() for p in self.parameters()])
        str_dict["parameters"] = {}
        for parameter_name, parameter in self.named_parameters():
            str_dict["parameters"][parameter_name] = parameter.numel()
        return str(str_dict)

    def create_losses_metrics(self):
        return {
            "Loss": {var: Average(min) for var in list(self.classifiers.keys()) + list(self.regressors.keys())}
        }

    def create_classifications_metrics(self, training: bool):
        dbcs = self.dumb_baseline_train_accuracy if training else self.dumb_baseline_val_accuracy
        return {
            "Accuracy": {var: Accuracy() for var in self.classifiers},
            "M-F1": {var: MacroF1Score() for var in self.classifiers},
            "CKS": {var: CohenKappaScore() for var in self.classifiers},
            "DBCS": {var: DumbBaselineComparisonScore(dbcs[var]) for var in self.classifiers}
        }

    def create_regressions_metrics(self):
        return {
            "MAE": {var: MeanAverageError() for var in self.regressors}
        }

    def create_grad_norm_metrics(self):
        return {
            "GradNorm": {var: Average(max) for var in list(self.classifiers.keys()) + list(self.regressors.keys())}
        }

    def grad_norm(self):
        return sum(p.grad.detach().norm() for p in list(self.parameters()) if p.grad is not None).item()


if __name__ == "__main__":
    # this was an old version, it doesn't work anymore
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
