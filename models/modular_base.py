from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW

from callbacks.base_callback import Callback
from metrics.accuracy import Accuracy
from metrics.average import Average
from metrics.cks import CohenKappaScore
from metrics.dbcs import DumbBaselineComparisonScore
from metrics.mae import MeanAverageError
from metrics.metrics import Metrics
from metrics.mf1 import MacroF1Score
from metrics.nmae import NormalizedMeanAverageError


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
        self.train_stds = {}
        self.val_stds = {}

        self.optimizer = None

        self.classification_metrics = ["Accuracy", "M-F1", "CKS", "DBCS"]
        self.regression_metrics = ["MAE", "NMAE"]

        self.tokens_pooler = None
        self.reports_pooler = None
        self.predictions_pooler = {}

        self.reports_processor = None
        self.set_reports_transformation_method("identity")

    def encode_report(self, report):
        return self.token_codec.encode(self.tokenizer.tokenize(self.preprocessor.preprocess(report)))

    def set_tokens_pooling_method(self, pool_mode: str, **pool_args):
        if pool_mode is None:
            print("set_tokens_pooling_method skipped: pool_mode is None")
        elif pool_mode == "max":
            def _max(x, explain=False):
                x_max = x.max(dim=2, keepdim=True).values
                tokens_importance = None
                if explain:
                    tokens_absolute_importance = (x == x_max).sum(dim=3)
                    tokens_importance = tokens_absolute_importance.float() / tokens_absolute_importance.sum(dim=2, keepdim=True)
                    tokens_importance[x.abs().sum(dim=3) == 0] = 0
                return x_max, tokens_importance
            self.tokens_pooler = _max
        else:
            raise ValueError("Unknown tokens pooling mode: " + pool_mode)

    def set_reports_pooling_method(self, pool_mode: str, **pool_args):
        if pool_mode is None:
            print("set_reports_pooling_method skipped: pool_mode is None")
        elif pool_mode == "max":
            def _max(x, explain=False):
                x_max = x.max(dim=1, keepdim=True).values
                reports_importance = None
                if explain:
                    reports_absolute_importance = (x == x_max).sum(dim=3)
                    reports_importance = reports_absolute_importance.float() / reports_absolute_importance.sum(dim=1, keepdim=True)
                    reports_importance[x.abs().sum(dim=3) == 0] = 0
                    reports_importance = reports_importance.squeeze(2)
                return x_max, reports_importance
            self.reports_pooler = _max
        else:
            raise ValueError("Unknown reports pooling mode: " + str(pool_mode))

    def set_predictions_pooling_method(self, pool_mode: str, **pool_args):  # TODO: test
        if pool_mode == "argmax":
            self.predictions_pooler[pool_args['var']] = lambda x: torch.nn.functional.one_hot(x.argmax(dim=3).mode(keepdim=True).values, x.shape[3])
        if pool_mode == "mean":
            self.predictions_pooler[pool_args['var']] = lambda x: x.mean(dim=3, keepdim=True)
        else:
            raise ValueError("Unknown predictions pooling mode: " + pool_mode)

    def set_reports_transformation_method(self, process_mode: str, **transformation_args):
        if process_mode == "identity":
            self.reports_processor = lambda x: x
        elif process_mode == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(128, 2, 32, 0.1)
            self.reports_processor_net = nn.TransformerEncoder(encoder_layer, 1).to(self.current_device())
            self.reports_processor = lambda x: self.reports_processor_net(x.squeeze(2).permute(1, 0, 2), src_key_padding_mask=x.squeeze(2).abs().sum(dim=2) == 0).permute(1, 0, 2).unsqueeze(2)
        else:
            raise ValueError("Unknown reports processing mode: " + process_mode)

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
        out = {}
        x_orig_shape = x.shape
        # x.shape: (num_records, num_reports, num_tokens)
        x, not_padded = self.pre_feature_extraction(x)
        # x.shape: (tot_non_padding_reports, num_tokens)
        features = self.extract_features(x)
        features[x == 0] = 0
        # features.shape: (tot_non_padding_reports, num_tokens, num_features)
        features = self.post_feature_extraction(features, not_padded, x_orig_shape)
        # features.shape: (num_records, num_reports, num_tokens, num_features)
        if pool_tokens:
            features, tokens_importance = self.tokens_pooler(features, explain)
            # features.shape: (num_records, num_reports, 1, num_features)
            # tokens_importance.shape: (num_records, num_reports, num_tokens)
            out.update({"tokens_importance": tokens_importance})
            features = self.reports_processor(features)
            # features.shape: (num_records, num_reports, 1, num_features)
            if pool_reports:
                features, reports_importance = self.reports_pooler(features, explain)
                # features.shape: (num_records, 1, 1, num_features)
                # reports_importance.shape: (num_records, num_reports)
                out.update({"reports_importance": reports_importance})
        # features /= features.norm(dim=3)
        classes = {var: classifier(features) for var, classifier in self.classifiers.items()}
        # classifier(features).shape: (num_records, num_reports, num_tokens, num_classes)
        regressions = {var: regressor(features) for var, regressor in self.regressors.items()}
        # regressor(features).shape: (num_records, num_reports, num_tokens, 1)
        if pool_predictions:
            classes.update({var: self.predictions_pooler[var](classes[var]) for var in self.classifiers if var in self.predictions_pooler})
            regressions.update({var: self.predictions_pooler[var](regressions[var]) for var in self.regressors if var in self.predictions_pooler})
        out.update({"features": features, **classes, **regressions})
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
            classes_weights = 1 / classes_occurrences
            classes_weights = torch.from_numpy(classes_weights).float().to(self.current_device())
            self.losses[var] = nn.CrossEntropyLoss(classes_weights)
            self.dumb_baseline_train_accuracy[var] = max(classes_occurrences) / sum(classes_occurrences)
            y_val = val_labels[var].dropna()
            self.dumb_baseline_val_accuracy[var] = (y_val == np.argmax(classes_occurrences)).sum() / len(y_val)
        for var in self.regressors:
            self.losses[var] = nn.MSELoss()  # TODO: multiply by a weight
            self.train_stds[var] = train_labels[var].dropna().values.std()
            self.val_stds[var] = val_labels[var].dropna().values.std()

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
            self.classifiers_l2_penalty = hyperparams["classifiers_l2_penalty"]
            self.regressors_l2_penalty = hyperparams["regressors_l2_penalty"]

            train_data = torch.tensor(train_data.astype(np.int16), device=self.current_device())
            val_data = torch.tensor(val_data.astype(np.int16), device=self.current_device())
            # torch does not have uint16 dtype: be aware that indices >= 32768 will be stored as negative numbers

            self.optimizer = Adam(self.parameters(), lr=hyperparams["learning_rate"])

            num_batches, num_val_batches = len(train_data) // batch_size, len(val_data) // batch_size
            train_metrics = Metrics({**self.create_losses_metrics(), **self.create_classifications_metrics(True), **self.create_regressions_metrics(True), **self.create_grad_norm_metrics()})
            val_metrics = Metrics({**self.create_losses_metrics(), **self.create_classifications_metrics(False), **self.create_regressions_metrics(False)})
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

        old_grad_vector = self.grad_vector()
        gradient_norms = {} # TODO: removable, it is used only during debug

        for var in self.classifiers:
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask].squeeze(2).squeeze(1) # TODO: mask indexing is slow
            grth = torch.tensor(labels[var][mask].to_list(), dtype=torch.long, device=device, requires_grad=False)
            loss = self.losses[var](preds, grth) / len(grth)
            if self.classifiers_l2_penalty > 0:
                loss += self.classifiers_l2_penalty * sum(p.norm() for p in self.classifiers[var].parameters())
            losses[var] = loss.detach().item()

            if training:
                loss.backward(retain_graph=True)
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                old_grad_vector = new_grad_vector
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
            if self.regressors_l2_penalty > 0:
                loss += self.regressors_l2_penalty * sum(p.norm() for p in self.regressors[var].parameters())
            losses[var] = loss.detach().item()

            if training:
                loss.backward(retain_graph=True)
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                old_grad_vector = new_grad_vector
                gradient_norms[var] = gradient_norm
                metrics["GradNorm"][var].update(gradient_norm, num_batches)

            metrics["Loss"][var].update(losses[var], num_batches)
            for metric_name in self.regression_metrics:
                metrics[metric_name][var].update(preds.cpu().detach().numpy(), grth.cpu().numpy())

        if self.activation_penalty != 0:
            regularization_loss = self.activation_penalty * forwarded["features"].abs().sum()
            if training:
                regularization_loss.backward()
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                gradient_norms["features_regularization_l1"] = gradient_norm

    def add_classification(self, task_name, num_classes, dropout):
        self.classifiers[task_name] = nn.Sequential(
            nn.Dropout(dropout).to(self.current_device()),
            nn.Linear(self.deep_features, num_classes).to(self.current_device())
        )
        for param in self.classifiers[task_name].parameters():
            param.requires_grad = False

    def add_regression(self, task_name, dropout):
        self.regressors[task_name] = nn.Sequential(
            nn.Dropout(dropout).to(self.current_device()),
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

    def create_regressions_metrics(self, training: bool):
        stds = self.train_stds if training else self.val_stds
        return {
            "MAE": {var: MeanAverageError() for var in self.regressors},
            "NMAE": {var: NormalizedMeanAverageError(stds[var]) for var in self.regressors}
        }

    def create_grad_norm_metrics(self):
        return {
            "GradNorm": {var: Average(max) for var in list(self.classifiers.keys()) + list(self.regressors.keys())}
        }

    def grad_vector(self):
        device = self.current_device()
        return torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.shape, device=device).flatten() for p in self.parameters()])

