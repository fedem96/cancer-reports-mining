from abc import abstractmethod, ABC
from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import StepLR

from callbacks.base_callback import Callback
from layers.gradient_reversal import GradientReversal
from metrics.accumulate_predictions import PredictionsAccumulator
from metrics.accuracy import Accuracy
from metrics.average import Average
from metrics.cks import CohenKappaScore
from metrics.dbcs import DumbBaselineComparisonScore
from metrics.f1 import F1Score
from metrics.mae import MeanAbsoluteError
from metrics.metrics import Metrics
from metrics.m_f1 import MacroF1Score
from metrics.nmae import NormalizedMeanAbsoluteError
from metrics.precision import Precision
from metrics.recall import Recall
from utils.chrono import Chronometer


class ModularBase(nn.Module, ABC):
    def __init__(self, modules_dict, deep_features, model_name, preprocessor, tokenizer, labels_codec, *args, **kwargs):
        super(ModularBase, self).__init__()
        self.__name__ = model_name
        self.net = nn.ModuleDict(OrderedDict({
            **modules_dict,
            "predictors": nn.ModuleDict({})
        }))

        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.labels_codec = labels_codec

        self.deep_features = deep_features

        self.predictors = self.net['predictors']
        self.predictors_l2_penalty = {}

        for module_name in modules_dict:
            if hasattr(self, module_name):
                raise NameError("Name {} unavailable".format(module_name))
            setattr(self, module_name, modules_dict[module_name])

        self.losses = {}
        self.num_classes = {}
        self.train_stds = {}
        self.val_stds = {}

        self.optimizer = None

        self.tokens_pooler = None
        self.reports_pooler = None
        self.predictions_pooler = {}

        self.reports_processor = None
        self.set_reports_transformation_method("identity")

        self.training_tasks = {}
        self.validation_tasks = {}

    def encode_report(self, report):
        return self.tokenizer.tokenize(self.preprocessor.preprocess(report), encode=True)

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
            encoder_layer = nn.TransformerEncoderLayer(**transformation_args)
            self.reports_processor_net = nn.TransformerEncoder(encoder_layer, 1).to(self.current_device())
            self.reports_processor = lambda x: self.reports_processor_net(x.squeeze(2).permute(1, 0, 2), src_key_padding_mask=x.squeeze(2).abs().sum(dim=2) == 0).permute(1, 0, 2).unsqueeze(2)
        else:
            raise ValueError("Unknown reports processing mode: " + process_mode)

    def current_device(self):
        return next(self.parameters()).device

    def pre_feature_extraction(self, x):
        x = x.flatten(start_dim=0, end_dim=1)
        if x.dtype == torch.int16:
            x = x.long()
            x[x < 0] += 65536
        not_padded = (x.sum(axis=1) != 0)
        return x[not_padded], not_padded

    @abstractmethod
    def extract_features(self, x):
        pass

    def post_feature_extraction(self, x, not_padded, x_orig_shape):
        out = torch.zeros((len(not_padded), x.shape[1], x.shape[-1]), device=self.current_device())
        out[not_padded] = x
        return out.reshape((*x_orig_shape[:2], *x.shape[-2:]))

    def forward(self, x, pool_tokens=True, pool_reports=True, pool_predictions=False, explain=False):
        out = {}
        x_orig_shape = x.shape
        # x.shape: (num_records, num_reports, num_tokens)
        x, not_padded = self.pre_feature_extraction(x)
        # x.shape: (tot_non_padding_reports, num_tokens)
        features = self.extract_features(x)
        features[(x != 0).sum(axis=1) == 0] = 0
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
        predictions = {var: classifier(features) for var, classifier in self.predictors.items()}
        # classifier(features).shape: (num_records, num_reports, num_tokens, num_classes)
        if pool_predictions:
            predictions.update({var: self.predictions_pooler[var](predictions[var]) for var in self.classifiers if var in self.predictions_pooler})
        out.update({"features": features, **predictions})
        if explain:
            out.update({}) # TODO: finish
        return out

    def init_losses(self, train_labels, val_labels):
        self.losses = {}
        self.num_classes = {}
        for var in filter(lambda t: "classification" in self.training_tasks[t]["type"], self.training_tasks):
            classes_occurrences = train_labels[var].value_counts().sort_index().to_numpy().astype(int)
            self.num_classes[var] = len(classes_occurrences)
            assert self.num_classes[var] == max(train_labels[var].dropna().unique()) + 1
            classes_weights = 1 / classes_occurrences
            classes_weights = torch.from_numpy(classes_weights).float().to(self.current_device())
            self.losses[var] = nn.CrossEntropyLoss(classes_weights)
            self.training_tasks[var]["highest_frequency"] = max(classes_occurrences) / sum(classes_occurrences)
            if var in val_labels:
                y_val = val_labels[var].dropna()
                self.validation_tasks[var]["highest_frequency"] = (y_val == np.argmax(classes_occurrences)).sum() / len(y_val)
        for var in filter(lambda t: "regression" in self.training_tasks[t]["type"], self.training_tasks):
            self.losses[var] = nn.MSELoss()  # TODO: multiply by a weight
            self.training_tasks[var]["std"] = train_labels[var].dropna().values.std()
            if var in val_labels:
                self.validation_tasks[var]["std"] = val_labels[var].dropna().values.std()

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
            self.l2_penalty = hyperparams["l2_penalty"]
            self.classifiers_l2_penalty = hyperparams["classifiers_l2_penalty"]
            self.regressors_l2_penalty = hyperparams["regressors_l2_penalty"]

            train_data = self.convert(train_data)
            val_data = self.convert(val_data)

            self.optimizer = Adam(self.parameters(), lr=hyperparams["learning_rate"])

            num_batches, num_val_batches = (len(train_data) + batch_size - 1) // batch_size, (len(val_data) + batch_size - 1) // batch_size
            train_metrics = Metrics({**self.create_losses_metrics(self.training_tasks), **self.create_metrics(self.training_tasks), **self.create_grad_norm_metrics()})
            val_metrics = Metrics({**self.create_losses_metrics(self.validation_tasks), **self.create_metrics(self.validation_tasks)})

            # scheduler = StepLR(self.optimizer, step_size=100, gamma=0.1) # TODO: get from argparse (use as callback?)
            for epoch in range(hyperparams["max_epochs"]):
                [c.on_epoch_start(self, epoch) for c in callbacks]
                train_metrics.reset(), val_metrics.reset()

                # shuffle dataset
                train_perm = np.random.permutation(len(train_data))
                # val_perm = np.random.permutation(len(val_data))         # even if shuffling the validation is not required, having a permutation is useful when using sparse data
                val_perm = np.arange(len(val_data))         # even if shuffling the validation is not required, having a permutation is useful when using sparse data

                # train epoch
                [c.on_train_epoch_start(self, epoch) for c in callbacks]
                for b in range(num_batches):
                    [c.on_train_batch_start(self) for c in callbacks]
                    batch, batch_labels = self.get_batch(train_data, train_labels, train_perm[b * batch_size: (b + 1) * batch_size])
                    self.train_step(batch, batch_labels, num_batches, metrics=train_metrics.metrics)
                    [c.on_train_batch_end(self, train_metrics.metrics) for c in callbacks]
                [c.on_train_epoch_end(self, epoch, train_metrics.metrics) for c in callbacks]

                if val_data is not None:
                    # validation epoch
                    [c.on_validation_epoch_start(self, epoch) for c in callbacks]
                    for b in range(num_val_batches):
                        [c.on_validation_batch_start(self) for c in callbacks]
                        batch, batch_labels = self.get_batch(val_data, val_labels, val_perm[b * batch_size: (b + 1) * batch_size])
                        self.validation_step(batch, batch_labels, num_val_batches, metrics=val_metrics.metrics)
                        [c.on_validation_batch_end(self, val_metrics.metrics) for c in callbacks]
                    [c.on_validation_epoch_end(self, epoch, val_metrics.metrics) for c in callbacks]

                [c.on_epoch_end(self, epoch) for c in callbacks]
                #scheduler.step()
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

        for var in self.predictors.keys():
            if var not in labels:
                continue
            mask = ~labels[var].isnull()
            if mask.sum() == 0:
                continue
            preds = forwarded[var][mask].squeeze()
            if preds.ndim == 0:
                preds = preds.unsqueeze(0).unsqueeze(0)
            elif preds.ndim == 1:
                preds = preds.unsqueeze(0)
            grth = torch.tensor(labels[var][mask].to_list(), device=device, requires_grad=False)
            if "classification" in self.training_tasks[var]["type"]:
                grth = grth.long()
            loss = self.losses[var](preds, grth) / len(grth)
            if self.training_tasks[var]['l2_penalty'] > 0:
                loss += self.training_tasks[var]['l2_penalty'] * sum(p.norm() for p in self.predictors[var].parameters())
            losses[var] = loss.detach().item()

            if training:
                loss.backward(retain_graph=True)
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                old_grad_vector = new_grad_vector
                gradient_norms[var] = gradient_norm
                metrics["GradNorm"][var].update(gradient_norm, num_batches)

            if "classification" in self.training_tasks[var]["type"]:
                preds = torch.argmax(preds.detach(), dim=1)

            metrics["Loss"][var].update(losses[var], num_batches)
            for metric_name, metric in metrics.items():
                if metric_name not in {"Loss", "GradNorm"}:
                    if var in metric:
                        metric[var].update(preds.cpu().detach().numpy(), grth.cpu().numpy())
                    else:
                        for cls in range(self.num_classes[var]):
                            if var + "_" + str(cls) in metric:
                                metric[var + "_" + str(cls)].update(preds.cpu().detach().numpy(), grth.cpu().numpy())

        if self.l2_penalty != 0:
            l2_regularization_loss = self.l2_penalty * sum(param.norm() for name, param in self.named_parameters() if "predictors" not in name)
            if training:
                l2_regularization_loss.backward()
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                gradient_norms["weights_l2"] = gradient_norm
                metrics["GradNorm"]["weights_l2"].update(gradient_norm, num_batches)

        if self.activation_penalty != 0:
            activation_regularization_loss = self.activation_penalty * forwarded["features"].abs().sum()
            if training:
                activation_regularization_loss.backward()
                new_grad_vector = self.grad_vector()
                gradient_norm = (new_grad_vector - old_grad_vector).norm().item()
                gradient_norms["features_l1"] = gradient_norm
                metrics["GradNorm"]["features_l1"].update(gradient_norm, num_batches)

    def add_classification(self, task_name, num_classes, dropout, l2_penalty, training_only=False):
        self.predictors[task_name] = nn.Sequential(
            nn.Dropout(dropout).to(self.current_device()),
            nn.Linear(self.deep_features, num_classes).to(self.current_device())
        )
        task = {"name": task_name, "type": "classification", "num_classes": num_classes, "l2_penalty": l2_penalty}
        self.training_tasks[task_name] = task
        if not training_only:
            self.validation_tasks[task_name] = task

    def add_regression(self, task_name, dropout, l2_penalty, training_only=False):
        self.predictors[task_name] = nn.Sequential(
            nn.Dropout(dropout).to(self.current_device()),
            nn.Linear(self.deep_features, 1).to(self.current_device()),
            nn.Sigmoid()
        )
        task = {"name": task_name, "type": "regression", "l2_penalty": l2_penalty}
        self.training_tasks[task_name] = task
        if not training_only:
            self.validation_tasks[task_name] = task

    def add_anti_classification(self, task_name, num_classes, dropout, l2_penalty, l, training_only=False):
        self.predictors[task_name] = nn.Sequential(
            GradientReversal(l),
            nn.Dropout(dropout).to(self.current_device()),
            nn.Linear(self.deep_features, num_classes).to(self.current_device())
        )
        task = {"name": task_name, "type": "anti_classification", "num_classes": num_classes, "l2_penalty": l2_penalty}
        self.training_tasks[task_name] = task
        if not training_only:
            self.validation_tasks[task_name] = task

    def add_anti_regression(self, task_name, dropout, l2_penalty, l, training_only=False):
        self.predictors[task_name] = nn.Sequential(
            GradientReversal(l),
            nn.Dropout(dropout).to(self.current_device()),
            nn.Linear(self.deep_features, 1).to(self.current_device())
        )
        task = {"name": task_name, "type": "anti_regression", "l2_penalty": l2_penalty}
        self.training_tasks[task_name] = task
        if not training_only:
            self.validation_tasks[task_name] = task

    def __str__(self):
        str_dict = {}
        str_dict["name"] = {self.model_name}
        str_dict["tot_parameters"] = sum([p.numel() for p in self.parameters()])
        str_dict["parameters"] = {}
        for parameter_name, parameter in self.named_parameters():
            str_dict["parameters"][parameter_name] = parameter.numel()
        return str(str_dict)

    def create_metrics(self, tasks):
        metrics_dict = {**self.create_classifications_metrics([t for t in tasks.values() if "classification" in t["type"]]),
            **self.create_regressions_metrics([t for t in tasks.values() if "regression" in t["type"]])}
        metrics_names = list(metrics_dict.keys())
        for m_name in metrics_names:
            if len(metrics_dict[m_name]) == 0:
                del metrics_dict[m_name]
        return metrics_dict

    def create_losses_metrics(self, tasks):
        return {
            "Loss": {var: Average(min) for var in tasks.keys()}
        }


    def create_detailed_metrics(self, tasks):
        metrics_dict = {**self.create_classifications_metrics([t for t in tasks.values() if "classification" in t["type"]]),
                **self.create_classifications_detailed_metrics([t for t in tasks.values() if "classification" in t["type"]]),
                **self.create_regressions_metrics([t for t in tasks.values() if "regression" in t["type"]])}
        metrics_names = list(metrics_dict.keys())
        for m_name in metrics_names:
            if len(metrics_dict[m_name]) == 0:
                del metrics_dict[m_name]
        return metrics_dict

    def create_classifications_metrics(self, tasks):
        return {
            "Accuracy": {t['name']: Accuracy() for t in tasks},
            "Precision": {t['name']: Precision() for t in tasks if t['num_classes'] == 2},
            "Recall": {t['name']: Recall() for t in tasks if t['num_classes'] == 2},
            "F1": {t['name']: F1Score() for t in tasks if t['num_classes'] == 2},
            "M-F1": {t['name']: MacroF1Score() for t in tasks if t['num_classes'] > 2},
            "CKS": {t['name']: CohenKappaScore() for t in tasks},
            "DBCS": {t['name']: DumbBaselineComparisonScore(t['highest_frequency']) for t in tasks}
        }

    def create_classifications_detailed_metrics(self, tasks):
        return {
            "Precisions": {t['name'] + "_" + str(cls): Precision(cls) for t in tasks for cls in range(t['num_classes'])},
            "Recalls": {t['name'] + "_" + str(cls): Recall(cls) for t in tasks for cls in range(t['num_classes'])},
            "F1s": {t['name'] + "_" + str(cls): F1Score(cls) for t in tasks for cls in range(t['num_classes'])}
        }

    def create_regressions_metrics(self, tasks):
        return {
            "MAE": {t['name']: MeanAbsoluteError() for t in tasks},
            "NMAE": {t['name']: NormalizedMeanAbsoluteError(t['std']) for t in tasks}
        }

    def create_grad_norm_metrics(self):
        return {
            "GradNorm": {var: Average(max) for var in list(self.predictors.keys()) + ["features_l1", "weights_l2"]}
        }

    def create_predictions_accumulator(self):
        return {
            "Predictions": {var: PredictionsAccumulator() for var in self.predictors.keys()}
        }

    def grad_vector(self):
        device = self.current_device()
        return torch.cat([p.grad.detach().flatten() if p.grad is not None else torch.zeros(p.shape, device=device).flatten() for p in self.parameters()])

    def get_training_classifications(self):
        return [task["name"] for task in self.training_tasks.values() if "classification" in task["type"]]

    def get_training_regressions(self):
        return [task["name"] for task in self.training_tasks.values() if "regression" in task["type"]]

    def get_validation_classifications(self):
        return [task["name"] for task in self.validation_tasks.values() if "classification" in task["type"]]

    def get_validation_regressions(self):
        return [task["name"] for task in self.validation_tasks.values() if "regression" in task["type"]]

    def convert(self, data):
        if not torch.is_tensor(data):
            return torch.tensor(data.astype(np.int16), device=self.current_device())
            # torch does not have uint16 dtype: be aware that indices in range [32768,65535] will be stored as negative numbers
        return data.to(self.current_device())

    def get_batch(self, data, labels, batch_indexes):
        if data.is_sparse:  # sparse tensor does not support array indexing
            mask = (data._indices()[0].view(-1, 1) == torch.tensor(batch_indexes).to(self.current_device()).view(1,-1)).sum(axis=1).bool()
            indices = data._indices()[:, mask]
            index_to_batchindex = {n: i for i, n in enumerate(batch_indexes)}
            indices[0] = torch.tensor([index_to_batchindex[i.item()] for i in indices[0]], device=self.current_device())
            batch = torch.sparse_coo_tensor(indices, data._values()[mask], (len(batch_indexes), *data.shape[1:])).to_dense()
        else:
            batch = data[batch_indexes]
        batch_labels = labels.iloc[batch_indexes].reset_index()
        return batch, batch_labels

    def evaluate(self, data, labels, batch_size):
        data = self.convert(data)
        metrics = Metrics({**self.create_losses_metrics(self.validation_tasks), **self.create_detailed_metrics(self.validation_tasks), **self.create_predictions_accumulator()})
        num_batches = (len(data) + batch_size - 1) // batch_size
        for b in range(num_batches):
            batch, batch_labels = self.get_batch(data, labels, range(b * batch_size, min(len(data), (b + 1) * batch_size)))
            self.validation_step(batch, batch_labels, num_batches, metrics.metrics)
        y_pred_dict = metrics.metrics.pop("Predictions")
        return metrics, y_pred_dict
