import json
import os
import sys

import numpy as np
from sklearn.svm import SVC

from utils.convert import sparse_tensor_to_csr_matrix


class SVM:
    def __init__(self, vocab_size, preprocessor, tokenizer, labels_codec, *args, **kwargs):
        self.vocab_size = vocab_size
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.labels_codec = labels_codec
        self.__name__ = "SVM"
        self.model = SVC(verbose=True, **{k:v for k,v in kwargs.items() if k in {"C", "class_weight"}})
        self.cls_var = None
        self.directory = kwargs['directory']

    def encode_report(self, report):
        return self.tokenizer.tokenize(self.preprocessor.preprocess(report), encode=True)

    def set_reports_pooling_method(self, *args, **kwargs):
        print("set_reports_pooling_method() not supported: skipping", file=sys.stderr)

    def set_tokens_pooling_method(self, *args, **kwargs):
        print("set_tokens_pooling_method() not supported: skipping", file=sys.stderr)

    def set_reports_transformation_method(self, *args, **kwargs):
        print("set_reports_transformation_method() not supported: skipping", file=sys.stderr)

    def add_classification(self, cls_var, num_classes, *args):
        if self.cls_var is not None:
            raise Exception("var already set")
        self.cls_var = cls_var

    def add_regression(self, reg_var, *args):
        print("add_regression() not yet supported: skipping", file=sys.stderr)

    def current_device(self):
        return "cpu"

    def parameters(self):
        print("parameters() not supported: skipping", file=sys.stderr)
        return []

    def named_parameters(self):
        print("named_parameters() not supported: skipping", file=sys.stderr)
        return []

    def to(self, device):
        print("to(device) not supported: skipping", file=sys.stderr)
        return self

    def fit(self, train_data, train_labels, val_data, val_labels, info, callbacks, **hyperparameters):
        print("starting training of SVM")
        if self.cls_var is None:
            raise Exception("var not set")

        train_labels = train_labels[self.cls_var].values
        train_data = sparse_tensor_to_csr_matrix(train_data)[~np.isnan(train_labels)]
        train_labels = train_labels[~np.isnan(train_labels)].to_numpy().astype(int)

        val_labels = val_labels[self.cls_var].values
        val_data = sparse_tensor_to_csr_matrix(val_data)[~np.isnan(val_labels)]
        val_labels = val_labels[~np.isnan(val_labels)].to_numpy().astype(int)

        self.model.fit(train_data, train_labels)

        print("evaluating train")
        train_metrics = self.evaluate(train_data, train_labels)
        print(train_metrics)
        with open(os.path.join(self.directory, "train_metrics.json"), "wt") as file:
            json.dump(train_metrics, file)

        print("evaluating val")
        val_metrics = self.evaluate(val_data, val_labels)
        print(val_metrics)
        with open(os.path.join(self.directory, "val_metrics.json"), "wt") as file:
            json.dump(val_metrics, file)

    def evaluate(self, data, labels):
        predictions = self.model.predict(data)
        correct = predictions == labels
        wrong = ~correct
        classes = sorted(set(predictions).union(set(labels)))
        metrics = {
            "Accuracy": (predictions == labels).sum() / len(labels),
        }
        for c in classes:
            # TN = np.logical_and(correct, predictions != c * np.ones_like(predictions)).sum()
            TP = np.logical_and(correct, predictions == c * np.ones_like(predictions)).sum()
            FN = np.logical_and(wrong,   labels      == c * np.ones_like(predictions)).sum()
            FP = np.logical_and(wrong,   predictions == c * np.ones_like(predictions)).sum()
            p = TP / (TP + FP) if TP + FP > 0 else 0
            r = TP / (TP + FN) if TP + FN > 0 else 0
            metrics.update({
                f"{c}-precision": p,
                f"{c}-recall": r,
                f"{c}-F1": 2 * p * r / (p + r) if p + r > 0 else 0
            })
        metrics.update({
            "M-precision": sum([v for k,v in metrics.items() if "precision" in k]) / len(classes),
            "M-recall":    sum([v for k,v in metrics.items() if "recall"    in k]) / len(classes),
            "M-F1":        sum([v for k,v in metrics.items() if "F1"        in k]) / len(classes)
        })
        return metrics
