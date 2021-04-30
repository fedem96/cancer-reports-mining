import json
import os
import re
import sys

from graphviz import Source
import numpy as np
from sklearn import tree
from sklearn.tree._tree import TREE_LEAF

from utils.convert import sparse_tensor_to_csr_matrix


class DecisionTree:
    def __init__(self, vocab_size, preprocessor, tokenizer, labels_codec, *args, **kwargs):
        self.vocab_size = vocab_size
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.labels_codec = labels_codec
        self.__name__ = "DecisionTree"
        self.model = tree.DecisionTreeClassifier(**{k:v for k,v in kwargs.items() if k in {"max_depth", "class_weight", "min_samples_split", "min_samples_leaf"}})
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
        print("starting training of decision tree")
        if self.cls_var is None:
            raise Exception("var not set")
        if train_data.shape[1] != 1 or val_data.shape[1] != 1:
            raise ValueError("this model does not support multi-instance: you have to concatenate the reports")

        train_labels = train_labels[self.cls_var].values
        train_data = sparse_tensor_to_csr_matrix(train_data)[~np.isnan(train_labels)]
        train_labels = train_labels[~np.isnan(train_labels)].to_numpy().astype(int)

        val_labels = val_labels[self.cls_var].values
        val_data = sparse_tensor_to_csr_matrix(val_data)[~np.isnan(val_labels)]
        val_labels = val_labels[~np.isnan(val_labels)].to_numpy().astype(int)

        self.model.fit(train_data, train_labels)

        # show learned tree
        print(re.sub(r"feature_(\d*)", lambda s: self.tokenizer.decode_token(int(s.group(1))), tree.export_text(self.model)))
        g = Source(tree.export_graphviz(self.model, out_file=None, feature_names=self.tokenizer.decode(range(0, self.tokenizer.num_tokens() + 1))))
        g.format = 'pdf'
        g.render(os.path.join(self.directory, "tree"), view=True)

        prune_duplicate_leaves(self.model)

        print(re.sub(r"feature_(\d*)", lambda s: self.tokenizer.decode_token(int(s.group(1))), tree.export_text(self.model)))
        g = Source(tree.export_graphviz(self.model, out_file=None, feature_names=self.tokenizer.decode(range(0, self.tokenizer.num_tokens() + 1))))
        g.format = 'pdf'
        g.render(os.path.join(self.directory, "simplified_tree"), view=True)

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


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and inner_tree.children_right[index] == TREE_LEAF)


def prune_duplicate_leaves(model):
    # Remove leaves if both
    decisions = model.tree_.value.argmax(axis=2).flatten().tolist() # Decision for each node
    _prune_index(model.tree_, decisions)


def _prune_index(tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(tree, tree.children_left[index]):
        _prune_index(tree, decisions, tree.children_left[index])
    if not is_leaf(tree, tree.children_right[index]):
        _prune_index(tree, decisions, tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(tree, tree.children_left[index]) and
        is_leaf(tree, tree.children_right[index]) and
        (decisions[index] == decisions[tree.children_left[index]]) and
        (decisions[index] == decisions[tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        tree.children_left[index] = TREE_LEAF
        tree.children_right[index] = TREE_LEAF
        ##print("Pruned {}".format(index))