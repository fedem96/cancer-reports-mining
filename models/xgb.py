import sys

import numpy as np
from scipy import sparse as sp
from scipy import stats
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit

import xgboost as xgb

from utils.convert import sparse_tensor_to_csr_matrix


class XGBoost:
    def __init__(self, vocab_size, preprocessor, tokenizer, labels_codec, *args, **kwargs):
        self.vocab_size = vocab_size
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.labels_codec = labels_codec
        self.__name__ = "XGBoost"
        self.model = None
        self.model_params = {k:v for k,v in kwargs.items() if k in {"max_depth", "eta", "gamma", "lambda", "alpha", "subsample", "n_estimators", "max_depth", "class_weight", "max_features", "learning_rate"}}
        print("model_params:", self.model_params)
        # self.epochs = kwargs.get("epochs", 10)
        self.model = xgb.XGBClassifier(n_jobs=-1, use_label_encoder=False, **self.model_params)
        self.cls_var = None
        self.directory = kwargs['directory']
        self.randomh = kwargs.get("randomh", False)
        self.num_classes = None

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
        self.num_classes = num_classes
        breakpoint()
        self.model_params["num_class"] = num_classes
        self.model_params["objective"] = "multi:softmax"

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

    def convert(self, data, labels):
        labels = labels[self.cls_var].values
        data = sparse_tensor_to_csr_matrix(data)[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)].to_numpy().astype(int)
        return data, labels

    def fit(self, train_data, train_labels, val_data, val_labels, info, callbacks, **hyperparameters):
        print("starting training of random forest")
        if self.cls_var is None:
            raise Exception("var not set")
        if train_data.shape[1] != 1 or val_data.shape[1] != 1:
            raise ValueError("this model does not support multi-instance: you have to concatenate the reports")

        train_data, train_labels = self.convert(train_data, train_labels)
        val_data, val_labels = self.convert(val_data, val_labels)

        if self.randomh:
            X = sp.vstack((train_data, val_data), format='csr')
            y = np.concatenate([train_labels, val_labels])

            clf_xgb = xgb.XGBClassifier(objective='multi:softmax', reg_lambda=0, use_label_encoder=False, num_class=self.num_classes)
            param_dist = {
                'n_estimators': stats.randint(20, 100),
                'learning_rate': stats.uniform(0.01, 0.4),
                # 'subsample': stats.uniform(0.3, 0.7),
                'max_depth': [4, 5, 6, 7, 8],
                "alpha": stats.uniform(0.5, 3),
                # "gamma": stats.uniform(0, 2),
                # 'colsample_bytree': stats.uniform(0.5, 0.9),
                # 'min_child_weight': [1, 2, 3, 4]
            }

            clf = RandomizedSearchCV(clf_xgb,
                                     param_distributions=param_dist,
                                     cv=PredefinedSplit(test_fold=np.concatenate([-1*np.ones(len(train_labels)), np.zeros(len(val_labels))])),
                                     n_iter=10,
                                     scoring='f1_macro' if self.num_classes != 2 else 'f1',
                                     error_score=0,
                                     verbose=3,
                                     n_jobs=-1,
                                     random_state=0,
                                     refit=False)

            clf.fit(X, y)
            print("best params")
            print(clf.best_params_)
            self.model = xgb.XGBClassifier(n_jobs=-1, use_label_encoder=False, **clf.best_params_)

        # train_data = xgb.DMatrix(train_data, label=train_labels)
        # self.model = xgb.train(self.model_params, train_data, self.epochs)
        self.model.fit(train_data, train_labels)

        print(self.model)

        # breakpoint()

    def evaluate(self, data, labels, batch_size=None, convert=True):
        if convert:
            data, labels = self.convert(data, labels)
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
        return metrics, {self.cls_var: lambda: predictions}
