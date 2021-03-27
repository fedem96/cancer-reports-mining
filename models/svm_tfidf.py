import sys

import numpy as np
from scipy import sparse
from sklearn.svm import SVC


class SVM:
    def __init__(self, vocab_size, preprocessor, tokenizer, token_codec, labels_codec, idf, *args, **kwargs):
        self.vocab_size = vocab_size
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.token_codec = token_codec
        self.labels_codec = labels_codec
        self.idf = idf
        self.__name__ = "SVM"
        self.model = SVC(class_weight="balanced", verbose=True)
        self.cls_var = None

    def set_reduce_method(self, reduce_type, reduce_mode):
        print("set_reduce_method() not supported: skipping", file=sys.stderr)
        pass

    def add_classification(self, cls_var, num_classes):
        if self.cls_var is not None:
            raise Exception("var already set")
        self.cls_var = cls_var

    def add_regression(self, reg_var):
        print("add_regression() not supported: skipping", file=sys.stderr)
        pass

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

    def to_tfidf_matrix(self, data):

        rs = []
        cs = []
        ds = []
        for i, encoded_report in enumerate(data):
            report = self.token_codec.decode(encoded_report)
            rs.append(i * np.ones(len(report)))
            cs.append(encoded_report)
            ds.append(np.array([self.idf.get_idf(token) for token in report]))

        r = np.concatenate(rs)
        c = np.concatenate(cs)
        d = np.concatenate(ds)
        tfidf_matrix = sparse.coo_matrix((d, (r, c)), shape=(len(data), self.token_codec.num_tokens()))
        return tfidf_matrix.tocsr()



    def fit(self, train_data, train_labels, val_data, val_labels, info, callbacks, **hyperparameters):
        if self.cls_var is None:
            raise Exception("var not set")
        train_data = self.to_tfidf_matrix(train_data)
        train_labels = train_labels[self.cls_var].values
        train_data = train_data[~np.isnan(train_labels)]
        train_labels = train_labels[~np.isnan(train_labels)].to_numpy().astype(int)
        val_data = self.to_tfidf_matrix(val_data)
        val_labels = val_labels[self.cls_var].values
        val_data = val_data[~np.isnan(val_labels)]
        val_labels = val_labels[~np.isnan(val_labels)].to_numpy().astype(int)
        self.model.fit(train_data, train_labels)
        print("evaluating train")
        print(self.evaluate(train_data, train_labels))
        print("evaluating val")
        print(self.evaluate(val_data, val_labels))
        
    def evaluate(self, data, labels):
        predictions = self.model.predict(data)
        TN = np.logical_and(predictions == labels, predictions == np.zeros_like(predictions)).sum()
        TP = np.logical_and(predictions == labels, predictions == np.ones_like(predictions)).sum()
        FN = np.logical_and(predictions != labels, predictions == np.zeros_like(predictions)).sum()
        FP = np.logical_and(predictions != labels, predictions == np.ones_like(predictions)).sum()
        return {
            "Accuracy":     (TP + TN) / (TP + TN + FP + FN),
            "0-precision":  TN / (TN + FN),
            "0-recall":     TN / (TN + FP),
            "1-precision":  TP / (TP + FP),
            "1-recall":     TP / (TP + FN)
        }