import json

import pandas as pd


class LabelCodec:
    def __init__(self, coder={}, decoder={}):
        self.coder = coder
        self.decoder = decoder
        assert len(self.coder) == len(self.decoder)

    def from_series(self, column_series):
        self.coder, self.decoder = {}, {}
        for k in range(len(column_series.unique())):
            v = column_series.unique()[k]
            self.coder[v] = k
            self.decoder[k] = v
        return self

    def from_list(self, class_names):
        self.coder, self.decoder = {}, {}
        for k, cls_name in enumerate(class_names):
            self.coder[cls_name] = k
            self.decoder[k] = cls_name
        return self

    def load(self, filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        self.coder = js["coder"]
        self.decoder = js["decoder"]
        assert len(self.coder) == len(self.decoder)
        return self

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump(self.to_json(), file)
        return self

    def to_json(self):
        return {"coder": self.coder, "decoder": self.decoder}

    def encode(self, label):
        return self.coder.get(label, 0)

    def encode_batch(self, labels):
        return labels.apply(lambda val: self.coder[val] if not pd.isnull(val) else val).astype("Int64")

    def decode(self, label):
        return self.decoder[label]

    def decode_batch(self, labels):
        return labels.apply(lambda val: self.decoder[val] if not pd.isnull(val) else val)


class LabelsCodec:
    def __init__(self, codecs={}):
        self.codecs = codecs

    def from_dataframe(self, dataset, columns):
        self.codecs = {}
        for column in columns:
            self.codecs[column] = LabelCodec().from_series(dataset[column])
        return self

    def from_mappings(self, mappings):
        self.codecs = {}
        for column in mappings:
            self.codecs[column] = LabelCodec().from_list(mappings[column])
        return self

    def load(self, filename):
        with open(filename, "rt") as file:
            j = json.load(file)
        self.codecs = {}
        for column in j:
            self.codecs[column] = LabelCodec(j[column]["coder"], j[column]["decoder"])
        return self

    def save(self, filename):
        j = {}
        for column in self.codecs:
            j[column] = self.codecs[column].to_json()
        with open(filename, "wt") as file:
            json.dump(j, file)
        return self

    def __getitem__(self, item):
        return self.codecs[item]
