import json
import re
from typing import List, Dict

import numpy as np
import pandas as pd


class LabelCodec:
    def __init__(self, encoder={}, decoder={}):
        self.encoder = encoder
        self.decoder = decoder

    def load(self, filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        self.encoder = js["encoder"]
        self.decoder = js["decoder"]
        assert len(self.encoder) == len(self.decoder)
        return self

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump(self.to_json(), file)
        return self

    def to_json(self):
        return {"encoder": self.encoder, "decoder": self.decoder}

    def encode(self, label):
        return self.encoder.get(label, 0)

    def encode_batch(self, labels):
        return labels.apply(lambda val: self.encoder[val] if not pd.isnull(val) else val)

    def decode(self, label):
        return self.decoder[label]

    def decode_batch(self, labels):
        return labels.apply(lambda val: self.decoder[val] if not pd.isnull(val) else val)


class AutoClassificationLabelCodec(LabelCodec):

    def __init__(self):
        super().__init__()
        self.codec_created = False

    def create_codec(self, values):
        encoder, decoder = {}, {}
        for k, cls_name in enumerate(sorted(values)):
            encoder[cls_name] = k
            decoder[k] = cls_name
        self.encoder = encoder
        self.decoder = decoder
        self.codec_created = True

    def encode(self, label):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'create_codec' first")
        return super().encode(label)

    def encode_batch(self, labels):
        if not self.codec_created:
            self.create_codec(labels.dropna().unique())
        return super().encode_batch(labels)

    def decode(self, label):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'update_codec' first")
        return super().decode(label)

    def decode_batch(self, labels):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'update_codec' first")
        return super().decode_batch(labels)


class AffineLabelCodec(LabelCodec):
    def __init__(self, a, b):
        super().__init__()
        if a == 0:
            raise ValueError("a can't be 0")
        self.a = a
        self.b = b

    def encode(self, label):
        return self.a * label + self.b

    def encode_batch(self, labels):
        return self.a * labels + self.b

    def decode(self, label):
        return (label - self.b) / self.a

    def decode_batch(self, labels):
        return (labels - self.b) / self.a


class AutoRegressionLabelCodec(AffineLabelCodec):

    def __init__(self):
        super().__init__(1, 0)
        self.codec_created = False

    def create_codec(self, values):
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            raise ValueError("min and max are equal: can't autodetect affine transform to normalize data")
        self.a = 1 / (max_val - min_val)
        self.b = -min_val * self.a
        # a,b s.t.: a*min_val+b==0 and a*max_val+b==1
        self.codec_created = True

    def encode(self, label):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'create_codec' first")
        return super().encode(label)

    def encode_batch(self, labels):
        if not self.codec_created:
            self.create_codec(labels.dropna())
        return super().encode_batch(labels)

    def decode(self, label):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'update_codec' first")
        return super().decode(label)

    def decode_batch(self, labels):
        if not self.codec_created:
            raise Exception("encoder and decoder not created yet: call 'encode_batch' or 'update_codec' first")
        return super().decode_batch(labels)


class LabelCodecsSequence(LabelCodec):
    def __init__(self, codecs: List[LabelCodec]):
        self.codecs = codecs

    def encode(self, label):
        for codec in self.codecs:
            label = codec.encode(label)
        return label

    def encode_batch(self, labels):
        for codec in self.codecs:
            labels = codec.encode_batch(labels)
        return labels

    def decode(self, label):
        for codec in reversed(self.codecs):
            label = codec.decode(label)
        return label

    def decode_batch(self, labels):
        for codec in reversed(self.codecs):
            labels = codec.decode_batch(labels)
        return labels


class LabelCodecFactory:

    @staticmethod
    def from_series(column_series) -> LabelCodec:
        encoder, decoder = {}, {}
        for k in range(len(column_series.unique())):
            v = column_series.unique()[k]
            encoder[v] = k
            decoder[k] = v
        return LabelCodec(encoder, decoder)

    @staticmethod
    def from_list(class_names) -> LabelCodec:
        encoder, decoder = {}, {}
        for k, cls_name in enumerate(class_names):
            encoder[cls_name] = k
            decoder[k] = cls_name
        return LabelCodec(encoder, decoder)

    @staticmethod
    def from_pairs(class_names, encoded_values) -> LabelCodec:
        encoder, decoder = {}, {}
        for cls_name, v in zip(class_names, encoded_values):
            encoder[cls_name] = v
            decoder[v] = cls_name
        return LabelCodec(encoder, decoder)

    @staticmethod
    def from_transformation(transformation) -> LabelCodec:
        ty = transformation["type"]
        if ty == "regex_sub":
            return LabelCoder(RegexLookup(transformation["subs"]))
        elif ty == "filter":
            return LabelCoder(Filter(transformation["valid_set"]))
        else:
            raise ValueError("invalid transformation '{}'".format(ty))

    @staticmethod
    def from_classification_transformations(transformations) -> LabelCodecsSequence:
        codecs = []
        has_mapping = False
        for transformation in transformations:
            codecs.append(LabelCodecFactory.from_transformation(transformation))
            has_mapping = has_mapping or transformation['type'] == 'mapping'
        if not has_mapping:
            codecs.append(LabelCodecFactory.auto_classification_codec())
        codecs.append(LabelCodecFactory.cast("Int64"))
        return LabelCodecsSequence(codecs)

    @staticmethod
    def from_regression_transformations(transformations) -> LabelCodecsSequence:
        codecs = []
        for transformation in transformations:
            print("regressions transformations not implemented yet")
        codecs.append(LabelCodecFactory.auto_regression_codec())
        codecs.append(LabelCodecFactory.cast("Float64"))
        return LabelCodecsSequence(codecs)

    @staticmethod
    def cast(type_str: str):
        return LabelCaster(type_str)

    @staticmethod
    def auto_classification_codec():
        return AutoClassificationLabelCodec()

    @staticmethod
    def auto_regression_codec():
        return AutoRegressionLabelCodec()


class LabelsCodec:
    def __init__(self, codecs: Dict[str, LabelCodec]={}):
        self.codecs = codecs

    def from_dataframe(self, dataframe, columns):
        self.codecs = {}
        for column in columns:
            self.codecs[column] = LabelCodecFactory.from_series(dataframe[column])
        return self

    def load(self, filename):
        with open(filename, "rt") as file:
            j = json.load(file)
        self.codecs = {}
        for column in j:
            self.codecs[column] = LabelCodec(j[column]["encoder"], j[column]["decoder"])
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


class LabelsCodecFactory:

    @staticmethod
    def from_transformations(classifications: List[str], regressions: List[str], transformations: Dict[str, List]):
        codecs = {}
        for column in classifications:
            col_transformations = transformations[column] if column in transformations else []
            codecs[column] = LabelCodecFactory.from_classification_transformations(col_transformations)
        for column in regressions:
            col_transformations = transformations[column] if column in transformations else []
            codecs[column] = LabelCodecFactory.from_regression_transformations(col_transformations)
        return LabelsCodec(codecs)


class RegexLookup:
    def __init__(self, lookup_pairs=[]):
        self.lookup_pairs = [(re.compile(p, re.I), v) for p, v in lookup_pairs]

    def __getitem__(self, key):
        for pattern, value in self.lookup_pairs:
            if re.search(pattern, key):
                return value
        raise KeyError("key '{}' does not match any regex".format(key))


class Filter:
    def __init__(self, valid_values=set()):
        self.valid_values = set(valid_values)

    def __getitem__(self, key):
        if key in self.valid_values:
            return key
        return np.nan


class LabelCoder(LabelCodec):
    def __init__(self, encoder):
        super().__init__(encoder, {})

    def decode(self, label):
        return label

    def decode_batch(self, labels):
        return labels


class LabelCaster(LabelCoder):
    def __init__(self, type_str):
        self.type_str = type_str

    def encode_batch(self, labels):
        return labels.astype(self.type_str)
