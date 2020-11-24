from datetime import datetime
import json
import hashlib
from inspect import signature
from multiprocessing import Pool
import pickle
import re
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from utils.constants import *
from utils.labels_codec import LabelsCodec


def _convert_date(date_str, format):
    try:
        date = datetime.strptime(date_str, format)
        return date
    except ValueError:
        return np.nan


def read_csv(dataset_name, encoding="utf-8", types={}, converters={}, dates_format={}, handle_nans={}, decimal='.'):
    cols = set().union(types, handle_nans, dates_format)
    converters.update({col: (lambda d, f=dates_format[col]: _convert_date(d, f)) for col in dates_format})
    # converters.update({col: (lambda d, f=dates_format[col]: _convert_date(d, f)) for col in dates_format})
    df = pd.read_csv(dataset_name, encoding=encoding, usecols=cols, converters=converters, dtype=types, decimal=decimal)
    replace_nulls(df, handle_nans)
    print("csv read")
    return df


def replace_nulls(df, replacements_dict):
    for col in replacements_dict:
        df.loc[df.index[df[col].isnull()], col] = replacements_dict[col]

# def merge_and_extract(dataset, columns_names):
#     return dataset[columns_names].values
#
def merge_and_extract(dataset, columns_names):
    #dataset[columns_names] += " "
    merged_column = dataset[columns_names[0]]
    for column in columns_names[1:]:
        merged_column += " " + dataset[column]
    return merged_column.values
    # return dataset[columns_names].apply(lambda x: ' '.join(x), axis=1).values

def train_test_split(df, year_column, test_from_year):
    dfTest = df[df[year_column] >= test_from_year]
    dfTrain = df.drop(df.index[df[year_column] >= test_from_year])
    return dfTrain, dfTest


class Chronometer:
    def __init__(self, identifier):
        self.identifier = identifier

    def __enter__(self):
        self.start = timer()

    def __exit__(self, *args):
        #print("elapsed time ({}): {}".format(self.identifier, (timer() - self.start)))
        pass

def plot_hist_from_columns(df, columns, threshold=0, sort_by_index=False, plot_kind="bar"):
    import matplotlib.pyplot as plt
    import math

    ncol = math.ceil(math.sqrt(len(columns)))
    nrow = math.ceil(len(columns) / ncol)

    fig, axes = plt.subplots(nrow, ncol)

    for col in range(len(columns)):
        column = columns[col]
        r = col // ncol
        c = col % ncol
        counts = df[column].value_counts()
        if sort_by_index:
            counts = counts.sort_index()

        ax = axes
        if len(columns) > 1 and nrow == 1:
            ax = axes[c]
        if nrow > 1:
            ax = axes[r, c]

        counts[counts > threshold].dropna().plot(kind=plot_kind, ax=ax)

    plt.show()


def plot_hists(columns):
    import matplotlib.pyplot as plt
    import math

    ncol = math.ceil(math.sqrt(len(columns)))
    nrow = math.ceil(len(columns) / ncol)

    fig, axes = plt.subplots(nrow, ncol)

    for col in range(len(columns)):
        column = columns[col]
        r = col // ncol
        c = col % ncol

        ax = axes
        if len(columns) > 1 and nrow == 1:
            ax = axes[c]
        if nrow > 1:
            ax = axes[r, c]

        ax.hist(column, bins=range(min(column), max(column)+1))

    plt.show()


def df_to_data(dataframe, pipeline, cols):
    replace_nulls(dataframe, {col: "" for col in cols})
    reports = merge_and_extract(dataframe, cols)

    with Pool(6) as pool:
        data = pool.map(pipeline, reports)  # since pipeline is going to be pickled, I can't use a lambda
    return data


def prepare_dataset(dataset, train_config):
    classifications, regressions = train_config.get("classifications", {}), train_config.get("regressions", {})
    transformations, mappings = train_config.get("transformations", {}), train_config.get("mappings", {})
    for column in classifications:
        if column in transformations:
            for transf in transformations[column]:
                ty = transf["type"]
                if ty == "regex_sub":
                    for s in transf["subs"]:
                        regex = re.compile(s[0], re.I)
                        dataset[column] = dataset[column].apply(lambda v: s[1] if regex.match(str(v)) else v)
                        #dataset.loc[dataset.index[dataset[column].apply(lambda v: None != regex.match(str(v)))], column] = s[1]
                elif ty == "filter":
                    dataset.loc[dataset.index[dataset[column].apply(lambda v: v not in transf["valid_set"])], column] = np.NaN
                else:
                    raise ValueError("invalid transformation '{}' for classification problem".format(ty))
        # dataset[column] = columns_codec[column].encode_batch(dataset[column])


def get_columns_codec(dataset, train_config):
    classifications, mappings = train_config.get("classifications", {}), train_config.get("mappings", {})
    for column in classifications:
        if column not in mappings:
            mappings[column] = sorted(dataset[column].dropna().unique())
    return LabelsCodec().from_mappings(mappings)


def get_encoded_labels(dataset, columns, columns_to_encode, columns_codec):
    labels = dataset[columns].copy()
    for column in columns_to_encode:
        labels[column] = columns_codec[column].encode_batch(labels[column])
    return labels


def caching(function, *args):
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print("creating cache dir")

    def _obj_to_hex(o):
        return "&".join([o.__module__, o.__qualname__, str(signature(o)), str(o.__defaults__), str(o.__kwdefaults__)]) if callable(o) else str(o)

    hex_digest = hashlib.sha256(bytes(str([_obj_to_hex(arg) for arg in args]), "utf-8")).hexdigest()
    file_cache = os.path.join(CACHE_DIR, hex_digest)
    if not os.path.exists(file_cache):
        print("calculating result")
        result = function(*args)
        with open(file_cache, "wb") as file:
            pickle.dump(result, file)
    else:
        print("loading result from cache")
        with open(file_cache, "rb") as file:
            result = pickle.load(file)
    return result
