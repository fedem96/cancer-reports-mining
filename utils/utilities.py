from datetime import datetime
import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd

from utils.constants import *

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
        print("elapsed time ({}): {}".format(self.identifier, (timer() - self.start)))

def plot_hist(df, columns, threshold=0, sort_by_index=False, plot_kind="bar"):
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