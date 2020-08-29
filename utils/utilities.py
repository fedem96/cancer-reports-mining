from datetime import datetime
import json
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import spacy

from utils.constants import *

def _convert_date(date_str, format):
    try:
        date = datetime.strptime(date_str, format)
        return date
    except ValueError:
        return np.nan


def read_csv(dataset_name, encoding="utf-8", types={}, converters={}, dates_format={}, handle_nans={}, decimal='.'):
    start = timer()
    cols = set().union(types, handle_nans, dates_format)
    converters.update({col: (lambda d, f=dates_format[col]: _convert_date(d, f)) for col in dates_format})
    converters.update({col: (lambda d, f=dates_format[col]: _convert_date(d, f)) for col in dates_format})
    end = timer()
    print("1:", (end-start))
    start = end
    df = pd.read_csv(dataset_name, encoding=encoding, usecols=cols, converters=converters, dtype=types, decimal=decimal)
    end = timer()
    print("2:", (end-start))
    start = end
    for col in handle_nans:
        df.loc[df.index[df[col].isnull()], col] = handle_nans[col]

    end = timer()
    print("3:", (end-start))
    start = end
    print("csv read")
    return df


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