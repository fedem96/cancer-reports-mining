from datetime import datetime

import numpy as np
import pandas as pd


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


def bar(values, title=None, xlabel=None, ylabel=None, output_file=None):
    import matplotlib.pyplot as plt
    from collections import Counter

    fig, axes = plt.subplots(1, 1)

    counts = Counter(values)

    xs = counts.keys()
    heights = [counts[x] for x in xs]
    axes.bar(xs, heights)

    if title is not None: plt.title(title)
    if xlabel is not None: plt.xlabel(xlabel)
    if ylabel is not None: plt.ylabel(ylabel)

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.clf()

def plot_bar(df, column, counts_threshold=0, sort_by_index=True, output_file=None):
    plot_bars(df, [column], counts_threshold, sort_by_index, output_file)


def plot_bars(df, columns, counts_threshold=0, sort_by_index=True, output_file=None):
    import matplotlib.pyplot as plt
    import math

    ncol = math.ceil(math.sqrt(len(columns)))
    nrow = math.ceil(len(columns) / ncol)

    fig, axes = plt.subplots(nrow, ncol)

    for col in range(len(columns)):
        r = col // ncol
        c = col % ncol
        counts = df[columns[col]].dropna().astype(str).value_counts()
        if sort_by_index:
            counts = counts.sort_index()

        ax = axes
        if len(columns) > 1 and nrow == 1:
            ax = axes[c]
        if nrow > 1:
            ax = axes[r, c]

        counts[counts > counts_threshold].dropna().plot(kind="bar", ax=ax)
        ax.set_title(columns[col])

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.clf()


def plot_hist(df, column, output_file=None):
    plot_hists(df, [column], output_file)


def plot_hists(df, columns, output_file=None):
    import matplotlib.pyplot as plt
    import math

    ncol = math.ceil(math.sqrt(len(columns)))
    nrow = math.ceil(len(columns) / ncol)

    fig, axes = plt.subplots(nrow, ncol)

    for col in range(len(columns)):
        r = col // ncol
        c = col % ncol
        column = df[columns[col]].dropna()

        ax = axes
        if len(columns) > 1 and nrow == 1:
            ax = axes[c]
        if nrow > 1:
            ax = axes[r, c]

        if len(column) > 0:
            ax.hist(column, bins=range(math.floor(min(column)), math.ceil(max(column))))
            ax.set_title(columns[col])

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
        plt.clf()
