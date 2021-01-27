import re
from multiprocessing import Pool

import numpy as np
import pandas as pd

from utils.cache import caching
from utils.labels_codec import LabelsCodec
from utils.utilities import replace_nulls, merge_and_extract


class Dataset:

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.dataframe = pd.read_csv(csv_file)
        self.dataframe.columns = self.dataframe.columns.str.replace('%', '%25')  # TODO: change
        self.input_cols = None
        self.full_pipe = None
        self.data = None
        self.classifications = None
        self.regressions = None
        self.transformations = None
        self.mappings = None
        self.columns_codec = None
        self.labels = None
        self.index_to_key = None
        self.key_to_index = None
        self._nunique = None
        self._update_nunique()

    def set_input_cols(self, input_cols):
        self.input_cols = input_cols

    def process_records(self, full_pipe):
        self.full_pipe = full_pipe

        def _df_to_data(dataframe, pipeline, cols):
            replace_nulls(dataframe, {col: "" for col in cols})
            reports = merge_and_extract(dataframe, cols)

            with Pool(6) as pool:
                data = pool.map(pipeline, reports)  # since pipeline is going to be pickled, I can't use a lambda
            return data

        self.data = caching(_df_to_data)(self.dataframe, full_pipe, self.input_cols)

    def prepare_for_training(self, classifications=[], regressions=[], transformations={}, mappings={}):
        self.classifications, self.regressions = classifications, regressions
        self.transformations, self.mappings = transformations, mappings
        for column in classifications:
            if column in transformations:
                for transf in transformations[column]:
                    ty = transf["type"]
                    if ty == "regex_sub":
                        for s in transf["subs"]:
                            regex = re.compile(s[0], re.I)
                            self.dataframe[column] = self.dataframe[column].apply(lambda v: s[1] if regex.match(str(v)) else v)
                            # self.dataframe.loc[self.dataframe.index[self.dataframe[column].apply(lambda v: None != regex.match(str(v)))], column] = s[1]
                    elif ty == "filter":
                        self.dataframe.loc[self.dataframe.index[
                                        self.dataframe[column].apply(lambda v: v not in transf["valid_set"])], column] = np.NaN
                    else:
                        raise ValueError("invalid transformation '{}' for classification problem".format(ty))
            # dataset[column] = columns_codec[column].encode_batch(dataset[column])

        self._update_nunique()

    def get_columns_codec(self):
        if self.columns_codec is None:
            for column in self.classifications:
                if column not in self.mappings:
                    self.mappings[column] = sorted(self.dataframe[column].dropna().unique())

            self.columns_codec = LabelsCodec().from_mappings(self.mappings)
        return self.columns_codec

    def set_columns_codec(self, columns_codec):
        self.columns_codec = columns_codec

    def encode_labels(self):
        labels = self.dataframe[self.classifications + self.regressions].copy()
        for column in self.classifications:
            labels[column] = self.columns_codec[column].encode_batch(labels[column])
        self.labels = labels

    def cut_sequences(self, max_length):
        self.data = [sequence[:max_length] for sequence in self.data]

    def group_by(self, attributes):

        def _group_by(self_dataframe, self_data, self_labels, attrs):
            index_to_key = {}
            key_to_index = {}
            grouped = self_dataframe.groupby(attrs)
            data = [[] for i in range(grouped.ngroups)]
            labels = pd.DataFrame(index=range(grouped.ngroups), columns=self_labels.columns)
            for index, (key, group) in enumerate(grouped):
                index_to_key[index] = key
                key_to_index[key] = index
                group_index = group.index

                for col in self_labels.columns:
                    labels.loc[index, col] = self_labels.at[group_index[0], col]

                for i in group_index:
                    data[index].append(self_data[i])
            return index_to_key, key_to_index, grouped, data, labels

        self.index_to_key, self.key_to_index, self.dataframe, self.data, self.labels = caching(_group_by)(self.dataframe, self.data, self.labels, attributes)

    def assert_disjuncted(self, other_dataset):
        assert len(set(self.key_to_index.keys()).intersection(other_dataset.key_to_index.keys())) == 0
        # number of patients in both sets: len( set(list(zip(*list(self.key_to_index.keys())))[0]).intersection(set(list(zip(*list(other_dataset.key_to_index.keys())))[0])) )

    def filter(self, filter_strategy):
        def _filter(self_dataframe, self_data, filter_strat):
            if callable(filter_strat):
                filter_fn = filter_strat
            elif type(filter_strat) == str:
                def _same_year(record, group_df):
                    mask = group_df['anno_referto'].values == group_df['anno_diagnosi'].values
                    if not any(mask):
                        mask = ~mask
                    record = [r for (r, b) in zip(record, mask) if b]
                    group_df = group_df[mask] # TODO: this line is slow
                    return record, group_df
                filter_dict = {"same_year": _same_year}
                filter_fn = filter_dict[filter_strat]
            else:
                raise ValueError("Invalid filter method")
            data = []
            dataframe = []
            for (key, group), record in zip(self_dataframe, self_data):
                r, g = filter_fn(record, group)
                dataframe.append((key, g))
                data.append(record)
            assert len(self_data) == len(data)
            assert len(dataframe) == len(data)
            return dataframe, data
        self.dataframe, self.data = _filter(self.dataframe, self.data, filter_strategy) # TODO: cache

    def reduce(self, reduce_strategy):
        def _reduce(self_dataframe, self_data, reduce_strat):
            if callable(reduce_strat):
                reduce_fn = reduce_strat
            elif type(reduce_strat) == str:
                def _most_recent(record, group_df):
                    mr_year = max(group_df['anno_referto'].values)
                    mask = group_df['anno_referto'].values == mr_year
                    if not any(mask):
                        mask = ~mask
                    group_df = group_df[mask] # TODO: this line is slow
                    record = [r for (r, b) in zip(record, mask) if b]
                    if len(record) > 1:
                        mr_isto = max(group_df['id_isto'].values)
                        mask = group_df['id_isto'].values == mr_isto
                        record = [r for (r, b) in zip(record, mask) if b]
                        # group_df = group_df[mask]
                    assert len(record) == 1
                    return record[0]
                reduce_dict = {"most_recent": _most_recent}
                reduce_fn = reduce_dict[reduce_strat]
            else:
                raise ValueError("Invalid reduce method")
            data = []
            for (key, group), record in zip(self_dataframe, self_data):
                data.append(reduce_fn(record, group))
            assert len(self_data) == len(data)
            return data
        self.data = caching(_reduce)(self.dataframe, self.data, reduce_strategy)

    def nunique(self, var):
        return self._nunique[var]

    def _update_nunique(self):
        self._nunique = {}
        for col in self.dataframe.columns:
            self._nunique[col] = self.dataframe[col].nunique()
