import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from utils.cache import caching
from utils.labels_codec import LabelsCodec
from utils.serialization import load
from utils.utilities import replace_nulls, merge_and_extract


class Dataset:

    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.dataframe = pd.read_csv(csv_file)
        self.dataframe.columns = self.dataframe.columns.str.replace('%', '%25')  # TODO: change
        self.input_cols = []
        self.encoded_input_cols = []
        self.full_pipe = None
        self.classifications = []
        self.regressions = []
        self.transformations = None
        self.mappings = None
        self.columns_codec = None
        self.labels = None
        self.index_to_key = None
        self.key_to_index = None
        self._nunique = None
        self._update_nunique()
        self.grouping_attributes = None
        self.filtering_columns = []
        self.reducing_strategy = None
        self.reducing_args = None

    def set_input_cols(self, input_cols):
        self.input_cols = input_cols

    def add_encoded_column(self, full_pipe, new_column_name, max_length=None):
        self.full_pipe = full_pipe

        def _df_to_data(dataframe, pipeline, cols):
            if type(dataframe) == list:
                return [_df_to_data(d, pipeline, cols) for d in dataframe]

            replace_nulls(dataframe, {col: "" for col in cols})
            reports = merge_and_extract(dataframe, cols)

            try:
                with Pool(6) as pool:
                    data = pool.map(pipeline, reports)  # since pipeline is going to be pickled, I can't use a lambda
            except pickle.PicklingError:
                print("Cannot pickle pipeline, using sequential version", file=sys.stderr)
                data = [pipeline(r) for r in reports]
            return data

        new_column = caching(_df_to_data)(self.dataframe, full_pipe, self.input_cols)
        if type(self.dataframe) == list:
            for df, new_col in zip(self.dataframe, new_column):
                df[new_column_name] = new_col
        else:
            self.dataframe[new_column_name] = new_column
        if max_length is not None:
            self.dataframe[new_column_name] = [sequence[:max_length] for sequence in self.dataframe[new_column_name]]
        self.encoded_input_cols.append(new_column_name)

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
        for column in self.classifications:
            self.dataframe[column] = self.columns_codec[column].encode_batch(self.dataframe[column])
        for column in self.regressions:
            self.dataframe[column] = self.columns_codec[column].encode_batch(self.dataframe[column])
        self._update_nunique()

    def get_labels_columns(self):
        return self.classifications + self.regressions

    def get_labels(self):
        if type(self.dataframe) == list:
            return pd.concat([df[self.get_labels_columns()].head(1) for df in self.dataframe])
        return self.dataframe[self.get_labels_columns()]

    def get_data(self, column_name):
        if type(self.dataframe) == list:
            return [df[column_name].values[0] for df in self.dataframe]
        return self.dataframe[column_name].values[0]

    def group_by(self, attributes):

        def _group_by(self_dataframe, attrs):
            index_to_key = {}
            key_to_index = {}
            grouped = self_dataframe.groupby(attrs)
            for index, (key, group) in enumerate(grouped):
                index_to_key[index] = key
                key_to_index[key] = index

            dataframe = []
            for key, group in grouped:
                dataframe.append(group)
            return index_to_key, key_to_index, dataframe

        self.index_to_key, self.key_to_index, self.dataframe = caching(_group_by)(self.dataframe, attributes)

    def lazy_group_by(self, attributes):
        if self.grouping_attributes is not None:
            raise Exception("Already grouped: cannot group twice")
        self.grouping_attributes = attributes


    def assert_disjuncted(self, other_dataset):
        assert len(set(self.key_to_index.keys()).intersection(other_dataset.key_to_index.keys())) == 0
        # number of patients in both sets: len( set(list(zip(*list(self.key_to_index.keys())))[0]).intersection(set(list(zip(*list(other_dataset.key_to_index.keys())))[0])) )

    def filter(self, filter_strategy, filter_args=None):
        def _filter(self_dataframe, filter_strat, f_args):
            if callable(filter_strat):
                filter_fn = filter_strat
            elif type(filter_strat) == str:
                if filter_strat == "classifier":
                    model = load(filter_args["path"])
                    model.eval()
                    self.add_encoded_column(model.encode_report, filter_args["encoded_data_column"], filter_args["max_length"])
                def _same_year(group_df, *args):
                    mask = group_df['anno_referto'].values == group_df['anno_diagnosi'].values
                    if not any(mask):
                        mask = ~mask
                    group_df = group_df[mask] # TODO: this line is slow
                    return group_df
                def _with_classifier(group_df, fa):
                    encoded_data_column = fa["encoded_data_column"]
                    torch_record = [torch.tensor(report) for report in group_df[encoded_data_column]]
                    mask = model(torch_record)['sede_icdo3'].argmax(dim=1).cpu().numpy().astype(bool)
                    if any(mask):
                        group_df = group_df[mask]
                    return group_df
                filter_dict = {"same_year": _same_year, "classifier": _with_classifier}
                filter_fn = filter_dict[filter_strat]
            else:
                raise ValueError("Invalid filter method")
            dataframe = []
            for key, group in self_dataframe:
                r, g = filter_fn(group, f_args)
                dataframe.append((key, g))
            return dataframe
        self.dataframe = _filter(self.dataframe, filter_strategy, filter_args)

    def lazy_filter(self, filter_strategy, filter_args=None):
        def _filter(self_dataframe, filter_strat, f_args):
            if callable(filter_strat):
                filter_fn = filter_strat
            elif type(filter_strat) == str:
                if filter_strat == "classifier":
                    model = load(filter_args["path"])
                    model.eval()
                    self.add_encoded_column(model.encode_report, filter_args["encoded_data_column"], filter_args["max_length"])
                def _same_year(df, *args):
                    return df['anno_referto'].values == df['anno_diagnosi'].values
                def _with_classifier(df, fa):
                    encoded_data_column = fa["encoded_data_column"]
                    torch_reports = [torch.tensor(report, device=model.current_device()) for report in df[encoded_data_column]]
                    batch_size = 64
                    filter_result = []
                    for i in range(0, len(torch_reports), batch_size):
                        batch = torch_reports[i: min(i+batch_size, len(torch_reports))]
                        filter_result.append( model(batch)['sede_icdo3'].argmax(dim=1).cpu().numpy().astype(bool) )
                    return np.concatenate(filter_result)
                filter_dict = {"same_year": _same_year, "classifier": _with_classifier}
                filter_fn = filter_dict[filter_strat]
            else:
                raise ValueError("Invalid filter method")
            return filter_fn(self_dataframe, f_args)

        filter_result = _filter(self.dataframe, filter_strategy, filter_args)
        filtering_column_name = "_filter_" + str(len(self.filtering_columns))
        self.filtering_columns.append(filtering_column_name)
        self.dataframe[filtering_column_name] = filter_result

    def filter_now(self):
        def _filter(self_dataframe, filtering_columns):
            dataframe = []
            for group in self_dataframe:
                mask = group[filtering_columns].prod(axis=1).astype(bool)
                if any(mask):
                    dataframe.append(group[mask])
                else:
                    dataframe.append(group)
            return dataframe

        self.dataframe = _filter(self.dataframe, self.filtering_columns)

    def reduce(self, reduce_strategy, reduce_args=None):
        def _reduce(self_dataframe, reduce_strat, r_args):
            if callable(reduce_strat):
                reduce_fn = reduce_strat
            elif type(reduce_strat) == str:
                def _most_recent(group_df, ra):
                    mr_year = max(group_df['anno_referto'].values)
                    mask = group_df['anno_referto'].values == mr_year
                    if not any(mask):
                        mask = ~mask
                    group_df = group_df[mask] # TODO: this line is slow
                    if len(group_df) > 1:
                        mr_isto = max(group_df['id_isto'].values)
                        mask = group_df['id_isto'].values == mr_isto
                        group_df = group_df[mask]
                    assert len(group_df) == 1
                    return group_df
                reduce_dict = {"most_recent": _most_recent}
                reduce_fn = reduce_dict[reduce_strat]
            else:
                raise ValueError("Invalid reduce method")
            dataframe = []
            for group in self_dataframe:
                dataframe.append(reduce_fn(group, r_args))
            return dataframe #TODO: finish and check
        self.dataframe = caching(_reduce)(self.dataframe, reduce_strategy, reduce_args)

    def lazy_reduce(self, reduce_strategy, reduce_args=None):
        if self.reducing_strategy is not None:
            raise Exception("Already reduced: cannot reduce twice")
        self.reducing_strategy = reduce_strategy
        self.reducing_args = reduce_args

    def reduce_now(self):
        self.reduce(self.reducing_strategy, self.reducing_args)

    def compute_lazy(self):
        if self.grouping_attributes is not None:
            self.group_by(self.grouping_attributes)

            if len(self.filtering_columns) > 0:
                self.filter_now()

            if self.reducing_strategy is not None:
                self.reduce_now()

    def nunique(self, var):
        return self._nunique[var]

    def _update_nunique(self):
        self._nunique = {}
        for col in self.dataframe.columns:
            if col in self.input_cols or col in self.encoded_input_cols:
                continue
            self._nunique[col] = self.dataframe[col].nunique()
        if self.labels is not None:
            for col in self.labels.columns:
                self._nunique[col] = self.labels[col].nunique()
