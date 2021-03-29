import pickle
import re
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch

from utils.cache import caching
from utils.labels_codec import LabelsCodecFactory
from utils.serialization import load
from utils.utilities import replace_nulls, merge_and_extract, to_gpu_if_available


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
        self.encoded_data_column = None

    def set_input_cols(self, input_cols):
        self.input_cols = input_cols
        replace_nulls(self.dataframe, {col: "" for col in input_cols})

    def set_encoded_data_column(self, column_name):
        if type(self.dataframe) == list:
            raise Exception("cannot execute operation: dataframe is already grouped")
        if column_name not in self.dataframe.columns:
            raise Exception("column '{}' does not exists".format(column_name))
        if self.encoded_data_column is not None:
            raise Exception("encoded data column already exists")
        self.encoded_data_column = column_name

    def add_encoded_column(self, full_pipe, new_column_name, max_length=None):

        if len(self.input_cols) == 0:
            raise Exception("input columns not set")

        self.full_pipe = full_pipe

        def _df_to_data(dataframe, pipeline, cols):
            if type(dataframe) == list:
                return [_df_to_data(d, pipeline, cols) for d in dataframe]

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

    # TODO: rename this function
    def prepare_for_training(self, classifications=[], regressions=[], transformations={}):
        self.classifications, self.regressions = classifications, regressions
        self.transformations = transformations
        self._update_nunique()

    def set_classifications(self, classifications):
        assert len(classifications) > 0
        self.classifications = classifications
        self._update_nunique()

    def set_regressions(self, regressions):
        assert len(regressions) > 0
        self.classifications = regressions
        self._update_nunique()

    def set_transformations(self, transformations):
        assert len(transformations) > 0
        self.classifications = transformations

    def _update_columns_codec(self):
        # unique_values = {col: self.dataframe[col].dropna().unique() for col in self.classifications + self.regressions}
        self.columns_codec = LabelsCodecFactory.from_transformations(self.classifications, self.regressions, self.transformations)

    def get_columns_codec(self):
        if self.columns_codec is None:
            self._update_columns_codec()
        return self.columns_codec

    def set_columns_codec(self, columns_codec):
        self.columns_codec = columns_codec

    def encode_labels(self):
        if self.columns_codec is None:
            self._update_columns_codec()
        for column in self.classifications:
            self.dataframe[column] = self.columns_codec[column].encode_batch(self.dataframe[column])
        for column in self.regressions:
            self.dataframe[column] = self.columns_codec[column].encode_batch(self.dataframe[column])
        self._update_nunique()

    def get_labels_columns(self):
        return self.classifications + self.regressions

    # TODO: very slow, speedup
    def get_labels(self):
        if type(self.dataframe) == list:
            cols = self.get_labels_columns()
            tmp = [df.loc[:,cols] for df in self.dataframe]
            l = [t.head(1) for t in tmp]
            return pd.concat(l)
        return self.dataframe[self.get_labels_columns()]

    def get_data(self, column_name, multi_layer=False):
        if type(self.dataframe) == list:
            data = [list(df[column_name].values) for df in self.dataframe]
            if not multi_layer:
                data = [d[0] for d in data]
            return data
        return list(self.dataframe[column_name].values)

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

        self.index_to_key, self.key_to_index, self.dataframe = _group_by(self.dataframe, attributes)

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
                    model = to_gpu_if_available(model)
                    model.eval()
                    self.add_encoded_column(model.encode_report, filter_args["encoded_data_column"], filter_args["max_length"])
                def _same_year(df, *args):
                    return df['anno_referto'].values == df['anno_diagnosi'].values
                def _with_classifier(df, fa):
                    tot_acceptable = 0
                    tot_unacceptable = 0
                    tot_forced_accept = 0
                    encoded_data_column = fa["encoded_data_column"]
                    torch_reports = [torch.tensor(report, device=model.current_device()) for report in df[encoded_data_column]]
                    batch_size = 64
                    filter_result = []
                    for i in range(0, len(torch_reports), batch_size):
                        batch = torch_reports[i: min(i+batch_size, len(torch_reports))]
                        filter_result.append( model(batch)['sede_icdo3'].argmax(dim=1).cpu().numpy().astype(bool) )
                    print("tot_acceptable: {}, tot_unacceptable: {}".format((np.concatenate(filter_result)).sum(), (~np.concatenate(filter_result)).sum()))
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
            tot_acceptable = 0
            tot_unacceptable = 0
            tot_forced_accept = 0
            tot_groups = 0
            tot_forced_groups = 0
            for group in self_dataframe:
                tot_groups += 1
                mask = group[filtering_columns].prod(axis=1).astype(bool)
                tot_acceptable += sum(mask)
                tot_unacceptable += sum(~mask)
                if any(mask):
                    dataframe.append(group[mask])
                else:
                    dataframe.append(group)
                    tot_forced_groups += 1
                    tot_forced_accept += sum(~mask)

            print("tot_acceptable: {}, tot_unacceptable: {}, forced: {}".format(tot_acceptable, tot_unacceptable, tot_forced_accept))
            print("tot_groups: {}, tot_forced_groups: {}".format(tot_groups, tot_forced_groups))
            return dataframe

        self.dataframe = _filter(self.dataframe, self.filtering_columns)

    def reduce(self, reduce_strategy, reduce_args=None):
        _self = self
        def _reduce(self_dataframe, reduce_strat, r_args):
            if callable(reduce_strat):
                reduce_fn = reduce_strat
            elif type(reduce_strat) == str:
                def _most_recent(group_df, ra): # DEPRECATED
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
                def _concatenate(group_df, ra):
                    if len(_self.input_cols) == 0:
                        raise Exception("input columns not set")
                    if _self.encoded_data_column is None:
                        raise Exception("encoded data column not set")
                    if len(group_df) == 1:
                        return group_df
                    values = np.concatenate(group_df[_self.encoded_data_column].values)
                    if ra is not None and "max_length" in ra:
                        values = values[:ra["max_length"]]
                    group_df.at[group_df.index[0], _self.encoded_data_column] = values
                    return group_df.head(1)
                reduce_dict = {"most_recent": _most_recent, "concatenate": _concatenate}
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

    def limit(self, n):
        if self.grouping_attributes is None:
            raise NotImplementedError("limit not implemented without group by")
        self.dataframe = self.dataframe[:n]
