import copy
import pickle
import sys
from multiprocessing.pool import Pool

import numpy as np
import pandas as pd
import torch

from utils.cache import caching
from utils.constants import *
from utils.labels_codec import LabelsCodecFactory
from utils.preprocessing import Preprocessor
from utils.serialization import load
from tokenizing.tokenizer import Tokenizer
from utils.utilities import replace_nulls, merge_and_extract, to_gpu_if_available


class Dataset:

    def __init__(self, directory, set_name, input_cols, tokenizer_file_name, max_report_length=None, max_record_size=None):
        self.directory = directory
        self.set_name = set_name
        self.csv_file = os.path.join(directory, set_name)
        self.dataframe = pd.read_csv(self.csv_file)
        self.input_cols = input_cols
        self.encoded_input_cols = []
        self.classifications = []
        self.regressions = []
        self.transformations = None
        self.labels = None
        self.labels_codec = None
        self.index_to_key = None
        self.key_to_index = None
        self._nunique = None
        self._update_nunique()
        self.grouping_attributes = None
        self.filtering_columns = []
        self.must_concatenate = False
        self.max_total_length = None
        self.encoded_data_column = None

        self.preprocessor = Preprocessor.get_default()
        self.tokenizer = Tokenizer().load(os.path.join(directory, tokenizer_file_name))

        self.max_report_length = max_report_length
        self.encoded_data_column = "encoded_data"
        replace_nulls(self.dataframe, {col: "" for col in input_cols})
        # self.add_encoded_column(self.full_pipe, self.encoded_data_column, max_report_length)
        self.max_record_size = max_record_size

        if self.tokenizer.num_tokens() >= 2 ** 15:
            if self.tokenizer.num_tokens() >= 2 ** 16:
                raise ValueError(f"ERROR: too much tokens for 16 bit: {self.tokenizer.num_tokens()} (maximum is {2**16-1})")
            print("WARNING: some token will be negative in pytorch because of lacking of torch.uint16", file=sys.stderr)

    def full_pipe(self, report):
        return self.tokenizer.tokenize(self.preprocessor.preprocess(report), encode=True)

    def add_encoded_column(self, full_pipe, new_column_name, max_length=None):

        if len(self.input_cols) == 0:
            raise Exception("input columns not set")

        def _df_to_data(dataframe, pipeline, cols, *args):
            if type(dataframe) == list:
                return [_df_to_data(d, pipeline, cols) for d in dataframe]

            reports = merge_and_extract(dataframe, cols)

            try:
                with Pool(6) as pool:
                    data = pool.map(pipeline, reports, chunksize=len(reports)//6)  # since pipeline is going to be pickled, I can't use a lambda
            except (pickle.PicklingError, RuntimeError, Exception):
                print("Cannot pickle pipeline, using sequential version", file=sys.stderr)
                data = [pipeline(r) for r in reports]
            return data

        new_column = caching(_df_to_data)(self.dataframe, full_pipe, self.input_cols, self.tokenizer.n_grams, self.tokenizer.codec.num_tokens())
        if type(self.dataframe) == list:
            for df, new_col in zip(self.dataframe, new_column):
                df[new_column_name] = new_col
        else:
            self.dataframe[new_column_name] = new_column
        if max_length is not None:
            self.dataframe[new_column_name] = [sequence[:max_length] for sequence in self.dataframe[new_column_name]]
        self.encoded_input_cols.append(new_column_name)

    def copy_column(self, src_col, dest_col):
        assert src_col in self.dataframe.columns        # src_col must already be in the dataset
        assert dest_col not in self.dataframe.columns   # dest_col must not already be in the dataset
        self.dataframe[dest_col] = self.dataframe[src_col]

    def set_classifications(self, classifications):
        self.classifications = copy.deepcopy(classifications)
        self._update_nunique()

    def set_regressions(self, regressions):
        self.regressions = copy.deepcopy(regressions)
        self._update_nunique()

    def set_transformations(self, transformations):
        self.transformations = transformations

    def create_labels_codec(self, prepare_labels_steps):
        assert self.labels_codec is None  # labels codec already exists
        self.labels_codec = LabelsCodecFactory.from_transformations(self.classifications, self.regressions, prepare_labels_steps)
        return self.labels_codec

    def get_labels_codec(self):
        return self.labels_codec

    def set_labels_codec(self, labels_codec):
        self.labels_codec = labels_codec

    def encode_labels(self):
        assert self.labels_codec is not None  # labels codec must be created or set first
        for column in self.classifications:
            self.dataframe[column] = self.labels_codec[column].encode_batch(self.dataframe[column])
        for column in self.regressions:
            self.dataframe[column] = self.labels_codec[column].encode_batch(self.dataframe[column])
        self._update_nunique()

    def remove_examples_without_labels(self):
        mask = np.ones(len(self.dataframe)).astype(bool)
        for column in self.classifications:
            mask = mask & self.dataframe[column].isna().values
        for column in self.regressions:
            mask = mask & self.dataframe[column].isna().values

        self.dataframe.drop(self.dataframe[mask].index, inplace=True)

    def get_labels_columns(self):
        return self.classifications + self.regressions

    # TODO: very slow, speedup
    def get_labels(self):
        def _get_labels(dataframe, labels_columns, nuniques):
            if type(dataframe) == list:
                cols = labels_columns
                tmp = [df.loc[:,cols] for df in dataframe]
                l = [t.head(1) for t in tmp]
                return pd.concat(l).reset_index(drop=True)
            return self.dataframe[labels_columns].reset_index(drop=True)
        return caching(_get_labels)(self.dataframe, self.get_labels_columns(), [self.nunique(col) for col in self.get_labels_columns()])

    def get_data(self, data_type="indices", column_name=None):
        if column_name is None:
            column_name = self.encoded_data_column
        print(data_type)
        if data_type == "indices":
            return self.get_data_as_tokens_indices(column_name)
        elif data_type == "bag":
            data = self.get_data_as_tfidf_vectors(column_name)
            return torch.sparse_coo_tensor(data.indices(), data.values() > 0, data.shape).int()
        elif data_type == "tfidf":
            return self.get_data_as_tfidf_vectors(column_name)
        return "unknown data type: '{}'".format(data_type)

    def get_data_as_tokens_indices(self, column_name):
        def _get_data_as_tokens_indices(dataframe, col_name, self_max_record_size, must_concatenate):  # must_concatenate is required for caching
            # the number of tokens is between 40000 and 50000: 16 bit for indices are enough
            if type(dataframe) == list:
                records = [[report.astype(np.uint16) for report in record[col_name].values] for record in dataframe]
                records = [[report[report != 0] if len(report[report != 0]) > 0 else np.array([0]) for report in record] for record in records]
                reports_lengths = [[len(report) for report in record] for record in records]
                records_sizes = [len(record) for record in records]
                max_report_length = max([max(lengths) for lengths in reports_lengths])
                max_record_size = max(records_sizes)
                if self_max_record_size is not None and self_max_record_size < max_record_size:
                    max_record_size = self_max_record_size
                data = np.stack(                                                                        # stack records to create dataset
                    [
                        np.pad(                                                                         # pad record
                                np.stack([                                                              # stack reports to create record
                                    np.pad(report, (0,max_report_length))[:max_report_length] # pad report
                                    for report in record
                                    if (report != 0).sum() > 0
                                ] + [np.zeros((max_report_length), dtype=np.uint16)]
                            ), ((0,max_record_size),(0,0)))[:max_record_size]
                        for record in records
                    ]
                )
                return data # data.shape: (num_records, num_reports, num_tokens)
            # without group by, each row is treated as a single-report record
            max_report_length = max([len(report) for report in dataframe[col_name].values])
            return np.expand_dims(np.stack([np.pad(report.astype(np.uint16), (0,max_report_length-len(report))) for report in self.dataframe[col_name].values]), 1)
        return caching(_get_data_as_tokens_indices)(self.dataframe, column_name, self.max_record_size, self.must_concatenate)

    def get_data_as_tfidf_vectors(self, column_name):
        def _get_data_as_tfidf_vectors(dataframe, col_name, must_concatenate): # dataframe and must_concatenate are required for caching
            indices_data = self.get_data_as_tokens_indices(col_name)

            records_indexes = []
            reports_indexes = []
            tokens_indexes = []
            tokens_values = []
            for record_index, encoded_record in enumerate(indices_data):
                non_padding_reports = encoded_record.sum(axis=1) != 0
                record = self.tokenizer.decode_ndarray(encoded_record[non_padding_reports])
                for report_non_padding_index, report_original_index in enumerate(np.where(non_padding_reports)[0]):
                    report = record[report_non_padding_index]
                    for token in report:
                        if token != "":
                            records_indexes.append(record_index)
                            reports_indexes.append(report_original_index.item())
                            tokens_indexes.append(self.tokenizer.encode_token(token))
                            tokens_values.append(self.tokenizer.get_idf(token, encode=True))
            indexes = [records_indexes, reports_indexes, tokens_indexes]
            return torch.sparse_coo_tensor(indexes, tokens_values, (*indices_data.shape[:2], self.tokenizer.num_tokens()+1)).coalesce()
        return caching(_get_data_as_tfidf_vectors)(self.dataframe, column_name, self.must_concatenate)

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
                    encoded_data_column = fa["encoded_data_column"]
                    max_report_length = fa["max_length"] or max([len(report) for report in df[encoded_data_column]])
                    reports = np.stack([
                        np.pad(report[:max_report_length].astype(np.uint16), (0, max(0, max_report_length - len(report))))
                        for report in df[encoded_data_column]
                    ])
                    torch_reports = torch.tensor(reports.astype(np.int16), device=model.current_device()).unsqueeze(1)
                    batch_size = 64
                    filter_result = []
                    for i in range(0, len(torch_reports), batch_size):
                        batch = torch_reports[i: min(i+batch_size, len(torch_reports))]
                        filter_result.append( model(batch, pool_reports=False)['sede_icdo3'].argmax(dim=3).squeeze().cpu().numpy().astype(bool) )
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

    def filter(self):
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

    def concatenate_reports(self, max_total_length=None):
        def _concatenate_reports(self_dataframe, input_cols, encoded_data_column, max_tot_len):
            if len(input_cols) == 0:
                raise Exception("input columns not set")
            if encoded_data_column is None:
                raise Exception("encoded data column not set")
            def _concatenate(group_df, max_tot_l):
                if len(group_df) == 1:
                    return group_df
                values = np.concatenate(group_df[encoded_data_column].values)
                if max_tot_len is not None:
                    values = values[:max_tot_l]
                group_df.at[group_df.index[0], encoded_data_column] = values
                return group_df.head(1)
            dataframe = []
            for group in self_dataframe:
                dataframe.append(_concatenate(group, max_tot_len))
            return dataframe
        self.dataframe = caching(_concatenate_reports)(self.dataframe, self.input_cols, self.encoded_data_column, max_total_length)

    def lazy_concatenate_reports(self, max_total_length=None):
        if self.must_concatenate:
            raise Exception("Concatenate already specified: cannot concatenate twice")
        self.must_concatenate = True
        self.max_total_length = max_total_length

    def compute_lazy(self):
        if self.grouping_attributes is not None:
            self.group_by(self.grouping_attributes)

            if len(self.filtering_columns) > 0:
                self.filter()

            if self.must_concatenate:
                self.concatenate_reports(self.max_total_length)

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
