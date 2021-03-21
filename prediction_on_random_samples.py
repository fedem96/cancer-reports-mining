import argparse
import json

import numpy as np
import pandas as pd
import torch

from utils.constants import *
from utils.dataset import Dataset
from utils.serialization import load
from utils.utilities import merge_and_extract

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-ds", "--data-seed", help="seed for random data shuffling", default=None, type=int)
# parser.add_argument("-f", "--filter",
#                     help="report filtering strategy",
#                     default=None, type=str, choices=['same_year', 'classifier'], metavar='STRATEGY')
# parser.add_argument("-fa", "--filter-args",
#                     help="args for report filtering strategy",
#                     default=None, type=json.loads)
parser.add_argument("-gb", "--group-by",
                    help="list of (space-separated) grouping attributes to make multi-report predictions.",
                    default=None, nargs="+", type=str, metavar=('ATTR1', 'ATTR2'))
# parser.add_argument("-im", "--input-mappings", help="how to map the input", default={}, type=json.loads)
# parser.add_argument("-it", "--input-transformations", help="how to transform the input", default={}, type=json.loads)
parser.add_argument("-m", "--model", help="model to use", default=None, type=str, required=True)
# parser.add_argument("-ma", "--model-args", help="saved model to train", default=None, type=json.loads)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
# parser.add_argument("-rm", "--reduce-mode", help="how to reduce", default=None, type=str)
# parser.add_argument("-rt", "--reduce-type", help="what to reduce", default=None, type=str,
#                     choices=["data", "features", "logits", "eval"])
parser.add_argument("-s", "--set", help="set of the dataset", choices=["training", "validation", "test"], default="validation", type=str)
args = parser.parse_args()
print("args:", vars(args))
# if args.group_by is not None:
#     assert args.reduce_mode in {"data": {"most_recent"}, "features": {"max"}, "logits": {"mean"}, "eval": {"argmax"}}[args.reduce_type]  # TODO: multiple reduce modes

assert args.group_by is not None # TODO: without groupby not handled

device = "cuda" if torch.cuda.is_available() else "cpu"
input_cols = ["diagnosi", "macroscopia", "notizie"]
model = load(args.model)
model.eval()
torch.set_grad_enabled(False)
classifications, regressions = list(model.classifiers.keys()), list(model.regressors.keys())

assert model.reduce_type == "features" # TODO: only features reduce type is handled

DATA_COL = "encoded_data"
dataset = Dataset(os.path.join(args.dataset_dir, args.set + "_set.csv"))
dataset.set_input_cols(input_cols)
dataset.add_encoded_column(model.encode_report, DATA_COL, args.max_length)
dataset.prepare_for_training(classifications, regressions, {}, {}) # TODO: transformations and mappings
dataset.set_columns_codec(model.labels_codec)
dataset.encode_labels()

multi_layer = False
if args.group_by is not None:
    dataset.lazy_group_by(args.group_by)
    dataset.compute_lazy()
    multi_layer = True

if args.data_seed is not None:
    np.random.seed(args.data_seed)

data, labels = dataset.get_data(DATA_COL, multi_layer), dataset.get_labels().reset_index(drop=True)

while True:
    index = np.random.randint(0,len(data))
    print("\nrandom index: {}".format(index))
    encoded_record = [torch.tensor(t, device=device) for t in data[index]]
    record_labels = labels.loc[index]
    out = model([encoded_record])
    record = list(merge_and_extract(dataset.dataframe[index], input_cols))
    print("['" + "',\n'".join(record) + "']")
    for cls_var in classifications:
        print(cls_var)
        prediction_idx = out[cls_var].argmax().item()
        prediction = model.labels_codec.codecs[cls_var].decode(prediction_idx)
        print("prediction:  {}, prediction index:  {}, prediction logits: {}".format(prediction, prediction_idx, out[cls_var].cpu().numpy()))
        if not pd.isnull(record_labels[cls_var]):
            grth_idx = record_labels[cls_var].item()
            grth = model.labels_codec.codecs[cls_var].decode(grth_idx)
            print("groundtruth: {}, groundtruth index: {}".format(grth, grth_idx))
        else:
            print("no groundtruth available")
        if 'all_features' in out:
            record_features = out['all_features'][0] # [0] because we want the first (and only) record of the batch
            for idx_fr, fr in enumerate(model.features_reducers):
                num_equals = [(fr(record_features, dim=0) == record_features[i]).sum().item() for i in range(len(record))]
                if sum(num_equals) == 0:
                    print("this reduce method does not support insights on importance of reports")
                else:
                    print("importance of reports: {}".format(num_equals))
                    print("most important report: {}".format(np.argmax(num_equals)))
        else:
            print("this model does not support insights on importance of reports")

        print()
    print()
