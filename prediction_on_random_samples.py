import argparse

import numpy as np
import pandas as pd
import torch

from utils.constants import *
from utils.dataset import Dataset
from utils.serialization import load
from utils.utilities import merge_and_extract

parser = argparse.ArgumentParser(description='Predict random samples')
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-df", "--data-format", help="data format to use as input to the model", default="indices", type=str, choices=["indices", "tfidf"])
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
parser.add_argument("-m", "--model", help="model to use", default=None, type=str, required=True)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
parser.add_argument("-s", "--set", help="set of the dataset", choices=["training", "validation", "test"], default="validation", type=str)
args = parser.parse_args()
print("args:", vars(args))

assert args.group_by is not None # TODO: without groupby not handled

device = "cuda" if torch.cuda.is_available() else "cpu"
input_cols = ["diagnosi", "macroscopia", "notizie"]
model = load(args.model)
model.eval()
torch.set_grad_enabled(False)
classifications, regressions = model.get_validation_classifications(), model.get_validation_regressions()

DATA_COL = "encoded_data"
dataset = Dataset(args.dataset_dir, args.set + "_set.csv", input_cols)
dataset.add_encoded_column(model.encode_report, DATA_COL, args.max_length)
dataset.set_classifications(classifications)
dataset.set_regressions(regressions)
dataset.set_labels_codec(model.labels_codec)
dataset.encode_labels()

multi_layer = False
if args.group_by is not None:
    dataset.lazy_group_by(args.group_by)
    dataset.compute_lazy()
    multi_layer = True

if args.data_seed is not None:
    np.random.seed(args.data_seed)

data, labels = dataset.get_data(args.data_format), dataset.get_labels()

while True:
    index = np.random.randint(0,len(data))
    print("\nrandom index: {}".format(index))
    encoded_record = torch.tensor(data[index].astype(np.int16)).unsqueeze(0)
    record_labels = labels.loc[index]
    out = model(encoded_record, explain=True)
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
        if 'reports_importance' in out:
            reports_importance = out['reports_importance'][0][:len(record)]
            print("importance of reports: {}".format(reports_importance))
            print("most important report (0-based index): {}".format(np.argmax(reports_importance)))
            if 'tokens_importance' in out:
                tokens_importance = out['tokens_importance'][0][:len(record)]
                for i, report in enumerate(record):
                    report_tokens_str = model.tokenizer.tokenize(model.preprocessor.preprocess(report))
                    print("report {}".format(i))
                    print(list(zip(report_tokens_str, tokens_importance[i].cpu().numpy())))
                    print()
            else:
                print("this model does not support insights on importance of tokens")
        else:
            print("this model does not support insights on importance of reports")
        print()

    for reg_var in regressions:
        print(reg_var)
        encoded_prediction = out[reg_var].item()
        prediction = model.labels_codec.codecs[reg_var].decode(encoded_prediction)
        print("prediction:  {}, encoded prediction:  {}".format(prediction, encoded_prediction))
        if not pd.isnull(record_labels[reg_var]):
            encoded_grth = record_labels[reg_var].item()
            grth = model.labels_codec.codecs[reg_var].decode(encoded_grth)
            print("groundtruth: {}, encoded groundtruth: {}".format(grth, encoded_grth))
        else:
            print("no groundtruth available")
        if 'all_features' in out:  # TODO: make compatible with new source code
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
