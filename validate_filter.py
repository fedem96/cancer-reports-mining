import argparse
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

from utils.constants import *
from utils.dataset import Dataset
from utils.serialization import load
from utils.utilities import merge_and_extract, to_gpu_if_available

parser = argparse.ArgumentParser(description='Experiment to validate the filter')
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, OLD_DATASET), type=str)
parser.add_argument("-ds", "--data-seed", help="seed for random data shuffling", default=None, type=int)
parser.add_argument("-m", "--model", help="model to use", default=None, type=str, required=True)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
parser.add_argument("-s1", "--set1", help="first set of the dataset", choices=["training", "validation", "test", "unsupervised"], default="validation", type=str)
parser.add_argument("-s2", "--set2", help="second set of the dataset", choices=["training", "validation", "test", "unsupervised"], default="unsupervised", type=str)
args = parser.parse_args()
print("args:", vars(args))

input_cols = ["diagnosi", "macroscopia", "notizie"]
model = load(args.model)
model.eval()
model = to_gpu_if_available(model)
device = model.current_device()
print("model device:", device)
torch.set_grad_enabled(False)

DATA_COL = "encoded_data"
dataset1 = Dataset(os.path.join(args.dataset_dir, args.set1 + "_set.csv"))
dataset1.set_input_cols(input_cols)
dataset1.set_classifications(["sede_icdo3"])
dataset1.add_encoded_column(model.encode_report, DATA_COL, args.max_length)
dataset1.set_columns_codec(model.labels_codec)
dataset1.encode_labels()
data1, labels1 = dataset1.get_data(DATA_COL), dataset1.get_labels().reset_index(drop=True)

dataset2 = Dataset(os.path.join(args.dataset_dir, args.set2 + "_set.csv"))
dataset2.set_input_cols(input_cols)
dataset2.add_encoded_column(model.encode_report, DATA_COL, args.max_length)
# dataset2.set_columns_codec(model.labels_codec)
# dataset2.encode_labels()
data2, labels2 = dataset2.get_data(DATA_COL), dataset2.get_labels().reset_index(drop=True)

if args.data_seed is not None:
    np.random.seed(args.data_seed)


def padding_tensor(sequences, max_len):
    num = len(sequences)
    out_dims = (num, max_len)
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    mask = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
        mask[i, :length] = 1
    return out_tensor, mask

data1 = padding_tensor([torch.tensor(report, device=device, dtype=torch.int16) for report in data1], 100)[0]
data2 = padding_tensor([torch.tensor(report, device=device, dtype=torch.int16) for report in data2], 100)[0]

cancer_ids = set(dataset1.dataframe.id_tot.values)
mask = labels1[labels1['sede_icdo3'].astype("Int64") == 1].index.values
data1 = data1[mask]
preds1 = model(data1)['sede_icdo3']
l1 = torch.nn.functional.softmax(preds1, dim=1)[:,1].cpu().tolist()
plt.hist(l1, bins=20, alpha=0.5)
print("number of breast cancer samples: " + str(len(data1)))

not_cancer = ~dataset2.dataframe.id_tot.isin(cancer_ids)
all_zero = dataset2.dataframe.encoded_data.apply(lambda x: x.sum()==0 if (isinstance(x, Iterable) and all(pd.notna(x))) else True)
data2 = data2[not_cancer & (~all_zero)]
perm2 = np.random.permutation(len(data2))
data2 = data2[perm2[:len(data1)]]
preds2 = model(data2)['sede_icdo3']
l2 = torch.nn.functional.softmax(preds2, dim=1)[:,1].cpu().tolist()
plt.hist(l2, bins=20, alpha=0.5)
print("number of non-cancer samples: " + str(len(data2)))

plt.legend(['breast cancer', 'non-cancers'])
plt.xlabel('breast cancer probability')
plt.ylabel("occurences (total samples: 2045)")
plt.show()

print("percentage of rejected non-cancers reports: {}%".format(100 * len([l for l in l2 if l<0.5]) / len(l2)))

n = 1000
perc = sum([1 for i in range(n) if l1[np.random.randint(0,len(l1))] > max([l2[np.random.randint(0,len(l2))] for _ in range(10)])]) / n
print("1 breast cancer VS 10 non-cancers: {}% times the cancer report as a higher probability".format(perc))

y_true = np.concatenate([np.ones(len(l1)), np.zeros(len(l2))])
y_scores = np.concatenate([np.array(l1), np.array(l2)])
precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
plt.title('filter classifier')
plt.show()

f1_scores = [2 * p*r/(p+r) for p,r in zip(precision, recall)]
plt.plot(thresholds, precision[:-1])
plt.plot(thresholds, recall[:-1])
plt.plot(thresholds, f1_scores[:-1])
plt.legend(['precision', 'recall', 'f1'])
plt.xlabel('threshold')
plt.ylabel('score')
plt.title('filter classifier')
plt.show()