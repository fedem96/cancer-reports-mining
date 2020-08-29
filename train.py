from collections import defaultdict
from multiprocessing import Pool
from timeit import default_timer as timer

import numpy as np

from utils.constants import *
from utils.preprocessing import preprocess
from utils.tokens_handler import TokensHandler
from utils.utilities import *

# dataset
dataset = read_csv(TRAINING_SET_FILE, 'utf-8', types={"stadio": 'Int64', "dimensioni": float, "sede_icdo3": str, "morfologia_icdo3": str,
                                                      "notizie": str, "diagnosi": str, "macroscopia": str},
                   handle_nans={"notizie": "", "diagnosi": "", "macroscopia": ""})

#dataset = dataset.iloc[:3870]

th = TokensHandler()
reports = dataset[["notizie", "diagnosi", "macroscopia"]].apply(lambda x: preprocess(' '.join(x)), axis=1).values
print("total reports:", len(reports))

print("tokenizing reports")
def f(report):
    return th.get_tokens_idx(report)
with Pool(8) as pool:
    data = pool.map(f, reports)  # since f is going to be pickled, if I use directly th.get_tokens_idx is very worse


print("encoding reports")
encode = defaultdict(lambda: {})
decode = defaultdict(lambda: {})
for attr in ["sede_icdo3", "morfologia_icdo3"]:
    for k in range(len(dataset[attr].unique())):
        v = dataset[attr].unique()[k]
        encode[attr][v] = k
        decode[attr][k] = v
    dataset[attr] = dataset[attr].apply(lambda val: encode[attr][val])

labels = dataset[["stadio", "dimensioni", "sede_icdo3", "morfologia_icdo3"]]

mask = [len(d) > 0 for d in data]
data = [data[d] for d in range(len(data)) if mask[d]]
labels = labels.iloc[mask].reset_index()

# tasks
classifications = ["stadio", "sede_icdo3", "morfologia_icdo3"]  # -> one-hot and cross-entropy
regressions = ["dimensioni"]                                    # -> squared loss??

# model
# from models.emb_max_fc import EmbMaxLin
# model = EmbMaxLin(th.num_tokens()+1, 256, 256, 256)
from models.transformer import Transformer
model = Transformer(th.num_tokens()+1, 256, 8, 256, 256, 0.1)
for cls_var in classifications:
    model.add_classification(cls_var, dataset[cls_var].nunique())

for reg_var in regressions:
    model.add_regression(reg_var)

model.fit(data, labels, 50, 128)
print("finish")

