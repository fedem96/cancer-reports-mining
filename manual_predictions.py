import argparse
from pprint import pprint
import os
import re

import pandas as pd

from utils.constants import *
from utils.dataset import Dataset

parser = argparse.ArgumentParser(description='Manually predict the variables that require finding a number in the text')
parser.add_argument("-ps", "--print-samples", help="whether to print samples or not, in the end", default=False, action="store_true")
args = parser.parse_args()

dataset_dir = os.path.join(DATASETS_DIR, NEW_DATASET)
input_cols = ["diagnosi", "macroscopia"]
labels_cols = ['tipo_T', 'metastasi', 'modalita_T', 'modalita_N', 'stadio_T', 'stadio_N', 'grading', 'dimensioni', 'recettori_estrogeni', 'recettori_progestin', 'numero_sentinella_asportati', 'numero_sentinella_positivi', 'mib1', 'cerb', 'ki67']

tokenizer_file_name = f"tokenizer-1gram.json"
sets = {
    "train": Dataset(dataset_dir, TRAINING_SET, input_cols, tokenizer_file_name),
    "val": Dataset(dataset_dir, VALIDATION_SET, input_cols, tokenizer_file_name)
}
training, validation = sets["train"], sets["val"]
t = training.tokenizer
p = training.preprocessor

for set_name in ["train", "val"]:
    dataset = sets[set_name]
    dataset.lazy_group_by('id_paz')
    # if args.filter is not None:
    #     dataset.lazy_filter(args.filter, args.filter_args)
    # dataset.lazy_concatenate_reports()
    dataset.compute_lazy()

tr_labels = pd.concat([df.loc[:,labels_cols].head(1) for df in training.dataframe]).reset_index(drop=True)
val_labels = pd.concat([df.loc[:,labels_cols].head(1) for df in validation.dataframe]).reset_index(drop=True)
tr_txts = p.preprocess_batch([" ".join(df.loc[:,input_cols].values.flatten().tolist()) for df in training.dataframe])
val_txts = p.preprocess_batch([" ".join(df.loc[:,input_cols].values.flatten().tolist()) for df in validation.dataframe])


def num_substr(substrs, txts):
    if type(substrs) == str:
        substrs = [substrs]
    s = sum([any([s in txt for s in substrs]) in txt for txt in txts])
    return s, s/len(txts)


def search_context(expr, txt, context=5):
    if tuple != type(context) != list:
        context = (context, context)
    return list(map(lambda m: txt[max(0, m.span()[0] - context[0]): m.span()[1] + context[1]], re.finditer(expr, txt)))


def evaluate(txts, labels, var, predictor):
    num_correct = sum([predictor(txts[i]) == labels[var].values[i] for i in labels[labels[var].notna()].index])
    tot_predicted = sum([predictor(txts[i]) is not None for i in labels[labels[var].notna()].index])
    tot_notna = labels[var].notna().sum()
    return num_correct/tot_notna, num_correct/tot_predicted, tot_predicted/tot_notna, tot_predicted, tot_notna


def predict_er(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    results = search_context("\se\s?\.?\s?r\s", s, (0,50))
    stop_regex = "p\s?\.?\s?g\s?\.?\s?r|progest"
    # values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r[:len(r) if re.search(stop_regex, s) is None else re.search(stop_regex, s).span()[0]]).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    if len(values) == 0:
        results = search_context("estrog", s, (0,50))
        # values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r[:len(r) if re.search(stop_regex, s) is None else re.search(stop_regex, s).span()[0]]).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
        values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
        if len(values) == 0:
            return None
    return max([0, *values])


def predict_pgr(s):
    # k =  re.search("ki67", s)
    # if k is not None:
    #     s = s[:k.span()[0]]
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    results = search_context("p\s?\.?\s?g\s?\.?\s?r", s, (0,50))
    values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r[:len(r) if "ki67" not in r else re.search("ki67", s).span()[0]]).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    if len(values) == 0:
        results = search_context("progest", s, (0,50))
        values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r[:len(r) if "ki67" not in r else re.search("ki67", s).span()[0]]).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
        if len(values) == 0:
            return None
    return max([0, *values])


def predict_ki67(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    results = search_context("ki67", s, (0,50))
    values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    if len(values) == 0:
        return None
    return max([0, *values])


def predict_cerb(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    results = search_context("cerb", s, (0,50))
    values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    if len(values) == 0:
        return None
    return max([0, *values])


def predict_mib1(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    results = search_context("mib1", s, (0,50))
    values = list(filter(lambda v: v is not None, map(lambda r: int(re.search("(\d?\d?\d) %", r).group(1)) if re.search("(\d?\d?\d) %",r) is not None else None,results)))
    if len(values) == 0:
        return None
    return max([0, *values])


predictors = {
    "recettori_estrogeni": predict_er,
    "recettori_progestin": predict_pgr,
    "ki67": predict_ki67,
    "cerb": predict_cerb,
    "mib1": predict_mib1
}

for var in ["recettori_estrogeni", "recettori_progestin", "ki67", "cerb", "mib1"]:
    print(var.upper())
    print("TRAIN: tot accuracy: {:.4f}, accuracy on predicted: {:.4f}, predicted/tot: {:.4f}, predicted: {}, tot: {}".format(*evaluate(tr_txts, tr_labels, var, predictors[var])))
    print("VAL:   tot accuracy: {:.4f}, accuracy on predicted: {:.4f}, predicted/tot: {:.4f}, predicted: {}, tot: {}".format(*evaluate(val_txts, val_labels, var, predictors[var])))
    print("TR+VA: tot accuracy: {:.4f}, accuracy on predicted: {:.4f}, predicted/tot: {:.4f}, predicted: {}, tot: {}".format(*evaluate(tr_txts+val_txts, pd.concat([tr_labels,val_labels]), var, predictors[var])))
    print()


if args.print_samples:
    samples_to_print = 20
    for var, expr in zip(["recettori_estrogeni", "recettori_progestin", "ki67", "cerb", "mib1"], ["\se\s?\.?\s?r|estrog", "(p\s?\.?\s?g\s?\.?\s?r|progest)", "ki67", "cerb", "mib1"]):
        if var != "cerb":
            continue
        print(var.upper())
        print(f"{samples_to_print} correct examples")
        pprint([(i,search_context(expr, tr_txts[i], (0,50)),predictors[var](tr_txts[i]),tr_labels[var].values[i]) for i in tr_labels[tr_labels[var].notna()].index if predictors[var](tr_txts[i]) == tr_labels[var].values[i]][:20])
        print(f"\n{samples_to_print} wrong examples")
        pprint([(i,search_context(expr, tr_txts[i], (0,50)),predictors[var](tr_txts[i]),tr_labels[var].values[i]) for i in tr_labels[tr_labels[var].notna()].index if predictors[var](tr_txts[i]) is not None and predictors[var](tr_txts[i]) != tr_labels[var].values[i]][:20])
        print(f"\n{samples_to_print} not predicted examples")
        pprint([(i,tr_txts[i],search_context(expr, tr_txts[i], (0,50)),predictors[var](tr_txts[i]),tr_labels[var].values[i]) for i in tr_labels[tr_labels[var].notna()].index if predictors[var](tr_txts[i]) is None][:20])
        print()
