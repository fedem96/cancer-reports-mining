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
    "val": Dataset(dataset_dir, VALIDATION_SET, input_cols, tokenizer_file_name),
    "test": Dataset(dataset_dir, TEST_SET, input_cols, tokenizer_file_name)
}
training, validation, test = sets["train"], sets["val"], sets["test"]
t = training.tokenizer
p = training.preprocessor

for set_name in ["train", "val", "test"]:
    dataset = sets[set_name]
    dataset.lazy_group_by('id_paz')
    # if args.filter is not None:
    #     dataset.lazy_filter(args.filter, args.filter_args)
    # dataset.lazy_concatenate_reports()
    dataset.compute_lazy()

tr_labels = pd.concat([df.loc[:,labels_cols].head(1) for df in training.dataframe]).reset_index(drop=True)
val_labels = pd.concat([df.loc[:,labels_cols].head(1) for df in validation.dataframe]).reset_index(drop=True)
te_labels = pd.concat([df.loc[:,labels_cols].head(1) for df in test.dataframe]).reset_index(drop=True)
tr_txts = p.preprocess_batch([" ".join(df.loc[:,input_cols].values.flatten().tolist()) for df in training.dataframe])
val_txts = p.preprocess_batch([" ".join(df.loc[:,input_cols].values.flatten().tolist()) for df in validation.dataframe])
te_txts = p.preprocess_batch([" ".join(df.loc[:,input_cols].values.flatten().tolist()) for df in test.dataframe])


def num_substr(substrs, txts):
    if type(substrs) == str:
        substrs = [substrs]
    s = sum([any([s in txt for s in substrs]) in txt for txt in txts])
    return s, s/len(txts)


def search_context(expr, txt, context=10):
    if tuple != type(context) != list:
        context = (context, context)
    return list(map(lambda m: txt[max(0, m.span()[0] - context[0]): m.span()[1] + context[1]], re.finditer(expr, txt)))


def search_after(expr, txt, context=10):
    assert type(context) == int
    return list(map(lambda m: txt[m.span()[1]: m.span()[1] + context], re.finditer(expr, txt)))


def evaluate(txts, labels, var, predictor):
    labels = labels.reset_index()
    num_correct = sum([predictor(txts[i]) == labels[var].values[i] for i in labels[labels[var].notna()].index])
    tot_predicted = sum([predictor(txts[i]) is not None for i in labels[labels[var].notna()].index])
    tot_notna = labels[var].notna().sum()
    return num_correct/tot_notna, num_correct/tot_predicted, tot_predicted/tot_notna, tot_predicted, tot_notna


def extract_value(r, value_regex, stop_regex):
    end = len(r) if not re.search(stop_regex, r) else re.search(stop_regex, r).span()[0]
    result = re.search(value_regex, r[:end])
    if result is None:
        return None
    return int(result.group(1))


def predict(markers, s, value_regex, stop_regex, context):
    for marker in markers:
        results = search_after(marker, s, context)
        values = list(
            filter(
                lambda v: v is not None,
                map(
                    lambda r: extract_value(r, value_regex, stop_regex),
                    filter(lambda r: re.search(value_regex, r), results)
                )
            )
        )
        if len(values) > 0:
            return max([0, *values])
    return None


def predict_er(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    markers = [r" er\s?:", r" er\s.{,10}:", " er ", "estrogen[i|o].{,10}:", "estrogen[i|o]"]
    value_regex = "(\d?\d?\d) %"
    stop_regex = "ki67|mib1|cerb|pgr|progest|;"
    return predict(markers, s, value_regex, stop_regex, 60)


def predict_pgr(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    markers = ["pgr?.{,10}:", "pgr?", "progest[in|erone].{,10}:", "progest[in|erone]"]
    value_regex = "(\d?\d?\d) %"
    stop_regex = r"ki67|mib1|cerb|estrogen|\ser\s;"
    return predict(markers, s, value_regex, stop_regex, 60)


def predict_ki67(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    markers = [r"ki67\s?.{,10}:", r"mib1\s?.{,10}:", "ki67", "mib1"]
    value_regex = "(\d?\d?\d) %"
    stop_regex = r"cerb|pgr|\ser\s|progest|estrog"
    return predict(markers, s, value_regex, stop_regex, 60)


def predict_cerb(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    markers = ["cerb.*score", r"cerb\s?.{,10}:", r"cerb\s?.{,20}:", r"cerb\s?.{,30}:"]
    value_regex = "(\d?\d?\d)"
    stop_regex = r"pgr|\ser\s|progest|estrog|ki67|mib1|;|\."
    result = predict(markers, s, value_regex, stop_regex,60)
    if result is not None:
        return result
    return None
    # return 0   # with 0 as default value, the overall accuracy increases, while the accuracy on predicted only slightly decreases


def predict_mib1(s):
    s = s.replace("negativo", "0 %").replace("-", "0 %")
    markers = [r"mib1\s?.{,10}:", r"mib1\s?.{,20}:", "mib1"]
    value_regex = "(\d?\d?\d) %"
    stop_regex = r"cerb|pgr|\ser\s|progest|estrog|ki67"
    return predict(markers, s, value_regex, stop_regex,60)


# TODO: find good default values for predictors by examining distribution when they do not find any marker
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
    print("TEST:  tot accuracy: {:.4f}, accuracy on predicted: {:.4f}, predicted/tot: {:.4f}, predicted: {}, tot: {}".format(*evaluate(te_txts, te_labels, var, predictors[var])))
    print()


if args.print_samples:
    samples_to_print = 20
    for var, expr in zip(["recettori_estrogeni", "recettori_progestin", "ki67", "cerb", "mib1"], ["\se\s?\.?\s?r|estrog", "(pgr?|progest)", "ki67", "cerb", "mib1"]):
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
