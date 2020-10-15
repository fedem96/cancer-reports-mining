import argparse
from multiprocessing import Pool
from timeit import default_timer as timer

import numpy as np

from utils.constants import *
from utils.labels_codec import LabelsCodec
from utils.preprocessing import *
from utils.tokenizing import TokenCodec, Tokenizer
from utils.utilities import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-c", "--codec", help="token codec filename", default=TOKEN_CODEC, type=str)
parser.add_argument("-i", "--idf", help="Inverse Document Frequencies filename", default=IDF, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-m", "--model", help="model to train", default=None, type=str)
args = parser.parse_args()

train_df = pd.read_csv(os.path.join(args.dataset_dir, TRAINING_SET))
val_df = pd.read_csv(os.path.join(args.dataset_dir, VALIDATION_SET))
test_df = pd.read_csv(os.path.join(args.dataset_dir, TEST_SET))
cols = ["diagnosi", "macroscopia", "notizie"]

p = Preprocessor.get_default()
t = Tokenizer()
tc = TokenCodec().load(os.path.join(args.dataset_dir, args.codec))

print("tokenizing reports and encoding tokens")

with Chronometer(1):
    def f(report): return tc.encode(t.tokenize(p.preprocess(report)))
    def df_to_data(dataframe):
        replace_nulls(dataframe, {col: "" for col in cols})
        reports = merge_and_extract(dataframe, cols)

        with Pool(6) as pool:
            data = pool.map(f, reports)  # since f is going to be pickled, I can't use a lambda
        return data

    train, val, test = [df_to_data(dataframe) for dataframe in [train_df, val_df, test_df]]

print("encoding reports")

with Chronometer(2):
    with open("configs/train_example.json", "rt") as file:
        train_config = json.load(file)


    classifications, regressions = train_config.get("classifications", {}), train_config.get("regressions", {})
    transformations, mappings = train_config.get("transformations", {}), train_config.get("mappings", {})


    def prepare_dataset(dataset, train_config):
        classifications, regressions = train_config.get("classifications", {}), train_config.get("regressions", {})
        transformations, mappings = train_config.get("transformations", {}), train_config.get("mappings", {})
        for column in classifications:
            if column in transformations:
                for transf in transformations[column]:
                    ty = transf["type"]
                    if ty == "regex_sub":
                        for s in transf["subs"]:
                            regex = re.compile(s[0], re.I)
                            dataset[column] = dataset[column].apply(lambda v: s[1] if regex.match(str(v)) else v)
                            #dataset.loc[dataset.index[dataset[column].apply(lambda v: None != regex.match(str(v)))], column] = s[1]
                    elif ty == "filter":
                        dataset.loc[dataset.index[dataset[column].apply(lambda v: v not in transf["valid_set"])], column] = np.NaN
                    else:
                        raise ValueError("invalid transformation '{}' for classification problem".format(ty))
            # dataset[column] = columns_codec[column].encode_batch(dataset[column])


    def get_columns_codec(dataset, train_config):
        classifications = train_config.get("classifications", {})
        for column in classifications:
            if column not in mappings:
                mappings[column] = sorted(dataset[column].dropna().unique())
        return LabelsCodec().from_mappings(mappings)


    for df in[train_df, val_df, test_df]: prepare_dataset(df, train_config)
    columns_codec = get_columns_codec(train_df, train_config)


with Chronometer(3):
    def get_encoded_labels(dataset, columns, columns_to_encode):
        labels = dataset[columns]
        for column in columns_to_encode:
            labels[column] = columns_codec[column].encode_batch(labels[column])
        return labels


    train_labels, val_labels, test_labels = [get_encoded_labels(dataset, classifications + regressions, classifications)
                                                                for dataset in [train_df, val_df, test_df]]

print("creating model")
# model
# from models.emb_max_fc import EmbMaxLin
# model = EmbMaxLin(tc.num_tokens()+1, 256, 256, 256)
from models.transformer import Transformer
# model = Transformer(tc.num_tokens()+1, 256, 8, 256, 1, 0.1)
model = Transformer(tc.num_tokens()+1, 128, 8, 128, 1, 0.1)
for cls_var in classifications:
    model.add_classification(cls_var, train_df[cls_var].nunique())

for reg_var in regressions:
    model.add_regression(reg_var)

print("begin training")
train = train[:4096]
train_labels = train_labels.iloc[:4096]
val = val[:1024]
val_labels = val_labels.iloc[:1024]

for column in train_labels.columns:
    print(train_labels[column].value_counts())
    print()

model.fit(train, train_labels, 3, 64, val, val_labels).print_best("prova.txt")
# model.fit(train, train_labels, 3, 64).print_best()
print("done")

# TODO: add transformations
# TODO: clean code and apis
# TODO: speedup script
# TODO: save train config
# TODO: speedup model
# TODO: experiment
