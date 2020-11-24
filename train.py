import argparse
from datetime import datetime
from importlib import import_module
from multiprocessing import Pool
from timeit import default_timer as timer

import numpy as np

from utils.constants import *
from utils.labels_codec import LabelsCodec
from utils.preprocessing import *
from utils.tokenizing import TokenCodec, Tokenizer
from utils.utilities import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-ap", "--activation-penalty", help="weight for activation (abs) regularization", default=0.00001, type=float)
parser.add_argument("-b", "--batch-size", help="batch size to use for training", default=115, type=int)
parser.add_argument("-c", "--codec", help="token codec filename", default=TOKEN_CODEC, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-e", "--epochs", help="number of maximum training epochs", default=100, type=int)
parser.add_argument("-i", "--idf", help="Inverse Document Frequencies filename", default=IDF, type=str)
parser.add_argument("-lr", "--learning-rate", help="learning rate for Adam optimizer", default=0.00001, type=float)
parser.add_argument("-m", "--model", help="model to train", default=None, type=str)
parser.add_argument("-ma", "--model-args", help="model to train", default=None, type=str)
parser.add_argument("-n", "--name", help="name to use when saving the model", default=None, type=str)
parser.add_argument("-o", "--out", help="file where to save best values of the metrics", default=None, type=str)
parser.add_argument("-t", "--train-config", help="json with train configuration", default=os.path.join("configs", "train_example.json"), type=str)
args = parser.parse_args()

train_df = pd.read_csv(os.path.join(args.dataset_dir, TRAINING_SET))
val_df = pd.read_csv(os.path.join(args.dataset_dir, VALIDATION_SET))
cols = ["diagnosi", "macroscopia", "notizie"]

p = Preprocessor.get_default()
t = Tokenizer()
tc = TokenCodec().load(os.path.join(args.dataset_dir, args.codec))

print("tokenizing reports and encoding tokens")


def full_pipe(report):
    return tc.encode(t.tokenize(p.preprocess(report)))


train, val = [caching(df_to_data, dataframe, full_pipe, cols) for dataframe in [train_df, val_df]]

print("encoding reports")
with open(args.train_config, "rt") as file:
    train_config = json.load(file)


classifications, regressions = train_config.get("classifications", []), train_config.get("regressions", [])
transformations, mappings = train_config.get("transformations", {}), train_config.get("mappings", {})


for df in[train_df, val_df]: prepare_dataset(df, train_config)
columns_codec = get_columns_codec(train_df, train_config)


train_labels, val_labels = [get_encoded_labels(dataset, classifications + regressions, classifications, columns_codec)
                                                            for dataset in [train_df, val_df]]

print("train labels distribution:")
for column in train_labels.columns:
    print(train_labels[column].value_counts())
    print()

print("creating model")
model_name = args.name or str(datetime.now())
model_dir = os.path.join(MODELS_DIR, model_name)
os.makedirs(model_dir)

with open(os.path.join(model_dir, "train_config.json"), "wt") as file:
    json.dump(train_config, file)
with open(os.path.join(model_dir, "args.json"), "wt") as file:
    json.dump(vars(args), file)

# model
# from models.emb_max_fc import EmbMaxLin
# model = EmbMaxLin(tc.num_tokens()+1, 256, 256, 256)
#from models.transformer import Transformer
# model = Transformer(tc.num_tokens()+1, 256, 8, 256, 1, 0.1)
#model = Transformer(tc.num_tokens()+1, 128, 8, 128, 1, 0.1, model_dir)

module, class_name = args.model.rsplit(".", 1)
Model = getattr(import_module(module), class_name)
model_args = {"vocab_size": tc.num_tokens()+1, "directory": model_dir}
if args.model_args is not None:
    with open(args.model_args, "rt") as file:
        model_args.update(json.load(file))
model = Model(**model_args)

for cls_var in classifications:
    model.add_classification(cls_var, train_df[cls_var].nunique())

# for reg_var in regressions:
#     model.add_regression(reg_var)

print("parameters:", sum([p.numel() for p in model.parameters()]))
for parameters in model.parameters():
    print("\t{}: {}".format(parameters.name, parameters.numel()))
print("begin training")
#plot_hists([[len(t) for t in train]])
max_length = 90
train, val = [t[:max_length] for t in train], [v[:max_length] for v in val]  # truncate long sequences
#plot_hists([[len(t) for t in train]])

hyperparameters = {"learning_rate": args.learning_rate, "activation_penalty": args.activation_penalty}
hyperparameters.update({"max_epochs": args.epochs, "batch_size": args.batch_size})
with open(os.path.join(model_dir, "hyperparameters.json"), "wt") as file:
    json.dump(hyperparameters, file)
model.fit(train, train_labels, val, val_labels, **hyperparameters).print_best(args.out)
print("done")

# TODO: add transformations
# TODO: clean code and apis
# TODO: speedup script
# TODO: speedup model
# TODO: experiment
