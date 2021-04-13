import argparse
import json
from importlib import import_module

from callbacks.early_stopping import EarlyStoppingSW
from callbacks.logger import MetricsLogger
from callbacks.model_checkpoint import ModelCheckpoint
from utils.chrono import Chronostep
from utils.constants import *
from utils.dataset import Dataset
from utils.idf import InverseFrequenciesCounter
from utils.preprocessing import *
from utils.tokenizing import TokenCodec, Tokenizer
from utils.utilities import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-ap", "--activation-penalty", help="weight for activation (abs) regularization", default=0, type=float)
parser.add_argument("-b", "--batch-size", help="batch size to use for training", default=128, type=int)
parser.add_argument("-c", "--codec", help="token codec filename", default=TOKEN_CODEC, type=str)
parser.add_argument("-cr", "--concatenate-reports", help="whether to concatenate reports of the same record before training", default=False, action='store_true')
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-ds", "--data-seed", help="seed for random data shuffling", default=None, type=int)
parser.add_argument("-e", "--epochs", help="number of maximum training epochs", default=50, type=int)
parser.add_argument("-f", "--filter",
                    help="report filtering strategy",
                    default=None, type=str, choices=['same_year', 'classifier'], metavar='STRATEGY')
parser.add_argument("-fa", "--filter-args",
                    help="args for report filtering strategy",
                    default=None, type=json.loads)
parser.add_argument("-gb", "--group-by",
                    help="list of (space-separated) grouping attributes to make multi-report predictions.",
                    default=None, nargs="+", type=str, metavar=('ATTR1', 'ATTR2'))
parser.add_argument("-i", "--idf", help="Inverse Document Frequencies filename", default=IDF, type=str)
parser.add_argument("-ic", "--input-cols", help="Inverse Document Frequencies filename", default=["diagnosi", "macroscopia", "notizie"], nargs="+", type=str)
parser.add_argument("-lr", "--learning-rate", help="learning rate for Adam optimizer", default=0.00001, type=float)
parser.add_argument("-lp", "--labels-preprocessing", help="how to preprocess the labels", default={}, type=json.loads)
parser.add_argument("-m", "--model", help="model to train", default=None, type=str, required=True)
parser.add_argument("-ma", "--model-args", help="model to train", default=None, type=json.loads)
parser.add_argument("-ns", "--net-seed", help="seed for model random weights generation", default=None, type=int)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
parser.add_argument("-mtl", "--max-total-length", help="maximum sequence length after concatenation (cut long sequences)", default=None, type=int)
parser.add_argument("-n", "--name", help="name to use when saving the model", default=None, type=str)
parser.add_argument("-o", "--out", help="file where to save best values of the metrics", default=None, type=str) # TODO: add print best
parser.add_argument("-pr", "--pool-reports", help="whether (and how) to pool reports (i.e. aggregate features of reports in the same record)", default=None, type=str, choices=["max"])
parser.add_argument("-pt", "--pool-tokens", help="how to pool tokens (i.e. aggregate features of tokens in the same report)", default=None, choices=["max"], required=True)
parser.add_argument("-rt", "--reports-transformation", help="how to transform reports (i.e. how to obtain a deep representation of the reports)", default="identity", choices=["identity", "transformer"])
parser.add_argument("-rta", "--reports-transformation-args", help="args for report transformation", default={}, type=json.loads)
parser.add_argument("-tc", "--train-classifications", help="list of classifications", default=[], nargs="+", type=str)
parser.add_argument("-tr", "--train-regressions", help="list of regressions", default=[], nargs="+", type=str)
args = parser.parse_args()
print("args:", vars(args))

p = Preprocessor.get_default()
t = Tokenizer()
tc = TokenCodec().load(os.path.join(args.dataset_dir, args.codec))
idf = InverseFrequenciesCounter().load(os.path.join(args.dataset_dir, args.idf))

DATA_COL = "encoded_data"

with Chronostep("reading input"):

    classifications, regressions = args.train_classifications, args.train_regressions
    transformations = args.labels_preprocessing

with Chronostep("encoding reports"):
    sets = {"train": Dataset(os.path.join(args.dataset_dir, TRAINING_SET)), "val": Dataset(os.path.join(args.dataset_dir, VALIDATION_SET))}


    def full_pipe(report):
        return tc.encode(t.tokenize(p.preprocess(report)))


    for set_name in ["train", "val"]:
        dataset = sets[set_name]
        dataset.set_input_cols(args.input_cols)
        dataset.add_encoded_column(full_pipe, DATA_COL, args.max_length)
        dataset.set_encoded_data_column(DATA_COL)
        dataset.prepare_for_training(classifications, regressions, transformations)
        if set_name == "train":
            columns_codec = dataset.get_columns_codec()
        else:
            dataset.set_columns_codec(columns_codec)
        dataset.encode_labels()

        if args.group_by is not None:

            dataset.lazy_group_by(args.group_by)

            if args.filter is not None:
                dataset.lazy_filter(args.filter, args.filter_args)

            if args.concatenate_reports:
                dataset.lazy_concatenate_reports(args.max_total_length)

            dataset.compute_lazy()

    training, validation = sets["train"], sets["val"]
    if args.group_by is not None:
        training.assert_disjuncted(validation)

    # training.limit(16840)
    # validation.limit(2048)

with Chronostep("calculating labels distributions"):
    training_labels = training.get_labels()
    validation_labels = validation.get_labels()
    for column in training.classifications:
        print("training")
        print(training_labels[column].value_counts()) # TODO: show also percentages
        print("validation")
        print(validation_labels[column].value_counts())
        print()
    for column in training.regressions:
        print(column)
        print("training")
        print("min:", training_labels[column].min())
        print("max:", training_labels[column].max())
        print("mean:", training_labels[column].mean())
        print("std:", training_labels[column].std())
        print("validation")
        print("min:", validation_labels[column].min())
        print("max:", validation_labels[column].max())
        print("mean:", validation_labels[column].mean())
        print("std:", validation_labels[column].std())
        print()

with Chronostep("creating model"):
    model_name = args.name or random_name(str(args.model).split(".")[-1])
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_name)
    os.makedirs(model_dir)

    with open(os.path.join(model_dir, "args.json"), "wt") as file:
        json.dump(vars(args), file)

    module, class_name = args.model.rsplit(".", 1)
    Model = getattr(import_module(module), class_name)
    model_args = {"vocab_size": tc.num_tokens()+1, "preprocessor": p, "tokenizer": t, "token_codec": tc, "labels_codec": training.get_columns_codec(), "idf": idf}
    if args.model_args is not None:
        model_args.update(args.model_args)
    if args.net_seed is not None:
        model_args.update({"net_seed": args.net_seed})
    model = Model(**model_args)
    model = to_gpu_if_available(model)
    model.set_tokens_pooling_method(args.pool_tokens)
    model.set_reports_transformation_method(args.reports_transformation)
    model.set_reports_pooling_method(args.pool_reports)

    for cls_var in classifications:
        model.add_classification(cls_var, training.nunique(cls_var))

    for reg_var in regressions:
        model.add_regression(reg_var)

print("created model: " + model_name)
print("model device:", model.current_device())

print("parameters:", sum([p.numel() for p in model.parameters()]))
for parameter_name, parameter in model.named_parameters():
    print("\t{}: {}".format(parameter_name, parameter.numel()))

hyperparameters = {"learning_rate": args.learning_rate, "activation_penalty": args.activation_penalty,
                   "max_epochs": args.epochs, "batch_size": args.batch_size}
with open(os.path.join(model_dir, "hyperparameters.json"), "wt") as file:
    json.dump(hyperparameters, file)

info = {**{k: v for k, v in vars(args).items() if k in {"data_seed", "net_seed", "filter", "filter_args", "concatenate_reports"
                                                        "model_args", "pool_reports", "pool_tokens"}},
        "name": model_name, "dataset": args.dataset_dir.split(".")[-1]}

if args.data_seed is not None:
    np.random.seed(args.data_seed)

# from utils.utilities import hist
# hist([len(ex) for ex in training.get_data(DATA_COL)])

tb_dir = os.path.join(model_dir, "logs")
callbacks = [MetricsLogger(terminal='table', tensorboard_dir=tb_dir, aim_name=model.__name__, history_size=10),
             ModelCheckpoint(model_dir, 'Loss', verbose=True, save_best=True),
             #EarlyStoppingSW('Loss', min_delta=1e-5, patience=10, verbose=True, from_epoch=10)
            ]

with Chronostep("training"):
    model.fit(training.get_data(DATA_COL), training_labels, validation.get_data(DATA_COL), validation_labels, info, callbacks, **hyperparameters)

# TODO: clean code and apis
# TODO: speedup script
# TODO: experiment
