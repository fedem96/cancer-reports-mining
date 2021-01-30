import argparse
import json
from importlib import import_module

from utils.chrono import Chronostep
from utils.constants import *
from utils.dataset import Dataset
from utils.preprocessing import *
from utils.tokenizing import TokenCodec, Tokenizer
from utils.utilities import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-ap", "--activation-penalty", help="weight for activation (abs) regularization", default=0, type=float)
parser.add_argument("-b", "--batch-size", help="batch size to use for training", default=128, type=int)
parser.add_argument("-c", "--codec", help="token codec filename", default=TOKEN_CODEC, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-ds", "--data-seed", help="seed for random data shuffling", default=None, type=int)
parser.add_argument("-e", "--epochs", help="number of maximum training epochs", default=50, type=int)
parser.add_argument("-f", "--filter",
                    help="report filtering strategy",
                    default=None, type=str, choices=['same_year'], metavar='STRATEGY')
parser.add_argument("-gb", "--group-by",
                    help="list of (space-separated) grouping attributes to make multi-report predictions.",
                    default=None, nargs="+", type=str, metavar=('ATTR1', 'ATTR2'))
parser.add_argument("-i", "--idf", help="Inverse Document Frequencies filename", default=IDF, type=str)
parser.add_argument("-lr", "--learning-rate", help="learning rate for Adam optimizer", default=0.00001, type=float)
parser.add_argument("-m", "--model", help="model to train", default=None, type=str, required=True)
parser.add_argument("-ma", "--model-args", help="model to train", default=None, type=json.loads)
parser.add_argument("-ns", "--net-seed", help="seed for model random weights generation", default=None, type=int)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
parser.add_argument("-n", "--name", help="name to use when saving the model", default=None, type=str)
parser.add_argument("-o", "--out", help="file where to save best values of the metrics", default=None, type=str)
parser.add_argument("-rm", "--reduce-mode", help="how to reduce", default=None, type=str, required=True)
parser.add_argument("-rt", "--reduce-type", help="what to reduce", default=None, type=str,
                    choices=["data", "features", "logits", "eval"], required=True)
parser.add_argument("-tc", "--train-classifications", help="list of classifications", default=[], nargs="+", type=str)
parser.add_argument("-tr", "--train-regressions", help="list of regressions", default=[], nargs="+", type=str)
args = parser.parse_args()
print("args:", vars(args))
assert args.reduce_mode in {"data": {"most_recent"}, "features": {"max"}, "logits": {"mean"}, "eval": {"argmax"}}[args.reduce_type]  # TODO: multiple reduce modes

p = Preprocessor.get_default()
t = Tokenizer()
tc = TokenCodec().load(os.path.join(args.dataset_dir, args.codec))

with Chronostep("reading input"):

    classifications, regressions = args.train_classifications, args.train_regressions
    transformations, mappings = {}, {}

with Chronostep("encoding reports"):
    input_cols = ["diagnosi", "macroscopia", "notizie"]

    sets = {"train": Dataset(os.path.join(args.dataset_dir, TRAINING_SET)), "val": Dataset(os.path.join(args.dataset_dir, VALIDATION_SET))}


    def full_pipe(report):
        return tc.encode(t.tokenize(p.preprocess(report)))


    for set_name in ["train", "val"]:
        dataset = sets[set_name]
        dataset.set_input_cols(input_cols)
        dataset.process_records(full_pipe)
        dataset.prepare_for_training(classifications, regressions, transformations, mappings)
        if set_name == "train":
            columns_codec = dataset.get_columns_codec()
        else:
            dataset.set_columns_codec(columns_codec)
        dataset.encode_labels()
        if args.max_length is not None:
            dataset.cut_sequences(args.max_length)

        if args.group_by is not None and (args.reduce_type != "eval" or set_name != "train"):
            dataset.group_by(args.group_by)

            if args.filter is not None:
                dataset.filter(args.filter)
            if args.reduce_type == "data":
                dataset.reduce(args.reduce_mode)

    training, validation = sets["train"], sets["val"]
    if args.group_by is not None and args.reduce_type != "eval":
        training.assert_disjuncted(validation)

print("train labels distribution:")
for column in training.labels.columns:
    print(training.labels[column].value_counts())
    print()

with Chronostep("creating model"):
    model_name = args.name or str(datetime.now()).replace(":", ";")
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_name)
    os.makedirs(model_dir)

    with open(os.path.join(model_dir, "args.json"), "wt") as file:
        json.dump(vars(args), file)

    module, class_name = args.model.rsplit(".", 1)
    Model = getattr(import_module(module), class_name)
    model_args = {"vocab_size": tc.num_tokens()+1, "directory": model_dir}
    if args.model_args is not None:
        model_args.update(args.model_args)
    if args.net_seed is not None:
        model_args.update({"net_seed": args.net_seed})
    model = Model(**model_args)
    model.set_reduce_method(args.reduce_type, args.reduce_mode)

    for cls_var in classifications:
        model.add_classification(cls_var, training.nunique(cls_var))

    # for reg_var in regressions:
    #     model.add_regression(reg_var)

print("parameters:", sum([p.numel() for p in model.parameters()]))
for parameter_name, parameter in model.named_parameters():
    print("\t{}: {}".format(parameter_name, parameter.numel()))

hyperparameters = {"learning_rate": args.learning_rate, "activation_penalty": args.activation_penalty,
                   "max_epochs": args.epochs, "batch_size": args.batch_size, "data_seed": args.data_seed}
with open(os.path.join(model_dir, "hyperparameters.json"), "wt") as file:
    json.dump(hyperparameters, file)

with Chronostep("training"):
    model.fit(training.data, training.labels, validation.data, validation.labels, **hyperparameters).print_best(args.out)

# TODO: add transformations
# TODO: clean code and apis
# TODO: speedup script
# TODO: speedup model
# TODO: experiment
