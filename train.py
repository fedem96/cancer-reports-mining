import argparse
import json
import os
from importlib import import_module
from timeit import default_timer as timer

from utils.chrono import Chronostep
from utils.constants import *
from utils.dataset import Dataset
from utils.preprocessing import *
from utils.tokenizing import TokenCodec, Tokenizer
from utils.utilities import *

parser = argparse.ArgumentParser(description='Train a model')
parser.add_argument("-ap", "--activation-penalty", help="weight for activation (abs) regularization", default=0.00001, type=float)
parser.add_argument("-b", "--batch-size", help="batch size to use for training", default=115, type=int)
parser.add_argument("-c", "--codec", help="token codec filename", default=TOKEN_CODEC, type=str)
parser.add_argument("-d", "--dataset-dir", help="directory containing the dataset", default=os.path.join(DATASETS_DIR, NEW_DATASET), type=str)
parser.add_argument("-e", "--epochs", help="number of maximum training epochs", default=100, type=int)
parser.add_argument("-gb", "--group_by",
                    help="list of (space-separated) grouping attributes to make multi-report predictions. "
                         "If set, parameter 'reduce' must also be set",
                    default=None, nargs="+", type=str, metavar=('ATTR1', 'ATTR2'))
parser.add_argument("-i", "--idf", help="Inverse Document Frequencies filename", default=IDF, type=str)
parser.add_argument("-lr", "--learning-rate", help="learning rate for Adam optimizer", default=0.00001, type=float)
parser.add_argument("-m", "--model", help="model to train", default=None, type=str)
parser.add_argument("-ma", "--model-args", help="model to train", default=None, type=str)
parser.add_argument("-ml", "--max-length", help="maximum sequence length (cut long sequences)", default=None, type=int)
parser.add_argument("-n", "--name", help="name to use when saving the model", default=None, type=str)
parser.add_argument("-o", "--out", help="file where to save best values of the metrics", default=None, type=str)
parser.add_argument("-r", "--reduce",
                    help="grouping strategy to use in multi-report predictions.",
                    default=None, type=str, metavar='STRATEGY', required=True)
parser.add_argument("-t", "--train-config", help="json with train configuration", default=os.path.join("train_configs", "train_example.json"), type=str)
args = parser.parse_args()

print("args:", vars(args))

p = Preprocessor.get_default()
t = Tokenizer()
tc = TokenCodec().load(os.path.join(args.dataset_dir, args.codec))

with Chronostep("reading input"):
    with open(args.train_config, "rt") as file:
        train_config = json.load(file)

    classifications, regressions = train_config.get("classifications", []), train_config.get("regressions", [])
    transformations, mappings = train_config.get("transformations", {}), train_config.get("mappings", {})

    with open(args.reduce, "rt") as file:
        reduce_config = json.load(file)
    assert "reduce_type" in reduce_config and "reduce_mode" in reduce_config
    reduce_type = reduce_config["reduce_type"]
    reduce_mode = reduce_config["reduce_mode"]
    assert reduce_type in {"data", "features", "predictions"}

with Chronostep("encoding reports"):
    input_cols = ["diagnosi", "macroscopia", "notizie"]

    sets = {"train": Dataset(os.path.join(args.dataset_dir, TRAINING_SET)), "val": Dataset(os.path.join(args.dataset_dir, VALIDATION_SET))}


    def full_pipe(report):
        return tc.encode(t.tokenize(p.preprocess(report)))


    for set_name in ["train", "val"]:
        dataset = sets[set_name]
        dataset.set_input_cols(input_cols)
        dataset.process_records(full_pipe)
        dataset.prepare_for_training(**train_config)
        if set_name == "train":
            columns_codec = dataset.get_columns_codec()
        else:
            dataset.set_columns_codec(columns_codec)
        dataset.encode_labels()
        if args.max_length is not None:
            dataset.cut_sequences(args.max_length)

        if args.group_by is not None:
            dataset.group_by(args.group_by)

            if reduce_type == "data":
                dataset.reduce(reduce_mode)

    training, validation = sets["train"], sets["val"]
    if args.group_by is not None:
        training.assert_disjuncted(validation)

print("train labels distribution:")
for column in training.labels.columns:
    print(training.labels[column].value_counts())
    print()


with Chronostep("creating model"):
    model_name = args.name or str(datetime.now()).replace(":", ";")
    model_dir = os.path.join(TRAINED_MODELS_DIR, model_name)
    os.makedirs(model_dir)

    with open(os.path.join(model_dir, "train_config.json"), "wt") as file:
        json.dump(train_config, file)
    with open(os.path.join(model_dir, "args.json"), "wt") as file:
        json.dump(vars(args), file)

    module, class_name = args.model.rsplit(".", 1)
    Model = getattr(import_module(module), class_name)
    model_args = {"vocab_size": tc.num_tokens()+1, "directory": model_dir}
    if args.model_args is not None:
        with open(args.model_args, "rt") as file:
            model_args.update(json.load(file))
    model = Model(**model_args)
    model.set_reduce_method(reduce_type, reduce_mode)

    for cls_var in classifications:
        model.add_classification(cls_var, training.nunique(cls_var))

    # for reg_var in regressions:
    #     model.add_regression(reg_var)

print("parameters:", sum([p.numel() for p in model.parameters()]))
for parameters in model.parameters():
    print("\t{}: {}".format(parameters.name, parameters.numel()))

hyperparameters = {"learning_rate": args.learning_rate, "activation_penalty": args.activation_penalty}
hyperparameters.update({"max_epochs": args.epochs, "batch_size": args.batch_size})
with open(os.path.join(model_dir, "hyperparameters.json"), "wt") as file:
    json.dump(hyperparameters, file)

with Chronostep("training"):
    model.fit(training.data, training.labels, validation.data, validation.labels, **hyperparameters).print_best(args.out)

# TODO: add transformations
# TODO: clean code and apis
# TODO: speedup script
# TODO: speedup model
# TODO: experiment
