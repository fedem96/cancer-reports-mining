import argparse

from utils.constants import *
from utils.idf import InverseFrequenciesCounter
from utils.utilities import *


parser = argparse.ArgumentParser(description='Calculate Inverse Document Frequencies and save to file')
parser.add_argument("-d", "--dataset", help="directory of the dataset", required=True, type=str)
parser.add_argument("-t", "--training", help="training set filename", default=TRAINING_SET, type=str)
parser.add_argument("-o", "--output", help="filename where to save IDF", default=IDF, type=str)
args = parser.parse_args()

csv = [os.path.join(args.dataset, args.training),
    "utf-8",
    {"id_neopl": "Int64", "notizie": str, "diagnosi": str, "macroscopia": str},
    {},
    {},
    {"notizie": "", "diagnosi": "", "macroscopia": ""}
    ]


df = read_csv(*csv)

texts = merge_and_extract(df, ["notizie", "diagnosi", "macroscopia"])

InverseFrequenciesCounter().train(texts).save(os.path.join(args.dataset, args.output))
