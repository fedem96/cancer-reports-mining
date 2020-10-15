import argparse
import json

from utils.constants import *
from utils.tokenizing import TokenCodecCreator
from utils.utilities import *


parser = argparse.ArgumentParser(description='Calculate token coder (str to int) and decoder (int to str)')
parser.add_argument("-d", "--dataset", help="directory of the dataset", required=True, type=str)
parser.add_argument("-t", "--training", help="training set filename", default=TRAINING_SET, type=str)
parser.add_argument("-o", "--output", help="filename where to save the codec", default=TOKEN_CODEC, type=str)
args = parser.parse_args()

csv = [os.path.join(args.dataset, args.training),
    "utf-8",
    {"id_neopl": "Int64", "notizie": str, "diagnosi": str, "macroscopia": str},
    {},
    {},
    {"notizie": "", "diagnosi": "", "macroscopia": ""}
    ]


with Chronometer("cod"):
    df = read_csv(*csv)

    tcc = TokenCodecCreator()
    texts = merge_and_extract(df, ["notizie", "diagnosi", "macroscopia"])
    token_codec = tcc.create_codec(texts)
    token_codec.save(os.path.join(args.dataset, args.output))

print("codec created")
