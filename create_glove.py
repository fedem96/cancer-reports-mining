import argparse
from collections import defaultdict
import json

from utils.constants import *
from utils.glove import train, save
from utils.preprocessing import preprocess
from utils.utilities import *


parser = argparse.ArgumentParser(description='Calculate Inverse Document Frequencies and save to file')
parser.add_argument("-d", "--documents", help="documents", default=TRAINING_SET_FILE, type=str)
parser.add_argument("-o", "--output", help="file where to save IDF", default=IDF_FILE, type=str)
args = parser.parse_args()

csv = [args.documents,
    "utf-8",
    {"id_neopl": float, "lateralita": str, "grading": int, "livello_certezza": int, "stadio": int, "terapia": int, "dimensioni": float, "linfoadenec": int, "anno": str, "notizie": str, "diagnosi": str, "macroscopia": str},
    {},
    {"anno": "%Y"},
    {"notizie": "", "diagnosi": "", "macroscopia": ""}
    ]


df = read_csv(*csv)
df = df.head(10)

with open(INDEX_FILE, "r") as file:
    index = json.load(file)

num_docs = len(df)

window_size = 10
cooccurences = defaultdict(lambda: defaultdict(lambda: 0))
for n, d, m in zip(df["notizie"], df["diagnosi"], df["macroscopia"]):
    text = n + " " + d + " " + m

    tokens = tokens_to_strings(tokenize(preprocess(text)))
    for t in range(len(tokens) - window_size):
        idx1 = index[tokens[t]]
        for token2 in tokens[t+1:t+window_size+1]:
            idx2 = index[token2]
            cooccurences[idx1][idx2] += 1
            cooccurences[idx2][idx1] += 1
    
    t = max(0, len(tokens) - window_size)
    idx1 = index[tokens[t]]
    for token2 in tokens[t+1:-1]:
        idx2 = index[token2]
        cooccurences[idx1][idx2] += 1
        cooccurences[idx2][idx1] += 1



glv = train(cooccurences, d=64)
save(glv)
