import argparse
import json

from utils.constants import *
from utils.preprocessing import preprocess
from utils.utilities import *


parser = argparse.ArgumentParser(description='Calculate Inverse Document Frequencies and save to file')
parser.add_argument("-d", "--documents", help="documents", default=TRAINING_SET_FILE, type=str)
parser.add_argument("-o", "--output", help="file where to save the index", default=INDEX_FILE, type=str)
args = parser.parse_args()

csv = [args.documents,
    "utf-8",
    {"id_neopl": float, "lateralita": str, "grading": int, "livello_certezza": int, "stadio": int, "terapia": int, "dimensioni": float, "linfoadenec": int, "anno": str, "notizie": str, "diagnosi": str, "macroscopia": str},
    {},
    {"anno": "%Y"},
    {"notizie": "", "diagnosi": "", "macroscopia": ""}
    ]


df = read_csv(*csv)

num_docs = len(df)

count = 0
index = {}
for n, d, m in zip(df["notizie"], df["diagnosi"], df["macroscopia"]):
    text = n + " " + d + " " + m
    text = preprocess(text)
    tokens = tokenize(text)
    tokens = set(map(lambda token: str(token), tokens))
    for token in tokens:
        if token not in index:
            count += 1
            index[token] = count


with open(INDEX_FILE, "w") as file:
    json.dump(index, file)
