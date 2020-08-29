import argparse

from utils.constants import *
from utils.idf import train, save
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

num_docs = len(df)

texts = []
for n, d, m in zip(df["notizie"], df["diagnosi"], df["macroscopia"]):
    text = n + " " + d + " " + m
    texts.append(text)

idf = train(texts)
save(idf)