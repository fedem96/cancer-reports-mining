from collections import defaultdict
import json
import math

from utils.constants import *
from utils.preprocessing import preprocess
from utils.utilities import tokenize, tokens_to_strings

def train(documents):

    counts = defaultdict(lambda: 0)
    num_docs = len(documents)
    
    for doc in documents:
        text = preprocess(doc)
        tokens = tokenize(text)
        tokens = set(tokens_to_strings(tokens))
        for token in tokens:
            counts[token] += 1
            
    idf = {token: math.log(num_docs / counts[token]) for token in counts}

    return idf

def save(idf, filename=IDF_FILE):
    #import numpy as np
    with open(filename, "w") as file:
        json.dump(idf, file)

def load(filename=IDF_FILE):
    with open(filename, "r") as file:
        idf = json.load(file)
    return idf
