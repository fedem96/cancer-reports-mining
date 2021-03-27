from collections import defaultdict
import json
import math

from utils.preprocessing import Preprocessor
from utils.tokenizing import Tokenizer


class InverseFrequenciesCounter:

    def __init__(self):
        self.idf = {}

    def load(self, filename):
        with open(filename, "r") as file:
            self.idf = json.load(file)
        return self

    def save(self, filename):
        with open(filename, "w") as file:
            json.dump(self.idf, file)
        return self

    def train(self, documents):

        counts = defaultdict(lambda: 0)
        num_docs = len(documents)

        texts = Preprocessor.get_default().preprocess_batch(documents)
        all_tokens = Tokenizer().tokenize_batch(texts)

        for tokens in all_tokens:
            for token in set(tokens):
                counts[token] += 1

        self.idf = {token: math.log(num_docs / counts[token]) for token in counts}

        return self

    def get_idf(self, token):
        return self.idf.get(token, 0)
