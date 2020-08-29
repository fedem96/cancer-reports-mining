import json

import numpy as np
import spacy

from utils.constants import INDEX_FILE


class TokensHandler:

    def __init__(self, tokenizer='it', idx_file=INDEX_FILE):
        self._tknzr = spacy.load(tokenizer)
        self.idx_file = idx_file
        self.token_idx = self.get_index()

    def get_index(self):
        with open(self.idx_file, "rt") as file:
            idx = json.load(file)
        return idx

    def tokenize(self, text):
        tokens = self._tknzr(text)
        return tokens

    @staticmethod
    def tokens_to_strings(tokens):
        return [str(token) for token in tokens]

    def get_tokens_idx(self, report):
        tokens_str = TokensHandler.tokens_to_strings(self.tokenize(report))
        for token in tokens_str:
            assert type(self.token_idx[token]) == int
        tokens_idx = np.array([self.token_idx[token] for token in tokens_str])
        return tokens_idx

    def num_tokens(self):
        return len(self.token_idx)
