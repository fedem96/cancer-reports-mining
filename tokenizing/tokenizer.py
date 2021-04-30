import json
import math
from collections import defaultdict

import numpy as np
import spacy

from tokenizing.codec import TokenCodec


class Tokenizer:

    def __init__(self, tokenizer='it', n_grams=1, num_docs=None, codec=None):
        self._tknzr = spacy.load(tokenizer, disable=["tagger", "parser", "ner"])
        def extract_tokens(doc, n_grams_max=n_grams):
            tokens = []
            for n in range(1, n_grams_max+1):
                tokens += [" ".join([str(doc[i + j]) for j in range(min(n, len(doc)))]) for i in range(len(doc) + 1 - n)] # str(doc[i+j]) can be replaced with doc[i+j].lemma_ to get the lemma
            return tokens
        self._tknzr.add_pipe(extract_tokens)
        self.spacy_tokenizer = tokenizer
        self.n_grams = n_grams
        self.codec = codec
        self.num_docs = num_docs

    @staticmethod
    def from_json(js):
        return Tokenizer(js["tokenizer"], js["n_grams"], js["num_docs"], TokenCodec.from_json(js["codec"]))

    @staticmethod
    def load(filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        return Tokenizer.from_json(js)

    def to_json(self):
        return {"tokenizer": self.spacy_tokenizer, "n_grams": self.n_grams, "num_docs": self.num_docs, "codec": self.codec.to_json()}

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump(self.to_json(), file)
        return self

    def create_codec(self, texts, min_occurrences=0, max_occurrences=None):
        encoder = {}
        decoder = {}
        found_tokens = set()
        count = 0
        tokens_batch = self.tokenize_batch(texts)
        occurrences_dict = defaultdict(lambda: 0)
        for tokens in tokens_batch:
            for token in set(tokens):
                occurrences_dict[token] += 1
                if occurrences_dict[token] >= min_occurrences and token not in found_tokens:
                    found_tokens.add(token)

        for token in found_tokens:
            if max_occurrences is None or occurrences_dict[token] <= max_occurrences:
                count += 1
                encoder[token] = count
                decoder[count] = token

        occurrences_ndarray = np.array([0] + [occurrences_dict[decoder[i]] for i in range(1, count+1)])
        self.codec = TokenCodec(encoder, decoder, occurrences_ndarray)
        self.num_docs = len(texts)
        return self

    def tokenize(self, text, encode=False):
        tokens = self._tknzr(text)
        if encode:
            tokens = self.encode(tokens)
        return tokens

    def tokenize_batch(self, texts, encode=False):
        tokens_list = list(self._tknzr.pipe(texts))
        if encode:
            tokens_list = self.encode_batch(tokens_list)
        return tokens_list

    def encode_token(self, token):
        return self.codec.encode_token(token)

    def encode(self, tokens):
        return self.codec.encode(tokens)

    def encode_batch(self, tokens):
        return self.codec.encode_batch(tokens)

    def decode_token(self, token_idx):
        return self.codec.decode_token(token_idx)

    def decode(self, tokens_idxs):
        return self.codec.decode(tokens_idxs)

    def decode_batch(self, tokens_idxs_group):
        return self.codec.decode_batch(tokens_idxs_group)

    def decode_ndarray(self, tokens):
        return self.codec.decode_ndarray(tokens)

    def num_tokens(self):
        return self.codec.num_tokens()

    def get_idf(self, token_idx, encode=False):
        if encode:
            token_idx = self.encode_token(token_idx)
        return math.log(self.num_docs / self.codec.occurrences[token_idx])

    def get_idf_ndarray(self, tokens_idxs):
        return np.log(self.num_docs / self.codec.occurrences[tokens_idxs])
