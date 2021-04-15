import json
from collections import defaultdict

import numpy as np
import spacy

from utils.preprocessing import Preprocessor


class Tokenizer:

    def __init__(self, tokenizer='it'):
        self._tknzr = spacy.load(tokenizer, disable=["tagger", "parser", "ner"])
        self._tknzr.add_pipe(lambda doc: [str(token) for token in doc])

    def tokenize(self, text):
        tokens = self._tknzr(text)
        return tokens

    def tokenize_batch(self, texts):
        return list(self._tknzr.pipe(texts))


class TokenCodec:

    def __init__(self, encoder={}, decoder={}, occurrences={}):
        self.encoder = encoder
        self.decoder = decoder
        self.occurrences = occurrences
        self.tokens_str_ndarray = np.array([""] + list(self.decoder.values()))
        assert len(self.encoder) == len(self.decoder)

    def load(self, filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        self.encoder = js["encoder"]
        decoder = js["decoder"]
        self.decoder = {int(k): v for k, v in decoder.items()}
        self.occurrences = js["occurrences"]
        self.tokens_str_ndarray = np.array([""] + list(decoder.values()))
        assert len(self.encoder) == len(self.decoder)
        return self

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump({"encoder": self.encoder, "decoder": self.decoder, "occurrences": self.occurrences}, file)
        return self

    def encode_token(self, token):
        return self.encoder.get(token, 0)

    def encode(self, tokens):
        return np.array([self.encode_token(token) for token in tokens])

    def encode_batch(self, tokens_group):
        return [self.encode(tokens) for tokens in tokens_group]

    def decode_token(self, token_idx):
        return self.decoder.get(token_idx, "")

    def decode(self, tokens_idxs):
        return np.array([self.decode_token(token_idx) for token_idx in tokens_idxs])

    def decode_batch(self, tokens_idxs_group):
        return [self.decode(tokens_idxs) for tokens_idxs in tokens_idxs_group]

    def decode_ndarray(self, tokens):
        return self.tokens_str_ndarray[tokens]

    def num_tokens(self):
        return len(self.encoder)


class TokenCodecCreator:

    def __init__(self, preprocessor=Preprocessor.get_default(), tokenizer='it'):
        self.preprocessor = preprocessor
        self.tokenizer = Tokenizer(tokenizer)

    def create_codec(self, texts, min_occurrences=0):
        encoder = {}
        decoder = {}
        count = 0
        tokens_batch = self.tokenizer.tokenize_batch(self.preprocessor.preprocess_batch(texts))
        occurrences = defaultdict(lambda: 0)
        for tokens in tokens_batch:
            for token in tokens:
                occurrences[token] += 1
                if occurrences[token] >= min_occurrences and token not in encoder:
                    count += 1
                    encoder[token] = count
                    decoder[count] = token
        return TokenCodec(encoder, decoder, occurrences)
