import json

import numpy as np
import spacy

from utils.constants import INDEX_FILE


class Tokenizer:

    def __init__(self, tokenizer='it'):
        self._tknzr = spacy.load(tokenizer)

    def tokenize(self, text):
        tokens = self._tknzr(text)
        return tokens

    def tokenize_batch(self, texts):
        tokens = []
        for text in texts:
            tokens.append(self.tokenize(text))
        return tokens

    @staticmethod
    def token_to_string(token):
        return str(token)

    @staticmethod
    def tokens_to_strings(tokens):
        return [str(token) for token in tokens]


class TokenCodec:

    def __init__(self, coder={}, decoder={}, tokenizer='it'):
        self.coder = {}
        self.decoder = {}
        self._tknzr = spacy.load(tokenizer)
        assert len(self.coder) == len(self.decoder)

    def load(self, filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        self.coder = js["coder"]
        self.decoder = js["decoder"]
        assert len(self.coder) == len(self.decoder)
        return self

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump({"coder": self.coder, "decoder": self.decoder}, file)
        return self

    def encode_token(self, token):
        return self.coder.get(token, 0)

    def encode(self, tokens):
        return np.array([self.encode_token(token) for token in tokens])

    def encode_batch(self, tokens_group):
        return [self.encode(tokens) for tokens in tokens_group]

    def decode_token(self, token_idx):
        return self.decoder[token_idx]

    def decode(self, tokens_idxs):
        return np.array([self.decode_token(token_idx) for token_idx in tokens_idxs])

    def decode_batch(self, tokens_idxs_group):
        return [self.decode(tokens_idxs) for tokens_idxs in tokens_idxs_group]

    def num_tokens(self):
        return len(self.coder)


class TokenCodecCreator:
    def __init__(self, preprocessor, tokenizer='it'):
        self.preprocessor = preprocessor
        self.tokenizer = Tokenizer(tokenizer)

    def create_codec(self, texts):
        coder = {}
        decoder = {}
        preprocess = self.preprocessor.preprocess
        tokenize = self.tokenizer.tokenize
        for count in range(len(texts)):
            text = texts[count]
            text = preprocess(text)
            tokens = tokenize(text)
            tokens = set(map(lambda token: str(token), tokens))
            for token in tokens:
                if token not in coder:
                    count += 1
                    coder[token] = count
                    decoder[count] = token
        return TokenCodec(coder, decoder)
