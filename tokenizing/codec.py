import json

import numpy as np


class TokenCodec:

    def __init__(self, encoder={}, decoder={}, occurrences=None):
        self.encoder = encoder
        self.decoder = decoder
        self.occurrences = occurrences
        self.tokens_str_ndarray = np.array([""] + list(self.decoder.values()))
        assert len(self.encoder) == len(self.decoder)

    @staticmethod
    def from_json(js):
        encoder = js["encoder"]
        dec = js["decoder"]
        decoder = {int(k): v for k, v in dec.items()}
        occurrences = np.array(js["occurrences"])
        return TokenCodec(encoder, decoder, occurrences)

    @staticmethod
    def load(filename):
        with open(filename, "rt") as file:
            js = json.load(file)
        return TokenCodec.from_json(js)

    def to_json(self):
        return {"encoder": self.encoder, "decoder": self.decoder, "occurrences": self.occurrences.tolist()}

    def save(self, filename):
        with open(filename, "wt") as file:
            json.dump(self.to_json(), file)
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
