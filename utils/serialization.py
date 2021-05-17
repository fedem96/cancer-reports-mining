import pickle

import dill
import torch


def save(model, filepath):
    if filepath.endswith(".pth"):
        torch.save(model, filepath, dill)
    elif filepath.endswith(".pkl"):
        with open(filepath, "wb") as file:
            pickle.dump(model, file)
    else:
        raise ValueError(f"file extension not recognized: '{filepath}'")


def load(filepath):
    if filepath.endswith(".pth"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.load(filepath, map_location=device)
    elif filepath.endswith(".pkl"):
        with open(filepath, "rb") as file:
            return pickle.load(file)
    else:
        raise ValueError(f"file extension not recognized: '{filepath}'")
