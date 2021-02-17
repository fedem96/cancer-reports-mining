import dill
import torch


def save(model, filepath):
    torch.save(model, filepath, dill)


def load(filepath):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.load(filepath, map_location=device)
