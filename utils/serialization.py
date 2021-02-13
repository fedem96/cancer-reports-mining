import dill
import torch


def save(model, filepath):
    torch.save(model, filepath, dill)


def load(filepath):
    return torch.load(filepath)
