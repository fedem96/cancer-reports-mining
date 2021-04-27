import torch
from torch import nn

from models.nn.modular_base import ModularBase


class MultiLayerPerceptron(ModularBase):
    def __init__(self, vocab_size, hidden_sizes, net_seed=None, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device:", device)

        if net_seed is not None:
            torch.manual_seed(net_seed)

        if tuple != type(hidden_sizes) != list:
            hidden_sizes = [hidden_sizes]
        layers = []
        input_size = [vocab_size] + hidden_sizes
        for n in range(len(hidden_sizes)):
            layers.append(nn.Linear(input_size[n], hidden_sizes[n]).to(device))
            layers.append(nn.ReLU())

        modules = {
            "linear_layers": nn.Sequential(*layers)
        }

        super(MultiLayerPerceptron, self).__init__(modules, hidden_sizes[-1], "mlp", *args, **kwargs)

    def extract_features(self, x):
        return self.linear_layers(x).unsqueeze(1)
