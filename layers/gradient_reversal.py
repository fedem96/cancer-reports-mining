from torch import nn

from functional.gradient_reversal import gradient_reversal


class GradientReversal(nn.Module):

    def __init__(self, l):
        super().__init__()
        self.l = l

    def forward(self, x):
        return gradient_reversal(x, self.l)
