import numpy as np

from metrics.base.metric import Metric


class NormalizedMeanAverageError(Metric):
    def __init__(self, normalizing_divisor):
        super().__init__(min)
        self.reset()
        self.normalizing_divisor = normalizing_divisor

    def reset(self):
        self.error = 0
        self.num_examples = 0

    def update(self, preds, grth):
        self.error += np.abs(preds - grth).sum().item()
        self.num_examples += len(grth)

    def __call__(self, *args, **kwargs):
        return self.error / self.num_examples / self.normalizing_divisor
