import numpy as np

from metrics.base.metric import Metric


class PredictionsAccumulator(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.predictions = []

    def update(self, preds, grth):
        self.predictions.append(preds)

    def __call__(self, *args, **kwargs):
        return np.concatenate(self.predictions)

    def __str__(self):
        return str(self.__call__().tolist())

    def __repr__(self):
        return str(self.__call__().tolist())
