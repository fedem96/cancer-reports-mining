import numpy as np

from metrics.base.metric import Metric


class Recall(Metric):
    def __init__(self, cls):
        super().__init__(max)
        self.reset()
        self.cls = cls

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, preds, grth):
        correct = preds == grth
        wrong = ~correct
        for cls in np.unique(preds):
            if cls != self.cls:
                continue
            self.TP += (correct & (preds == cls)).sum().item()
            self.FP += (wrong & (preds == cls)).sum().item()
        for cls in np.unique(grth):
            if cls != self.cls:
                continue
            self.FN += (wrong & (grth == cls)).sum().item()

    def __call__(self, *args, **kwargs):
        d = self.TP + self.FN
        return self.TP / d if d > 0 else 0
