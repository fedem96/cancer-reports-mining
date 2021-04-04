from collections import defaultdict

import numpy as np

from metrics.base.metric import Metric


class Precision(Metric):
    def __init__(self, cls):
        super().__init__(max)
        self.reset()
        self.cls = cls

    def reset(self):
        self.TP = defaultdict(lambda: 0)
        self.FP = defaultdict(lambda: 0)
        self.FN = defaultdict(lambda: 0)

    def update(self, preds, grth):
        correct = preds == grth
        wrong = ~correct
        for cls in np.unique(preds):
            if cls != self.cls:
                continue
            self.TP[cls] += (correct & (preds == cls)).sum().item()
            self.FP[cls] += (wrong & (preds == cls)).sum().item()
        for cls in np.unique(grth):
            if cls != self.cls:
                continue
            self.FN[cls] += (wrong & (grth == cls)).sum().item()

    def __call__(self, *args, **kwargs):
        d = (self.TP[self.cls] + self.FP[self.cls])
        return self.TP[self.cls] / d if d > 0 else 0
