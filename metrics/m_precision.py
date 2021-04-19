from collections import defaultdict

import numpy as np

from metrics.base.metric import Metric


class MacroPrecision(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.TP = defaultdict(lambda: 0)
        self.FP = defaultdict(lambda: 0)
        self.FN = defaultdict(lambda: 0)

    def update(self, preds, grth):
        correct = preds == grth
        wrong = ~correct
        for cls in np.unique(preds):
            self.TP[cls] += (correct & (preds == cls)).sum().item()
            self.FP[cls] += (wrong & (preds == cls)).sum().item()
        for cls in np.unique(grth):
            self.FN[cls] += (wrong & (grth == cls)).sum().item()

    def __call__(self, *args, **kwargs):
        precisions = [self.TP[key] / (self.TP[key] + self.FP[key]) if (self.TP[key] + self.FP[key]) > 0 else 0 for key in self.FN]
        return sum(precisions) / len(precisions) if len(precisions) > 0 else float('nan')
