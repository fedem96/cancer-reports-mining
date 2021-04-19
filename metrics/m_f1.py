from collections import defaultdict

import numpy as np

from metrics.base.metric import Metric


class MacroF1Score(Metric):
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
        # f1 = 2 * (p * r) / (p + r)
        precisions = [self.TP[key] / (self.TP[key] + self.FP[key]) if (self.TP[key] + self.FP[key]) > 0 else 0 for key in self.FN]
        recalls = [self.TP[key] / (self.TP[key] + self.FN[key]) for key in self.FN]
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
        return sum(f1_scores) / len(f1_scores) if len(f1_scores) > 0 else float('nan')
