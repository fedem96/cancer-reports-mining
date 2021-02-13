from collections import defaultdict

import numpy as np

from metrics.base.metric import Metric


class CohenKappaScore(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.tot_preds = defaultdict(lambda: 0)
        self.tot_grth = defaultdict(lambda: 0)
        self.tot = 0

    def update(self, preds, grth):
        self.num_correct += (preds == grth).sum().item()
        self.tot += len(grth)

        values, counts = np.unique(preds, return_counts=True)
        for v, c in zip(values, counts):
            self.tot_preds[v.item()] += c.item()

        values, counts = np.unique(grth, return_counts=True)
        for v, c in zip(values, counts):
            self.tot_grth[v.item()] += c.item()

    def __call__(self, *args, **kwargs):
        accuracy = self.num_correct / self.tot
        by_chance_accuracy = sum([self.tot_preds[key] * self.tot_grth[key] for key in self.tot_preds]) / self.tot ** 2
        return (accuracy - by_chance_accuracy) / (1 - by_chance_accuracy)
