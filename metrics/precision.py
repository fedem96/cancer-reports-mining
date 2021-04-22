from metrics.base.metric import Metric


class Precision(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, preds, grth):
        correct = preds == grth
        wrong = ~correct
        self.TP += (correct & (preds == 1)).sum().item()
        self.FP += (wrong & (preds == 1)).sum().item()
        self.FN += (wrong & (grth == 1)).sum().item()

    def __call__(self, *args, **kwargs):
        d = self.TP + self.FP
        return self.TP / d if d > 0 else 0
