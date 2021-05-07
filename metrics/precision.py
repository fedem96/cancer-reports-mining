from metrics.base.metric import Metric


class Precision(Metric):
    def __init__(self, cls=1):
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
        self.TP += (correct & (preds == self.cls)).sum().item()
        self.FP += (wrong & (preds == self.cls)).sum().item()
        self.FN += (wrong & (grth == self.cls)).sum().item()

    def __call__(self, *args, **kwargs):
        d = self.TP + self.FP
        return self.TP / d if d > 0 else 0
