from metrics.base.metric import Metric


class F1Score(Metric):
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
        # f1 = 2 * (p * r) / (p + r)
        p = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
        r = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
        f1_score = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
        return f1_score
