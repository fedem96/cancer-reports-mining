from metrics.accuracy import Accuracy
from metrics.base.metric import Metric


class DumbBaselineComparisonScore(Metric):
    def __init__(self, dumb_baseline_accuracy):
        super().__init__(max)
        self.dumb_baseline_accuracy = dumb_baseline_accuracy
        self.accuracy = Accuracy()
        self.reset()

    def reset(self):
        self.accuracy.reset()

    def update(self, preds, grth):
        self.accuracy.update(preds, grth)

    def __call__(self, *args, **kwargs):
        return (self.accuracy() - self.dumb_baseline_accuracy) / (1 - self.dumb_baseline_accuracy)
