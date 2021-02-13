from metrics.base.metric import Metric


class Accuracy(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.num_examples = 0

    def update(self, preds, grth):
        self.num_correct += (preds == grth).sum().item()
        self.num_examples += len(grth)

    def __call__(self, *args, **kwargs):
        return self.num_correct / self.num_examples
