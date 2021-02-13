from metrics.base.metric import Metric


class Average(Metric):
    def __init__(self, compare):
        super().__init__(compare)
        self.reset()

    def reset(self):
        self.tot = 0

    def update(self, l, n):
        self.tot += l / n

    def __call__(self, *args, **kwargs):
        return self.tot
