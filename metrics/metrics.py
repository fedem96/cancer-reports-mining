class Metrics:
    def __init__(self, metrics={}):
        self.metrics = metrics

    def reset(self):
        for metric_group in self.metrics.values():
            for metric in metric_group.values():
                metric.reset()

    def __repr__(self):
        return self.metrics.__repr__()

    def __str__(self):
        return self.metrics.__str__()
