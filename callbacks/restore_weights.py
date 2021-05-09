import copy

import numpy as np

from callbacks.base_callback import Callback


class RestoreWeights(Callback):
    """Restore best weights after training
    """

    def __init__(self, monitor='Loss', verbose=False, mode='auto'):
        super(RestoreWeights, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.best_epoch = None
        self.best_state_dict = None

        if mode == 'auto':
            if 'loss' in monitor.lower():
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            assert mode in ['min', 'max']
            self.mode = mode

    def on_fit_start(self, model):
        self.best = np.Inf

    def on_epoch_start(self, model, epoch):
        pass

    def on_train_epoch_start(self, model, epoch):
        pass

    def on_train_batch_start(self, model):
        pass

    def on_train_batch_end(self, model, metrics):
        pass

    def on_train_epoch_end(self, model, epoch, metrics):
        pass

    def on_validation_epoch_start(self, model, epoch):
        pass

    def on_validation_batch_start(self, model):
        pass

    def on_validation_batch_end(self, model, metrics):
        pass

    def on_validation_epoch_end(self, model, epoch, metrics):
        self.last = sum(map(lambda v: v(), metrics[self.monitor].values())) / len(metrics[self.monitor])
        if self.mode == 'max':
            self.last = -self.last

        if self.last < self.best:
            self.best = self.last
            self.best_epoch = epoch
            self.best_state_dict = copy.deepcopy(model.state_dict())

    def on_epoch_end(self, model, epoch):
        pass

    def on_fit_end(self, model):
        if self.best_state_dict is None:
            print("WARNING: best_state_dict is None, weights not restored")
        else:
            if self.verbose:
                print(f"restoring best weights from epoch {self.best_epoch}")
            model.load_state_dict(self.best_state_dict)
