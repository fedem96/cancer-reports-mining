import numpy as np

from callbacks.base_callback import Callback


class EarlyStopping(Callback):

    def __init__(self):
        pass

    def on_fit_start(self, model):
        pass

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
        pass

    def on_epoch_end(self, model, epoch):
        pass

    def on_fit_end(self, model):
        pass


class EarlyStoppingSW(Callback):
    """Stop training only when loss does not improve over the mean of a fixed-size sliding window
    Arguments:
        patience: size of the sliding window (number of epochs)
    """

    def __init__(self, monitor='Loss', min_delta=0, patience=0, verbose=False, mode='auto', baseline=None, from_epoch=0):
        super(EarlyStoppingSW, self).__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.verbose = verbose
        self.baseline = baseline
        self.from_epoch = from_epoch

        if mode == 'auto':
            if 'loss' in monitor.lower():
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            assert mode in ['min', 'max']
            self.mode = mode

        if self.mode == 'max' and baseline is not None:
            self.baseline = -self.baseline

    def on_fit_start(self, model):
        self.window = []
        self.stopped_epoch = 0
        self.best = np.Inf
        self.baseline_reached = False

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

        if self.baseline is None or self.last < self.baseline:
            self.baseline_reached = True

        if self.window == []:
            self.window = [self.last]

        out_of_patience = self.patience <= 0 or len(self.window) == self.patience

        if epoch >= self.from_epoch and self.baseline_reached and self.last > np.mean(self.window) + self.min_delta and out_of_patience:
            model.stop_training = True
            self.stopped_epoch = epoch
        else:
            self.window.append(self.last)
            if len(self.window) > self.patience:
                self.window.pop(0)

        if self.last < self.best:
            self.best = self.last

    def on_epoch_end(self, model, epoch):
        pass

    def on_fit_end(self, model):
        if self.stopped_epoch > 0:
            if self.mode == 'max': self.last = -self.last
            if self.mode == 'max': self.best = -self.best
            if self.verbose:
                print('Early stop at epoch {:d} with best {} {:.3f}, last {:.3f}' .format(self.stopped_epoch, self.monitor, self.best, self.last))
