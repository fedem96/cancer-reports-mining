from callbacks.base_callback import Callback
import os

import numpy as np

from utils.serialization import save


class ModelCheckpoint(Callback):

    def __init__(self, output_dir, monitor, verbose=False,
                 save_best=True, save_last=False, save_freq=None, mode='auto'):
        assert save_best or save_last or save_freq is not None
        self.output_dir = output_dir
        self.monitor = monitor
        self.verbose = verbose
        self.save_best = save_best
        self.save_last = save_last
        self.save_freq = save_freq

        if mode == 'auto':
            if 'loss' in monitor.lower():
                self.mode = 'min'
            else:
                self.mode = 'max'
        else:
            assert mode in ['min', 'max']
            self.mode = mode

        self.best = None
        if save_freq is not None:
            assert type(save_freq) == int and save_freq > 0

    def on_fit_start(self, model):
        self.best = np.Inf

    def on_epoch_start(self, model, epoch):
        pass

    def on_train_epoch_start(self, model, epoch):
        pass

    def on_train_batch_start(self, model):
        pass

    def on_train_batch_end(self, model, epoch):
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
        metric_val = sum(map(lambda m: m(), metrics[self.monitor].values())) / len(metrics[self.monitor])
        if self.mode == 'max':
            metric_val *= -1
        if self.save_last:
            if self.verbose:
                print("epoch {}: saving last model".format(epoch))
            self.best = metric_val
            self._save_model(model, "model_last.pth")
        if self.save_best and metric_val < self.best:
            if self.verbose:
                print("epoch {}: saving new best model".format(epoch))
            self.best = metric_val
            self._save_model(model, "model_best.pth")
        if self.save_freq is not None and epoch % self.save_freq == 0:
            if self.verbose:
                print("epoch {}: periodic save".format(epoch))
            self._save_model(model, "model_{}.pth".format(epoch))

    def on_epoch_end(self, model, epoch):
        pass

    def on_fit_end(self, model):
        best = self.best if self.mode == 'min' else -self.best
        if self.verbose:
            print("best {}: {}".format(self.monitor, best))

    def _save_model(self, model, filename):
        save(model, os.path.join(self.output_dir, filename))
