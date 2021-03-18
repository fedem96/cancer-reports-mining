import os
from collections import defaultdict
from timeit import default_timer as timer

import aim
from colorama import Fore, Back, Style
from tableformatter import generate_table, AlternatingRowGrid

from callbacks.base_callback import Callback


def dict_str(dictionary, digits=3):
    return "{"+ " ".join([n + " " + str(round(l, digits)) for n, l in dictionary.items()]) + "}"


class MetricsLogger(Callback):

    def __init__(self, terminal=None, tensorboard_dir=None, aim_name=None, history_size=1):
        if terminal is None and tensorboard_dir is None and aim_name is not None:
            raise ValueError("the logger has to log something")
        self.tables = {}
        self.terminal = terminal
        self.tensorboard_dir = tensorboard_dir
        self.log_aim = aim_name is not None
        self.history_size = history_size
        self.history = defaultdict(lambda: defaultdict(lambda: []))
        self.best_train = defaultdict(lambda: defaultdict(lambda: None))
        self.best_val = defaultdict(lambda: defaultdict(lambda: None))
        self.compare = {}
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.sw_train = SummaryWriter(os.path.join(tensorboard_dir, "train"))
            self.sw_val = SummaryWriter(os.path.join(tensorboard_dir, "val"))
        if self.log_aim:
            aim.Session(experiment=aim_name)

    def prepare(self, epoch):#, metrics_groups):
        self.start = timer()
        self.epoch = epoch
        # for metrics_group in metrics_groups:
        #     if inspect.isclass(metrics_groups[metrics_group]):
        #         assert issubclass(metrics_groups[metrics_group], Metric)
        #         # metrics_train[metrics_group] = defaultdict(lambda cls=metrics_groups[metrics_group]: cls())
        #         # metrics_val[metrics_group] = defaultdict(lambda cls=metrics_groups[metrics_group]: cls())
        #         self.compare[metrics_group] = metrics_groups[metrics_group]().compare
        #     else:
        #         # metrics_train[metrics_group] = defaultdict(lambda: 0)
        #         # metrics_val[metrics_group] = defaultdict(lambda: 0)
        #         self.compare[metrics_group] = metrics_groups[metrics_group]
        #         assert metrics_groups[metrics_group] == min or metrics_groups[metrics_group] == max

    # @DeprecationWarning
    # def accumulate_train(self, metrics_train, dividend=1):
    #     for group in metrics_train:
    #         for metric in metrics_train[group]:
    #             if isinstance(self.metrics_train[group][metric], Metric):
    #                 self.metrics_train[group][metric].update(*metrics_train[group][metric])
    #             else:
    #                 self.metrics_train[group][metric] += metrics_train[group][metric] / dividend
    #
    # @DeprecationWarning
    # def accumulate_val(self, metrics_val, dividend=1):
    #     for group in metrics_val:
    #         for metric in metrics_val[group]:
    #             if isinstance(self.metrics_val[group][metric], Metric):
    #                 self.metrics_val[group][metric].update(*metrics_val[group][metric])
    #             else:
    #                 self.metrics_val[group][metric] += metrics_val[group][metric] / dividend

    def log(self, metrics_train, metrics_val):
        self.end = timer()
        if self.terminal == "table":
            self._log_console_table(metrics_train, metrics_val)
        elif self.terminal == "simple":
            self._log_console_simple(metrics_train, metrics_val)

        if self.tensorboard_dir is not None:
            self._log_tensorboard(metrics_train, metrics_val)

        if self.log_aim is not None:
            self._log_aim(metrics_train, metrics_val)

    def _log_console_simple(self, metrics_train, metrics_val):
        out = "epoch {}, elapsed time {}".format(self.epoch, (self.end-self.start)) + "\ntrain:"
        for group in metrics_train:
            out += " " + group + dict_str(metrics_train[group])
        if len(metrics_val) > 0:
            out += "\nval:"
            for group in metrics_val:
                out += " " + group + dict_str(metrics_val[group])
        print(out + "\n")

    def _log_console_table(self, metrics_train, metrics_val):
        print("\nepoch {}, elapsed time {}".format(self.epoch, (self.end-self.start)))
        for group in metrics_train:
            sup = self.epoch+1
            inf = max(0, sup - self.history_size)
            columns = [group] + ['epoch {}'.format(i) for i in range(inf, sup)] + ["Best"]

            rows = []
            for metric in metrics_train[group]:
                values = self.history[group][metric]
                value = metrics_train[group][metric]
                if callable(value):
                    value = value()
                best = self.best_train[group][metric]
                if best is None or metrics_train[group][metric].compare(value, best) == value:
                    best = self.best_train[group][metric] = value
                str_best = str(round(best, 4))
                color = Fore.GREEN if value == best else Fore.RESET
                values.append(color + str(round(value, 4)) + Fore.RESET)
                if group in metrics_val and metric in metrics_val[group]:
                    value = metrics_val[group][metric]
                    if callable(value):
                        value = value()
                    best = self.best_val[group][metric]
                    if best is None or metrics_val[group][metric].compare(value, best) == value:
                        best = self.best_val[group][metric] = value
                    color = Fore.GREEN if value == best else Fore.RESET
                    values[-1] += "\n" + color + str(round(value, 4)) + Fore.RESET
                    str_best += "\n" + Style.BRIGHT + str(round(best, 4)) + Style.NORMAL
                if len(values) == self.history_size + 1:
                    values.pop(0)
                rows.append([metric, *values, str_best])

            print(generate_table(rows, columns, grid_style=AlternatingRowGrid(Back.RESET, Back.LIGHTBLACK_EX)), end='')

    def _log_tensorboard(self, metrics_train, metrics_val):
        # for group in metrics_train:
        #     for name, value in metrics_train[group].items():
        #         self.tensorboard.add_scalars(f'{group}/{name}', {'train': value, 'val': metrics_val[group][name]}, self.epoch)
        for group in metrics_train:
            for name, value in metrics_train[group].items():
                if callable(value):
                    value = value()
                self.sw_train.add_scalar(f'{group}/{name}', value, self.epoch)
        for group in metrics_val:
            for name, value in metrics_val[group].items():
                if callable(value):
                    value = value()
                self.sw_val.add_scalar(f'{group}/{name}', value, self.epoch)

    def _log_aim(self, metrics_train, metrics_val):
        for group in metrics_train:
            for metric in metrics_train[group]:
                value = metrics_train[group][metric]
                if callable(value):
                    value = value()
                aim.track(value, name=group.replace("-", ""), epoch=self.epoch, var=metric, dataset='train')
        for group in metrics_val:
            for metric in metrics_val[group]:
                value = metrics_val[group][metric]
                if callable(value):
                    value = value()
                aim.track(value, name=group.replace("-", ""), epoch=self.epoch, var=metric, dataset='val')

    def log_hyper_parameters_and_best_metrics(self, info, hparams):
        metrics_train = {"{}_{}".format(group, metric): self.best_train[group][metric] for group in self.best_train for metric in self.best_train[group]}
        metrics_val = {"{}_{}".format(group, metric): self.best_val[group][metric] for group in self.best_val for metric in self.best_val[group]}
        metrics_train.update({"{}_AVG".format(group): sum(self.best_train[group].values()) / len(self.best_train[group]) for group in metrics_train if len(self.best_train[group]) > 0})
        metrics_val.update({"{}_AVG".format(group): sum(self.best_val[group].values()) / len(self.best_val[group]) for group in metrics_val if len(self.best_val[group]) > 0})
        self.sw_train.add_hparams(dict(hparams, set="train"), {"best/"+k: v for k,v in metrics_train.items()})
        self.sw_val.add_hparams(dict(hparams, set="val"), {"best/"+k: v for k,v in metrics_val.items()})
        try:
            aim.set_params(info, name='info')
            aim.set_params(hparams, name='hparams')
            aim.set_params({k.replace("-", "").lower().replace("accuracy", "acc"): v for k,v in metrics_train.items()}, name='train')
            aim.set_params({k.replace("-", "").lower().replace("accuracy", "acc"): v for k,v in metrics_val.items()}, name='val')
        except AttributeError:
            print("hey")

    def print_best(self,  output_file=None):
        out = "best values after " + str(1+self.epoch) + " epochs\ntrain:"
        for group in self.best_train:
            out += " " + group + dict_str(self.best_train[group])
        if len(self.best_val) > 0:
            out += "\nval:"
            for group in self.best_val:
                out += " " + group + dict_str(self.best_val[group])
        print(out + "\n")
        if output_file is not None:
            with open(output_file, "wt") as file:
                file.write(out)

    def close(self):
        if self.tensorboard_dir is not None:
            self.sw_train.close()
            self.sw_val.close()
        return self

    def on_fit_start(self, model):
        pass

    def on_epoch_start(self, model, epoch):
        self.prepare(epoch)

    def on_train_epoch_start(self, model, epoch):
        pass

    def on_train_batch_start(self, model):
        pass

    def on_train_batch_end(self, model, metrics):
        pass

    def on_train_epoch_end(self, model, epoch, metrics):
        self.training_metrics = metrics

    def on_validation_epoch_start(self, model, epoch):
        pass

    def on_validation_batch_start(self, model):
        pass

    def on_validation_batch_end(self, model, metrics):
        pass

    def on_validation_epoch_end(self, model, epoch, metrics):
        self.validation_metrics = metrics

    def on_epoch_end(self, model, epoch):
        self.log(self.training_metrics, self.validation_metrics)

    def on_fit_end(self, model):
        self.log_hyper_parameters_and_best_metrics(model.info, model.hyperparameters)
        self.close()


# class ColumnsAverage(Metric):
#
#     def __init__(self, column_metric):
#         super().__init__(column_metric().compare)
#         self.metrics = defaultdict(lambda: column_metric())
#         self.reset()
#
#     def reset(self):
#         [self.metrics[col].reset() for col in self.metrics]
#
#     def update(self, preds, grth):
#         [self.metrics[col].update(preds[col], grth[col]) for col in self.grth.columns]
#
#     def __call__(self, *args, **kwargs):
#         values = [self.metrics[col]() for col in self.metrics]
#         return sum(values) / len(values)
#
#
# class AvgMacroF1Score(ColumnsAverage):
#     def __init__(self):
#         super().__init__(MacroF1Score)
