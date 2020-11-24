import inspect
from collections import defaultdict
import os
from timeit import default_timer as timer

from tableformatter import generate_table, AlternatingRowGrid
from colorama import Fore, Back, Style


def dict_str(dictionary, digits=3):
    return "{"+ " ".join([n + " " + str(round(l, digits)) for n, l in dictionary.items()]) + "}"


class MetricsLogger:

    def __init__(self, terminal=None, tensorboard_dir=None, history_size=1):
        if terminal is None and tensorboard_dir is None:
            raise ValueError("the logger has to log something")
        self.metrics_train = {}
        self.metrics_val = {}
        self.tables = {}
        self.terminal = terminal
        self.tensorboard_dir = tensorboard_dir
        self.history_size = history_size
        self.history = defaultdict(lambda: defaultdict(lambda: []))
        self.best_train = defaultdict(lambda: defaultdict(lambda: None))
        self.best_val = defaultdict(lambda: defaultdict(lambda: None))
        self.compare = {}
        if tensorboard_dir is not None:
            from torch.utils.tensorboard import SummaryWriter
            self.sw_train = SummaryWriter(os.path.join(tensorboard_dir, "train"))
            self.sw_val = SummaryWriter(os.path.join(tensorboard_dir, "val"))

    def prepare(self, epoch, metrics_groups):
        self.start = timer()
        self.epoch = epoch
        for metrics_group in metrics_groups:
            if inspect.isclass(metrics_groups[metrics_group]):
                assert issubclass(metrics_groups[metrics_group], Metric)
                self.metrics_train[metrics_group] = defaultdict(lambda cls=metrics_groups[metrics_group]: cls())
                self.metrics_val[metrics_group] = defaultdict(lambda cls=metrics_groups[metrics_group]: cls())
                self.compare[metrics_group] = metrics_groups[metrics_group]().compare
            else:
                self.metrics_train[metrics_group] = defaultdict(lambda: 0)
                self.metrics_val[metrics_group] = defaultdict(lambda: 0)
                self.compare[metrics_group] = metrics_groups[metrics_group]
                assert metrics_groups[metrics_group] == min or metrics_groups[metrics_group] == max

    def accumulate_train(self, metrics_train, dividend=1):
        for group in metrics_train:
            for metric in metrics_train[group]:
                if isinstance(self.metrics_train[group][metric], Metric):
                    self.metrics_train[group][metric].update(*metrics_train[group][metric])
                else:
                    self.metrics_train[group][metric] += metrics_train[group][metric] / dividend

    def accumulate_val(self, metrics_val, dividend=1):
        for group in metrics_val:
            for metric in metrics_val[group]:
                if isinstance(self.metrics_val[group][metric], Metric):
                    self.metrics_val[group][metric].update(*metrics_val[group][metric])
                else:
                    self.metrics_val[group][metric] += metrics_val[group][metric] / dividend

    def log(self):
        self.end = timer()
        if self.terminal == "table":
            self._log_console_table()
        elif self.terminal == "simple":
            self._log_console_simple()

        if self.tensorboard_dir is not None:
            self._log_tensorboard()

    def _log_console_simple(self):
        out = "epoch {}, elapsed time {}".format(self.epoch, (self.end-self.start)) + "\ntrain:"
        for group in self.metrics_train:
            out += " " + group + dict_str(self.metrics_train[group])
        if len(self.metrics_val) > 0:
            out += "\nval:"
            for group in self.metrics_val:
                out += " " + group + dict_str(self.metrics_val[group])
        print(out + "\n")

    def _log_console_table(self):
        print("\nepoch {}, elapsed time {}".format(self.epoch, (self.end-self.start)))
        for group in self.metrics_train:
            sup = self.epoch+1
            inf = max(0, sup - self.history_size)
            columns = [group] + ['epoch {}'.format(i) for i in range(inf, sup)] + ["Best"]

            rows = []
            for metric in self.metrics_train[group]:
                values = self.history[group][metric]
                value = self.metrics_train[group][metric]
                if callable(value):
                    value = value()
                best = self.best_train[group][metric]
                if best is None or self.compare[group](value, best) == value:
                    best = self.best_train[group][metric] = value
                str_best = str(round(best, 4))
                color = Fore.GREEN if value == best else Fore.RESET
                values.append(color + str(round(value, 4)) + Fore.RESET)
                if group in self.metrics_val and metric in self.metrics_val[group]:
                    value = self.metrics_val[group][metric]
                    if callable(value):
                        value = value()
                    best = self.best_val[group][metric]
                    if best is None or self.compare[group](value, best) == value:
                        best = self.best_val[group][metric] = value
                    color = Fore.GREEN if value == best else Fore.RESET
                    values[-1] += "\n" + color + str(round(value, 4)) + Fore.RESET
                    str_best += "\n" + Style.BRIGHT + str(round(best, 4)) + Style.NORMAL
                if len(values) == self.history_size + 1:
                    values.pop(0)
                rows.append([metric, *values, str_best])

            print(generate_table(rows, columns, grid_style=AlternatingRowGrid(Back.RESET, Back.LIGHTBLACK_EX)), end='')

    def _log_tensorboard(self):
        # for group in self.metrics_train:
        #     for name, value in self.metrics_train[group].items():
        #         self.tensorboard.add_scalars(f'{group}/{name}', {'train': value, 'val': self.metrics_val[group][name]}, self.epoch)
        for group in self.metrics_train:
            for name, value in self.metrics_train[group].items():
                if callable(value):
                    value = value()
                self.sw_train.add_scalar(f'{group}/{name}', value, self.epoch)
        for group in self.metrics_val:
            for name, value in self.metrics_val[group].items():
                if callable(value):
                    value = value()
                self.sw_val.add_scalar(f'{group}/{name}', value, self.epoch)

    def close(self):
        if self.tensorboard_dir is not None:
            self.sw_train.close()
            self.sw_val.close()
        return self

    def print_best(self, output_file=None):
        out = "best values after " + str(1+self.epoch) + " epochs\ntrain:"
        for group in self.metrics_train:
            out += " " + group + dict_str(self.best_train[group])
        if len(self.metrics_val) > 0:
            out += "\nval:"
            for group in self.metrics_val:
                out += " " + group + dict_str(self.best_val[group])
        print(out + "\n")
        if output_file is not None:
            with open(output_file, "wt") as file:
                file.write(out)


class Metric:
    def __init__(self, compare):
        self.compare = compare


class MacroF1Score(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.TP = defaultdict(lambda: 0)
        self.FP = defaultdict(lambda: 0)
        self.FN = defaultdict(lambda: 0)

    def update(self, preds, grth):
        correct = preds == grth
        wrong = ~correct
        for cls in preds.unique().cpu().numpy():
            self.TP[cls] += (correct & (preds == cls)).sum().item()
            self.FP[cls] += (wrong & (preds == cls)).sum().item()
        for cls in grth.unique().cpu().numpy():
            self.FN[cls] += (wrong & (grth == cls)).sum().item()

    def __call__(self, *args, **kwargs):
        # f1 = 2 * (p * r) / (p + r)
        precisions = [self.TP[key] / (self.TP[key] + self.FP[key]) if (self.TP[key] + self.FP[key]) > 0 else 0 for key in self.FN]
        recalls = [self.TP[key] / (self.TP[key] + self.FN[key]) for key in self.FN]
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0 for p, r in zip(precisions, recalls)]
        # return f1_score(grth.cpu(), preds_classes.cpu(), average='macro')
        return sum(f1_scores) / len(f1_scores)


class CohenKappaScore(Metric):
    def __init__(self):
        super().__init__(max)
        self.reset()

    def reset(self):
        self.num_correct = 0
        self.tot_preds = defaultdict(lambda: 0)
        self.tot_grth = defaultdict(lambda: 0)
        self.tot = 0

    def update(self, preds, grth):
        self.num_correct += (preds == grth).sum().item()
        self.tot += len(grth)

        values, counts = preds.unique(return_counts=True)
        for v, c in zip(values, counts):
            self.tot_preds[v.item()] += c.item()

        values, counts = grth.unique(return_counts=True)
        for v, c in zip(values, counts):
            self.tot_grth[v.item()] += c.item()

    def __call__(self, *args, **kwargs):
        accuracy = self.num_correct / self.tot
        by_chance_accuracy = sum([self.tot_preds[key] * self.tot_grth[key] for key in self.tot_preds]) / self.tot ** 2
        return (accuracy - by_chance_accuracy) / (1 - by_chance_accuracy)


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
