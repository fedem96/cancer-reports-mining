from collections import defaultdict

from tableformatter import generate_table, AlternatingRowGrid
from colorama import Fore, Back, Style


def dict_str(dictionary, digits=3):
    return "{"+ " ".join([n + " " + str(round(l, digits)) for n, l in dictionary.items()]) + "}"


class MetricsLogger:

    def __init__(self, terminal=None, tensorboard=None, history_size=1):
        if terminal is None and tensorboard is None:
            raise ValueError("the logger has to log something")
        # TODO: init tensorboard
        self.metrics_train = {}
        self.metrics_val = {}
        self.tables = {}
        self.terminal = terminal
        self.tensorboard = tensorboard
        self.history_size = history_size
        self.history = defaultdict(lambda: defaultdict(lambda: []))
        self.best_train = defaultdict(lambda: defaultdict(lambda: None))
        self.best_val = defaultdict(lambda: defaultdict(lambda: None))
        self.compare = {}

    def prepare(self, epoch, metrics_groups):
        self.epoch = epoch
        for metrics_group in metrics_groups:
            self.metrics_train[metrics_group] = defaultdict(lambda: 0)
            self.metrics_val[metrics_group] = defaultdict(lambda: 0)
            self.compare[metrics_group] = metrics_groups[metrics_group]
            assert metrics_groups[metrics_group] == min or metrics_groups[metrics_group] == max

    def accumulate_train(self, metrics_train, dividend=1):
        for group in metrics_train:
            for metric in metrics_train[group]:
                self.metrics_train[group][metric] += metrics_train[group][metric] / dividend

    def accumulate_val(self, metrics_val, dividend=1):
        for group in metrics_val:
            for metric in metrics_val[group]:
                self.metrics_val[group][metric] += metrics_val[group][metric] / dividend

    def log(self):
        if self.terminal == "table":
            self._log_console_table()
        elif self.terminal == "simple":
            self._log_console_simple()

    def _log_console_simple(self):
        out = "epoch " + str(self.epoch)
        for group in self.metrics_train:
            out += " " + group + dict_str(self.metrics_train[group])
        print(out)

    def _log_console_table(self):

        for group in self.metrics_train:
            sup = self.epoch+1
            inf = max(0, sup - self.history_size)
            columns = [group] + ['epoch {}'.format(i) for i in range(inf, sup)] + ["Best"]

            rows = []
            for metric in self.metrics_train[group]:
                values = self.history[group][metric]
                value = self.metrics_train[group][metric]
                best = self.best_train[group][metric]
                if best is None or self.compare[group](value, best) == value:
                    best = self.best_train[group][metric] = value
                str_best = str(round(best, 4))
                color = Fore.GREEN if value == best else Fore.RESET
                values.append(color + str(round(value, 4)) + Fore.RESET)
                if group in self.metrics_val and metric in self.metrics_val[group]:
                    value = self.metrics_val[group][metric]
                    best = self.best_val[group][metric]
                    if best is None or self.compare[group](value, best) == value:
                        best = self.best_val[group][metric] = value
                    color = Fore.GREEN if value == best else Fore.RESET
                    values[-1] += "\n" + color + str(round(value, 4)) + Fore.RESET
                    str_best += "\n" + Style.BRIGHT + str(round(best, 4)) + Style.NORMAL
                if len(values) == self.history_size + 1:
                    values.pop(0)
                rows.append([metric, *values, str_best])

            print(generate_table(rows, columns, grid_style=AlternatingRowGrid(Back.RESET, Back.LIGHTWHITE_EX)))

