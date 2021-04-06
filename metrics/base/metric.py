from abc import abstractmethod, ABC


class Metric(ABC):
    def __init__(self, compare):
        self.compare = compare

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, preds, grth):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def __str__(self):
        return str(self.__call__())

    def __repr__(self):
        return str(self.__call__())
