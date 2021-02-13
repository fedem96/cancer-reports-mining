from abc import abstractmethod, ABC


class Callback(ABC):
    @abstractmethod
    def on_fit_start(self, model):
        pass

    @abstractmethod
    def on_epoch_start(self, model, epoch):
        pass

    @abstractmethod
    def on_train_epoch_start(self, model, epoch):
        pass

    @abstractmethod
    def on_train_batch_start(self, model):
        pass

    @abstractmethod
    def on_train_batch_end(self, model, metrics):
        pass

    @abstractmethod
    def on_train_epoch_end(self, model, epoch, metrics):
        pass

    @abstractmethod
    def on_validation_epoch_start(self, model, epoch):
        pass

    @abstractmethod
    def on_validation_batch_start(self, model):
        pass

    @abstractmethod
    def on_validation_batch_end(self, model, metrics):
        pass

    @abstractmethod
    def on_validation_epoch_end(self, model, epoch, metrics):
        pass

    @abstractmethod
    def on_epoch_end(self, model, epoch):
        pass

    @abstractmethod
    def on_fit_end(self, model):
        pass
