import abc

class Model(abc.ABC):
    @abc.abstractmethod
    def send_request(self):
        raise NotImplementedError()