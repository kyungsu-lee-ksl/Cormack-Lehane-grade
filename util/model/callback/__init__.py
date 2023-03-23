from abc import abstractmethod



class CallBack:

    def __init__(self):
        pass

    @abstractmethod
    def __on_batch_begin__(self, current_epoch: int, current_batch: int, feed_dict: dict, log: list, tensors: list):
        pass

    @abstractmethod
    def __on_batch_end__(self, current_epoch: int, current_batch: int, feed_dict: dict, log: list, tensors: list):
        pass

    # @abstractmethod
    # def __on_epoch_begin__(self, current_epoch: int, feed_dict: dict, log: list, tensors: list):
    #     pass

    @abstractmethod
    def __on_epoch_end__(self, current_epoch: int, dataset, log: list, tensors: list):
        pass
