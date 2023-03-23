import numpy as np
from typing import Tuple


class Dataset:

    def __init__(self, raw: list or np.ndarray, gnd: list or np.ndarray, batch_size: int, shuffle: bool = True):

        assert len(raw) == len(gnd), "lengths of raw and gnd must be the same!"

        self.batch_size = batch_size
        self.raw = np.asarray(raw)  # images
        self.gnd = np.asarray(gnd)  # ground truth (annotation/label)

        self._length = len(self.raw)
        self.indexes = np.asarray([i for i in range(self._length)])
        self.current_index = 0

        self._shuffle = shuffle
        if self._shuffle:
            self.shuffle()

        self._isEnd = False

    def __len__(self):
        return self._length

    def shuffle(self):
        np.random.shuffle(self.raw)
        np.random.shuffle(self.gnd)

    def reset(self):
        self.shuffle()
        self._isEnd = False
        self.current_index = 0

    def is_end(self):
        return self._isEnd

    def report_progress(self):
        return self.current_index // self.batch_size, len(self) // self.batch_size

    def auto_fetch(self) -> Tuple[np.ndarray, np.ndarray]:

        self._isEnd = False

        if self.current_index + self.batch_size >= len(self):
            self.current_index = len(self) - self.batch_size

            _r1 = self.raw[self.current_index:self.current_index + self.batch_size]
            _r2 = self.gnd[self.current_index:self.current_index + self.batch_size]

            if self._shuffle: self.shuffle()

            self._isEnd = True

        else:
            _r1 = self.raw[self.current_index:self.current_index + self.batch_size]
            _r2 = self.gnd[self.current_index:self.current_index + self.batch_size]

        self.current_index = (self.current_index + self.batch_size) % len(self)

        return _r1, _r2

    def auto_fetch_with_format(self, dictionary: dict) -> dict:

        assert 'raw' in dictionary.keys() and 'gnd' in dictionary.keys()

        raw, gnd = self.auto_fetch()

        return {
            dictionary['raw']: raw,
            dictionary['gnd']: gnd
        }
