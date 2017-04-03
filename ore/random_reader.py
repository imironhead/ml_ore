"""
reader class to handle different datasets.
"""
import numpy

from builtins import super
from reader import Reader


class RandomReader(Reader):
    """
    RandomReader read data in random order. The order is shuffled after each
    epoch.
    """
    def __init__(self, dataset, range_percentage=(0, 100), data_path=None):
        """
        """
        super().__init__(dataset, range_percentage, data_path)

        # reading ordering in current epoch.
        self._indice = numpy.random.permutation(self._source.size)

        # current reading position.
        self._position = 0

    def next_batch(self, batch_size, **options):
        """
        Reader next batch. RandomReader drops all samples in current epoch if
        the number of remain samples is less then batch_size. The reading order
        is shuffled for each new epoch.

        map_fn:
            Given batch samples, return new batch samples.
            map_fn(source_numpy_array), return target_numpy_array
        one_hot:
            return one_hot label if it's True

        Return[0]: batch samples
        Return[1]: True if this batch is from a new epoch.
        """
        is_new_epoch = False

        begin = self._position

        self._position += batch_size

        if self._position > self._indice.shape[0]:
            numpy.random.shuffle(self._indice)

            begin = 0

            is_new_epoch = True

            self._position = batch_size

            assert batch_size <= self._indice.shape[0]

        new_batch = self._source.batch(
            self._indice[begin:self._position], **options)

        return new_batch + (is_new_epoch,)
