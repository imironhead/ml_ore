"""
reader class to handle different datasets.
"""
from builtins import super
from reader import Reader
from six.moves import range


class DeterministicReader(Reader):
    """
    DeterministicReader read batch samples from source in fixed order. The
    order is the same as raw data if it's possible.

    Typical use case for DeterministicReader:
    Read test data without labels, use this reader to ensure samples are
    fetched in same and fixed order.
    """
    def __init__(self, dataset, range_percentage=(0, 100), data_path=None):
        """
        dataset:
            Index of specified dataset.
        range_percentage:
            A tuple to indicate part of raw data. For example, (0, 100) keep
            entire data set while (0, 50) keep half dataset.
        data_path:
            A user defined path to the dataset. Use default path if it's None.
        """
        super().__init__(dataset, range_percentage, data_path)

        self._position = 0

    def next_batch(self, batch_size, **options):
        """
        Reader next batch. However, DeterministicReader does not drop any
        samples. If the number of remain samples is less than batch_size,
        ignore the batch_size and return all remain samples.

        map_fn:
            Given batch samples, return new batch samples.
            map_fn(source_numpy_array), return target_numpy_array
        one_hot:
            Return one_hot label if it's True
        """
        is_new_epoch = self._position == 0

        head = self._position
        tail = min(self._position + batch_size, self.size)

        new_batch = self._source.batch(range(head, tail), **options)

        if tail == self.size:
            self._position = 0
        else:
            self._position += batch_size

        return new_batch + (is_new_epoch,)
