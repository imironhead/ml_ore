"""
reader class to handle different datasets.
"""
import numpy

import datasets


class Reader(object):
    """
    """
    def __init__(self, dataset_index, data_path=None, **options):
        """
        """
        self._source = datasets.prepare_source(dataset_index, data_path)
        self._indice = numpy.random.permutation(self._source.size)
        self._position = 0

    def cite(self):
        """
        """
        return self._source.cite

    def info(self):
        """
        """
        return self._source.info

    def size(self):
        """
        """
        return self._source.size

    def next_batch(self, batch_size, **options):
        """
        """
        is_new_epoch = False

        begin = self._position

        self._position += batch_size

        if self._position > len(self._indice):
            numpy.random.shuffle(self._indice)

            begin = 0

            is_new_epoch = True

            self._position = batch_size

            assert batch_size <= len(self._indice)

        new_batch = self._source.batch(
            self._indice[begin:self._position], **options)

        return new_batch + (is_new_epoch,)
