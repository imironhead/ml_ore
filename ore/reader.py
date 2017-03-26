"""
reader class to handle different datasets.
"""
import numpy

from source_cifar_10 import SourceCifar10
from source_kaggle_mnist import SourceKaggleMnist
from source_lsun import SourceLsun
from source_mnist import SourceMnist


class Reader(object):
    """
    """
    @staticmethod
    def create_source(dataset, data_path=None):
        """
        data source virtual constructor. data_path can be None to use default
        path.
        """
        source_clazzs = [
            SourceCifar10, SourceKaggleMnist, SourceLsun, SourceMnist]
        source_clazz = None

        for clazz in source_clazzs:
            if clazz.include(dataset):
                source_clazz = clazz
                break

        if source_clazz is None:
            raise Exception('dataset {} is not suported'.format(dataset))

        return source_clazz(dataset, data_path)

    def __init__(self, dataset, data_path=None):
        """
        """
        self._source = Reader.create_source(dataset, data_path)
        self._indice = numpy.random.permutation(self._source.size)
        self._position = 0

    @property
    def cite(self):
        """
        """
        return self._source.cite

    @property
    def info(self):
        """
        """
        return self._source.info

    @property
    def size(self):
        """
        """
        return self._source.size

    def next_batch(self, batch_size, **options):
        """
        map_fn: map_fn(source_numpy_array), return target_numpy_array
        one_hot: return one_hot label if it's True
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
