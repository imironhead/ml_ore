"""
reader class to handle different datasets.
"""
from source_cifar_10 import SourceCifar10
from source_kaggle_mnist import SourceKaggleMnist
from source_lsun import SourceLsun
from source_mnist import SourceMnist


class Reader(object):
    """
    """
    @staticmethod
    def create_source(dataset, range_percentage=(0, 100), data_path=None):
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

        return source_clazz(dataset, range_percentage, data_path)

    def __init__(self, dataset, range_percentage=(0, 100), data_path=None):
        """
        """
        # sanity check
        if range_percentage is None:
            raise Exception('range must be inside [0, 100)')

        head, tail = range_percentage

        if head < 0 or head >= 100 or tail <= head or tail > 100:
            raise Exception('range must be inside [0, 100)')

        self._source = Reader.create_source(
            dataset, range_percentage, data_path)

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
        map_fn:
            map_fn(source_numpy_array), return target_numpy_array
        one_hot:
            return one_hot label if it's True
        """
        raise Exception('Reader.next_batch: should never be here!')

        return None
