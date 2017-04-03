"""
a class to read mnist data set (kaggle version).
"""
import os
import numpy

import datasets


class SourceKaggleMnist(object):
    """
    a class to track raw mnist data.
    """
    @staticmethod
    def default_data_path(dataset):
        """
        default path of mnist dataset if it's not provided.
        """
        path_home = os.path.expanduser('~')

        return os.path.join(path_home, 'data', 'kaggle_mnist')

    @staticmethod
    def subsets():
        """
        """
        return [datasets.DATASET_KAGGLE_MNIST_TRAINING,
                datasets.DATASET_KAGGLE_MNIST_TEST]

    @staticmethod
    def include(dataset):
        """
        """
        return dataset in SourceKaggleMnist.subsets()

    @staticmethod
    def maybe_download(dataset, data_path):
        """
        download all 4 *.gz files if necessary.
        """
        # sanity check
        assert data_path is not None, 'data_path should not be None'

        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        path_file_test = os.path.join(data_path, 'test.csv')
        path_file_train = os.path.join(data_path, 'train.csv')

        if os.path.isfile(path_file_test) and os.path.isfile(path_file_train):
            return

        msg = 'auto downloading for kaggle data is not implemented. ' \
              'ore expect {} and {} exist.' \
              .format(path_file_test, path_file_train)

        raise Exception(msg)

    @staticmethod
    def pre_process(dataset, data_path):
        """
        """

    @staticmethod
    def default_map_fn(img):
        """
        remap the image. the default mapping is to do nothing.
        """
        return img

    def __init__(self, dataset, range_percentage=(0, 100), data_path=None):
        """
        """
        if data_path is None:
            data_path = SourceKaggleMnist.default_data_path(dataset)

        SourceKaggleMnist.maybe_download(dataset, data_path)
        SourceKaggleMnist.pre_process(dataset, data_path)

        if dataset == datasets.DATASET_KAGGLE_MNIST_TRAINING:
            path_file = os.path.join(data_path, 'train.csv')
        else:
            path_file = os.path.join(data_path, 'test.csv')

        assert os.path.isfile(path_file), 'need {}'.format(path_file)

        raw_data = numpy.genfromtxt(
            path_file, dtype=numpy.int32, delimiter=',', skip_header=1)

        if dataset == datasets.DATASET_KAGGLE_MNIST_TRAINING:
            self._images = raw_data[:, 1:]
            self._labels = raw_data[:, 0]
        else:
            self._images = raw_data
            self._labels = numpy.repeat(
                datasets.LABEL_INVALID, raw_data.shape[0])

        self._images = self._images.reshape(raw_data.shape[0], 28, 28, 1)
        self._images = self._images.astype(numpy.float32) / 127.5 - 1.0

        # NOTE: range must be dealt within each source due to the layout of
        #       sources may be different.
        head, tail = range_percentage
        size = self._labels.shape[0]
        head = head * size // 100
        tail = tail * size // 100

        if head >= tail:
            raise Exception('the range is too narrow')

        self._images = self._images[head:tail]
        self._labels = self._labels[head:tail]

    @property
    def cite(self):
        """
        """
        return 'https://www.kaggle.com/c/digit-recognizer/data'

    @property
    def info(self):
        """
        """
        return 'haha'

    @property
    def size(self):
        """
        """
        return self._labels.shape[0]

    def batch(self, idx_list=[], map_fn=default_map_fn.__func__,
              one_hot=False, **options):
        """
        idx_list: list of data indice.
        map_fn: map_fn(source_numpy_array), return target_numpy_array
        one_hot: return one_hot label if it's True
        """
        cnt = len(idx_list)
        ims = None
        idx = None

        for i, j in enumerate(idx_list):
            if j >= self._labels.shape[0]:
                raise Exception('invalid index {}'.format(j))

            img = self._images[j]
            img = map_fn(img)

            if ims is None:
                ims = numpy.zeros((cnt,) + img.shape)
                idx = numpy.zeros((cnt,), dtype=numpy.int32)

            ims[i] = img
            idx[i] = self._labels[j]

        if one_hot:
            tmp = idx
            idx = numpy.zeros((cnt, 10), dtype=numpy.float32)
            idx[numpy.arange(cnt), tmp] = 1.0

        return ims, idx
