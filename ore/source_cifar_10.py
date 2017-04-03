"""
"""
import numpy
import os
import pickle
import shutil
import tarfile

from six.moves import range, urllib

import datasets


class SourceCifar10(object):
    """
    """
    @staticmethod
    def default_data_path(dataset):
        """
        """
        path_home = os.path.expanduser('~')

        return os.path.join(path_home, 'data', 'cifar-10-batches-py')

    @staticmethod
    def subsets():
        """
        """
        return [
            datasets.DATASET_CIFAR_10_TRAINING,
            datasets.DATASET_CIFAR_10_TEST]

    @staticmethod
    def include(dataset):
        """
        """
        return dataset in SourceCifar10.subsets()

    @staticmethod
    def download(dataset, data_path):
        """
        https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

        result:
        data_path/data_batch_1
        data_path/data_batch_2
        data_path/data_batch_3
        data_path/data_batch_4
        data_path/data_batch_5
        data_path/test_batch
        """
        if data_path is None:
            data_path = SourceCifar10.default_data_path(dataset)

        all_there = True

        # check if all batches are ready.
        for i in range(1, 6):
            file_name = 'data_batch_{}'.format(i)
            file_path = os.path.join(data_path, file_name)

            if not os.path.isfile(file_path):
                all_there = False
                break

        # check if test batch is ready.
        file_path = os.path.join(data_path, 'test_batch')

        if not os.path.isfile(file_path):
            all_there = False

        # return if all batches are downloaded, unzipped and moved.
        if all_there:
            return

        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        # download source to temp_path:
        # data_path/cifar-10-python.tar.gz
        gzip_path = os.path.join(data_path, 'cifar-10-python.tar.gz')

        # download
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        print('downloading {}'.format(url))

        urllib.request.urlretrieve(url, gzip_path)

        # unzip, move
        temp_source_path = os.path.join(data_path, 'cifar-10-batches-py')

        # unzip data_path/cifar-10-python.tar.gz to data_path
        # become data_path/cifar-10-batches-py/*
        with tarfile.open(gzip_path) as tar:
            tar.extractall(data_path)

        # move data_path/cifar-10-batches-py/* to data_path
        for name in os.listdir(temp_source_path):
            source_path = os.path.join(temp_source_path, name)
            target_path = os.path.join(data_path, name)

            shutil.move(source_path, target_path)

        # remove data_path/cifar-10-batches-py
        shutil.rmtree(temp_source_path)

        # remove data_path/cifar-10-python.tar.gz
        os.remove(gzip_path)

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
            data_path = SourceCifar10.default_data_path(dataset)

        SourceCifar10.download(dataset, data_path)
        SourceCifar10.pre_process(dataset, data_path)

        if dataset == datasets.DATASET_CIFAR_10_TRAINING:
            names = ['data_batch_{}'.format(i) for i in range(1, 6)]
        else:
            names = ['test_batch']

        self._labels = []
        self._images = []

        for name in names:
            file_path = os.path.join(data_path, name)

            if not os.path.isfile(file_path):
                raise Exception('can not find {}'.format(file_path))

            with open(file_path, 'rb') as f:
                batch = pickle.load(f)

                images = batch[b'data']
                labels = batch[b'labels']
                labels = numpy.array(labels)
                images = images.reshape(10000, 3, 32, 32)
                images = images.transpose(0, 2, 3, 1)
                images = images.astype(numpy.float32)
                images = images / 127.5 - 1.0

                self._images.append(images)
                self._labels.append(labels)

        self._images = numpy.concatenate(self._images)
        self._labels = numpy.concatenate(self._labels)

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
        return """
            Learning Multiple Layers of Features from Tiny Images,
            Alex Krizhevsky, 2009.
        """

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
