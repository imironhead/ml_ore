"""
"""
import numpy
import os
import pickle

from builtins import range

import datasets


class SourceCifar10(object):
    """
    """
    @staticmethod
    def default_map_fn(img):
        """
        remap the image. the default mapping is to do nothing.
        """
        return img

    def __init__(self, dataset_index, path_dir):
        """
        """
        assert os.path.isdir(path_dir), '{} is not exist'.format(path_dir)

        if dataset_index == datasets.DATASET_CIFAR_10_TRAINING:
            names = ['data_batch_{}'.format(i) for i in range(1, 6)]
        else:
            names = ['test_batch']

        self._labels = []
        self._images = []

        for name in names:
            file_path = os.path.join(path_dir, name)

            if not os.path.isfile(file_path):
                raise Exception('can not find {}'.format(file_path))

            with open(file_path, 'rb') as f:
                batch = pickle.load(f)

                images = batch[b'data']
                labels = batch[b'labels']
                images = images.reshape(10000, 3, 32, 32) \
                    .transpose(0, 2, 3, 1).astype(numpy.float32)
                labels = numpy.array(labels)

                self._images.append(images)
                self._labels.append(labels)

        self._images = numpy.concatenate(self._images)
        self._labels = numpy.concatenate(self._labels)

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
              one_hot=False):
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
