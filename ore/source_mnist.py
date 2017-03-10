"""
a class to read mnist data set.
"""
import gzip
import os
import numpy

import datasets


class SourceMnist(object):
    """
    a class to track raw mnist data.
    """
    @staticmethod
    def default_map_fn(img):
        """
        remap the image. the default mapping is to do nothing.
        """
        return img

    @staticmethod
    def read32(bytestream):
        """
        read a 32 bit unsigned int fron the bytestream.
        """
        dt = numpy.dtype(numpy.uint32).newbyteorder('>')
        return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]

    @staticmethod
    def extract_images(path_file):
        """
        extract the images into a numpy array in shape [number, y, x, depth].
        """
        print('extracting', path_file)

        with gzip.open(path_file) as bytestream:
            magic = SourceMnist.read32(bytestream)

            if magic != 2051:
                raise Exception('invalid mnist data: {}'.format(path_file))

            size = SourceMnist.read32(bytestream)
            rows = SourceMnist.read32(bytestream)
            cols = SourceMnist.read32(bytestream)

            buff = bytestream.read(size * rows * cols)
            data = numpy.frombuffer(buff, dtype=numpy.uint8)
            data = data.reshape(size, rows, cols, 1)

        return data

    @staticmethod
    def extract_labels(path_file):
        """
        extract the labels into a numpy array with shape [index].
        """
        print('extracting', path_file)

        with gzip.open(path_file) as bytestream:
            magic = SourceMnist.read32(bytestream)

            if magic != 2049:
                raise Exception('invalid mnist data: {}'.format(path_file))

            size = SourceMnist.read32(bytestream)
            buff = bytestream.read(size)
            labels = numpy.frombuffer(buff, dtype=numpy.uint8)

        return labels.astype(numpy.int32)

    def __init__(self, dataset_index, path_dir):
        """
        """
        # sanity check
        assert os.path.isdir(path_dir), '{} is not exist'.format(path_dir)

        if dataset_index == datasets.DATASET_MNIST_TRAINING:
            name = 'train'
        else:
            name = 't10k'

        path_images = os.path.join(
            path_dir, '{}-images-idx3-ubyte.gz'.format(name))
        path_labels = os.path.join(
            path_dir, '{}-labels-idx1-ubyte.gz'.format(name))

        assert os.path.isfile(path_images), 'need {}'.format(path_images)
        assert os.path.isfile(path_labels), 'need {}'.format(path_labels)

        self._labels = SourceMnist.extract_labels(path_labels)
        self._images = SourceMnist.extract_images(path_images)
        self._images = self._images.astype(numpy.float32) / 127.5 - 1.0

    @property
    def cite(self):
        """
        """
        return 'http://yann.lecun.com/exdb/mnist/'

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

    def batch(self, idx_list=[], map_fn=default_map_fn.__func__):
        """
        idx_list: list of data indice.
        map_fn: map_fn(source_numpy_array), return target_numpy_array
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

            ims[i, :, :, :] = img
            idx[i] = self._labels[j]

        return ims, idx
