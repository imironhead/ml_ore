"""
a class to read lsun data set.
"""
import os
import lmdb
import numpy
import pickle
import scipy.misc
import StringIO

import datasets


class SourceLsun(object):
    """
    """
    @staticmethod
    def default_map_fn(img):
        """
        """
        w, h = img.shape[:2]

        x, y = (w / 2) - 128, (h / 2) - 128

        img = img[x:x + 256, y:y + 256, :]
        img = img.astype(numpy.float32)
        img = img / 127.5 - 1.0

        return scipy.misc.imresize(img, 25)

    def __init__(self, dataset_index, path_lmdb):
        """
        """
        # sanity check
        path_lmdb = '' if path_lmdb is None else path_lmdb
        path_keys = os.path.join(path_lmdb, 'keys.pkl')

        assert os.path.isdir(path_lmdb), '{} is not exist'.format(path_lmdb)

        # keys should have been dumped.
        assert os.path.isfile(path_keys), '{} is not exist'.format(path_keys)

        self._label = datasets.lsun_dataset_index_to_label(dataset_index)

        with open(path_keys, 'r') as kf:
            self._lmdb_keys = pickle.Unpickler(kf).load()

        self._lmdb_path = path_lmdb

    @property
    def cite(self):
        """
        https://github.com/fyu/lsun
        """
        return """
            @article{
                yu15lsun,
                Author = {
                    Yu, Fisher and Zhang, Yinda and Song, Shuran and Seff,
                    Ari and Xiao, Jianxiong
                },
                Title = {
                    LSUN: Construction of a Large-scale Image Dataset using
                    Deep Learning with Humans in the Loop
                },
                Journal = {arXiv preprint arXiv:1506.03365},
                Year = {2015}
            }
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
        return len(self._lmdb_keys)

    def batch(self, idx_list=[], map_fn=default_map_fn.__func__):
        """
        idx_list: list of data indice.
        map_fn: map_fn(source_numpy_array), return target_numpy_array
        """
        cnt = len(idx_list)
        ims = None

        with lmdb.open(self._lmdb_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    for i, j in enumerate(idx_list):
                        if j >= len(self._lmdb_keys):
                            raise Exception('invalid index {}'.format(j))

                        val = cursor.get(self._lmdb_keys[j])
                        sio = StringIO.StringIO(val)
                        img = scipy.misc.imread(sio)
                        img = map_fn(img)

                        if ims is None:
                            ims = numpy.zeros((cnt,) + img.shape)

                        ims[i, :, :, :] = img

        return ims, numpy.repeat(self._label, cnt)
