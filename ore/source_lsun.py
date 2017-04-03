"""
a class to read lsun data set.
"""
import os
import lmdb
import numpy
import pickle
import scipy.misc
import shutil
import StringIO
import zipfile

from six.moves import urllib

import datasets


class SourceLsun(object):
    """
    """
    @staticmethod
    def default_data_path(dataset):
        """
        """
        table = {
            datasets.DATASET_LSUN_BEDROOM_TRAINING: 'bedroom_train_lmdb',
            datasets.DATASET_LSUN_BEDROOM_VALIDATION: 'bedroom_val_lmdb',
            datasets.DATASET_LSUN_BRIDGE_TRAINING: 'bridge_train_lmdb',
            datasets.DATASET_LSUN_BRIDGE_VALIDATION: 'bridge_val_lmdb',
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_TRAINING:
                'church_outdoor_train_lmdb',
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION:
                'church_outdoor_val_lmdb',
            datasets.DATASET_LSUN_CLASSROOM_TRAINING: 'classroom_train_lmdb',
            datasets.DATASET_LSUN_CLASSROOM_VALIDATION: 'classroom_val_lmdb',
            datasets.DATASET_LSUN_CONFERENCE_ROOM_TRAINING:
                'conference_room_train_lmdb',
            datasets.DATASET_LSUN_CONFERENCE_ROOM_VALIDATION:
                'conference_room_val_lmdb',
            datasets.DATASET_LSUN_DINING_ROOM_TRAINING:
                'dining_room_train_lmdb',
            datasets.DATASET_LSUN_DINING_ROOM_VALIDATION:
                'dining_room_val_lmdb',
            datasets.DATASET_LSUN_KITCHEN_TRAINING: 'kitchen_train_lmdb',
            datasets.DATASET_LSUN_KITCHEN_VALIDATION: 'kitchen_val_lmdb',
            datasets.DATASET_LSUN_LIVING_ROOM_TRAINING:
                'living_room_train_lmdb',
            datasets.DATASET_LSUN_LIVING_ROOM_VALIDATION:
                'living_room_val_lmdb',
            datasets.DATASET_LSUN_RESTAURANT_TRAINING:
                'restaurant_train_lmdb',
            datasets.DATASET_LSUN_RESTAURANT_VALIDATION:
                'restaurant_val_lmdb',
            datasets.DATASET_LSUN_TOWER_TRAINING: 'tower_train_lmdb',
            datasets.DATASET_LSUN_TOWER_VALIDATION: 'tower_val_lmdb',
            datasets.DATASET_LSUN_TEST: 'test_lmdb',
        }

        path_home = os.path.expanduser('~')

        return os.path.join(path_home, 'data', 'lsun', table[dataset])

    @staticmethod
    def subsets():
        """
        """
        return [
            datasets.DATASET_LSUN_BEDROOM_TRAINING,
            datasets.DATASET_LSUN_BEDROOM_VALIDATION,
            datasets.DATASET_LSUN_BRIDGE_TRAINING,
            datasets.DATASET_LSUN_BRIDGE_VALIDATION,
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_TRAINING,
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION,
            datasets.DATASET_LSUN_CLASSROOM_TRAINING,
            datasets.DATASET_LSUN_CLASSROOM_VALIDATION,
            datasets.DATASET_LSUN_CONFERENCE_ROOM_TRAINING,
            datasets.DATASET_LSUN_CONFERENCE_ROOM_VALIDATION,
            datasets.DATASET_LSUN_DINING_ROOM_TRAINING,
            datasets.DATASET_LSUN_DINING_ROOM_VALIDATION,
            datasets.DATASET_LSUN_KITCHEN_TRAINING,
            datasets.DATASET_LSUN_KITCHEN_VALIDATION,
            datasets.DATASET_LSUN_LIVING_ROOM_TRAINING,
            datasets.DATASET_LSUN_LIVING_ROOM_VALIDATION,
            datasets.DATASET_LSUN_RESTAURANT_TRAINING,
            datasets.DATASET_LSUN_RESTAURANT_VALIDATION,
            datasets.DATASET_LSUN_TOWER_TRAINING,
            datasets.DATASET_LSUN_TOWER_VALIDATION,
            datasets.DATASET_LSUN_TEST,
        ]

    @staticmethod
    def include(dataset):
        """
        """
        return dataset in SourceLsun.subsets()

    @staticmethod
    def download(dataset, data_path):
        """
        """
        if data_path is None:
            data_path = SourceLsun.default_data_path(dataset)

        # ~/data/lsun/tower_val_lmdb/data.mdb
        # ~/data/lsun/tower_val_lmdb/lock.mdb
        path_data = os.path.join(data_path, 'data.mdb')
        path_lock = os.path.join(data_path, 'lock.mdb')

        if os.path.isfile(path_data) and os.path.isfile(path_lock):
            # downloaded
            return

        if not os.path.isdir(data_path):
            os.makedirs(data_path)

        # download
        table = {
            datasets.DATASET_LSUN_BEDROOM_TRAINING: ('bedroom', 'train'),
            datasets.DATASET_LSUN_BEDROOM_VALIDATION: ('bedroom', 'val'),
            datasets.DATASET_LSUN_BRIDGE_TRAINING: ('bridge', 'train'),
            datasets.DATASET_LSUN_BRIDGE_VALIDATION: ('bridge', 'val'),
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_TRAINING:
                ('church_outdoor', 'train'),
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION:
                ('church_outdoor', 'val'),
            datasets.DATASET_LSUN_CLASSROOM_TRAINING: ('classroom', 'train'),
            datasets.DATASET_LSUN_CLASSROOM_VALIDATION: ('classroom', 'val'),
            datasets.DATASET_LSUN_CONFERENCE_ROOM_TRAINING:
                ('conference_room', 'train'),
            datasets.DATASET_LSUN_CONFERENCE_ROOM_VALIDATION:
                ('conference_room', 'val'),
            datasets.DATASET_LSUN_DINING_ROOM_TRAINING:
                ('dining_room', 'train'),
            datasets.DATASET_LSUN_DINING_ROOM_VALIDATION:
                ('dining_room', 'val'),
            datasets.DATASET_LSUN_KITCHEN_TRAINING: ('kitchen', 'train'),
            datasets.DATASET_LSUN_KITCHEN_VALIDATION: ('kitchen', 'val'),
            datasets.DATASET_LSUN_LIVING_ROOM_TRAINING:
                ('living_room', 'train'),
            datasets.DATASET_LSUN_LIVING_ROOM_VALIDATION:
                ('living_room', 'val'),
            datasets.DATASET_LSUN_RESTAURANT_TRAINING: ('restaurant', 'train'),
            datasets.DATASET_LSUN_RESTAURANT_VALIDATION: ('restaurant', 'val'),
            datasets.DATASET_LSUN_TOWER_TRAINING: ('tower', 'train'),
            datasets.DATASET_LSUN_TOWER_VALIDATION: ('tower', 'val'),
            datasets.DATASET_LSUN_TEST: ('', 'test'),
        }

        names = table[dataset]

        temp_path = os.path.join(data_path, '_.zip')

        if not os.path.isfile(temp_path):
            url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?' \
                  'tag=latest&category={}&set={}'.format(*names)

            print('downloading {} to {}'.format(url, temp_path))

            urllib.request.urlretrieve(url, temp_path)

        # unzip
        zipfile.ZipFile(temp_path, 'r').extractall(data_path)

        # move
        name_lmdb = '_'.join([n for n in names if len(n) > 0]) + '_lmdb'
        path_mdbs = os.path.join(data_path, name_lmdb)

        # os.system('mv {} {}'.format(path_mdbs, data_path))
        for name in os.listdir(path_mdbs):
            source_path = os.path.join(path_mdbs, name)
            target_path = os.path.join(data_path, name)

            shutil.move(source_path, target_path)

        # cleanup
        shutil.rmtree(path_mdbs)
        os.remove(temp_path)

    @staticmethod
    def pre_process(dataset, data_path):
        """
        """
        keys_path = os.path.join(data_path, 'keys.pkl')

        if os.path.isfile(keys_path):
            return

        print('generating keys of lmdb: ' + data_path)

        keys = []

        with lmdb.open(data_path) as env:
            with env.begin(write=False) as txn:
                with txn.cursor() as cursor:
                    keys_iter = cursor.iternext_nodup(keys=True, values=False)

                    keys_count = env.stat()['entries']

                    for idx, key in enumerate(keys_iter):
                        keys.append(key)

                        if idx % 1000 == 0:
                            print 'found keys: {} / {}'.format(idx, keys_count)

        with open(keys_path, 'w') as kf:
            pickle.Pickler(kf).dump(keys)

    @staticmethod
    def default_map_fn(img):
        """
        """
        w, h = img.shape[:2]

        x, y = (w / 2) - 128, (h / 2) - 128

        img = img[x:x + 256, y:y + 256, :]

        img = scipy.misc.imresize(img, 25)

        # XXX: scipy.misc.imresize always return 0 ~ 255 ???
        return img / 127.5 - 1.0

    @staticmethod
    def dataset_to_label(dataset):
        """
        https://github.com/fyu/lsun/blob/master/category_indices.txt

        labels for all subset of lsun except the test set. I can not find the
        labels for the test set.
        """
        table = {
            datasets.DATASET_LSUN_BEDROOM_TRAINING:
                datasets.LABEL_LSUN_BEDROOM,
            datasets.DATASET_LSUN_BEDROOM_VALIDATION:
                datasets.LABEL_LSUN_BEDROOM,
            datasets.DATASET_LSUN_BRIDGE_TRAINING:
                datasets.LABEL_LSUN_BRIDGE,
            datasets.DATASET_LSUN_BRIDGE_VALIDATION:
                datasets.LABEL_LSUN_BRIDGE,
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_TRAINING:
                datasets.LABEL_LSUN_CHURCH_OUTDOOR,
            datasets.DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION:
                datasets.LABEL_LSUN_CHURCH_OUTDOOR,
            datasets.DATASET_LSUN_CLASSROOM_TRAINING:
                datasets.LABEL_LSUN_CLASSROOM,
            datasets.DATASET_LSUN_CLASSROOM_VALIDATION:
                datasets.LABEL_LSUN_CLASSROOM,
            datasets.DATASET_LSUN_CONFERENCE_ROOM_TRAINING:
                datasets.LABEL_LSUN_CONFERENCE_ROOM,
            datasets.DATASET_LSUN_CONFERENCE_ROOM_VALIDATION:
                datasets.LABEL_LSUN_CONFERENCE_ROOM,
            datasets.DATASET_LSUN_DINING_ROOM_TRAINING:
                datasets.LABEL_LSUN_DINING_ROOM,
            datasets.DATASET_LSUN_DINING_ROOM_VALIDATION:
                datasets.LABEL_LSUN_DINING_ROOM,
            datasets.DATASET_LSUN_KITCHEN_TRAINING:
                datasets.LABEL_LSUN_KITCHEN,
            datasets.DATASET_LSUN_KITCHEN_VALIDATION:
                datasets.LABEL_LSUN_KITCHEN,
            datasets.DATASET_LSUN_LIVING_ROOM_TRAINING:
                datasets.LABEL_LSUN_LIVING_ROOM,
            datasets.DATASET_LSUN_LIVING_ROOM_VALIDATION:
                datasets.LABEL_LSUN_LIVING_ROOM,
            datasets.DATASET_LSUN_RESTAURANT_TRAINING:
                datasets.LABEL_LSUN_RESTAURANT,
            datasets.DATASET_LSUN_RESTAURANT_VALIDATION:
                datasets.LABEL_LSUN_RESTAURANT,
            datasets.DATASET_LSUN_TOWER_TRAINING:
                datasets.LABEL_LSUN_TOWER,
            datasets.DATASET_LSUN_TOWER_VALIDATION:
                datasets.LABEL_LSUN_TOWER,
            datasets.DATASET_LSUN_TEST:
                datasets.LABEL_INVALID,
        }

        return table[dataset]

    def __init__(self, dataset, range_percentage=(0, 100), data_path=None):
        """
        """
        if data_path is None:
            data_path = SourceLsun.default_data_path(dataset)

        SourceLsun.download(dataset, data_path)
        SourceLsun.pre_process(dataset, data_path)

        # sanity check
        path_keys = os.path.join(data_path, 'keys.pkl')

        # keys should have been dumped.
        assert os.path.isfile(path_keys), '{} is not exist'.format(path_keys)

        self._label = SourceLsun.dataset_to_label(dataset)

        with open(path_keys, 'r') as kf:
            self._lmdb_keys = pickle.Unpickler(kf).load()

        self._lmdb_path = data_path

        # NOTE: range must be dealt within each source due to the layout of
        #       sources may be different.
        head, tail = range_percentage
        size = len(self._lmdb_keys)
        head = head * size // 100
        tail = tail * size // 100

        if head >= tail:
            raise Exception('the range is too narrow')

        self._lmdb_keys = self._lmdb_keys[head:tail]

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

    def batch(self, idx_list=[], map_fn=default_map_fn.__func__, **options):
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
                        img = img.astype(numpy.float32)
                        img = img / 127.5 - 1.0
                        img = map_fn(img)

                        if ims is None:
                            ims = numpy.zeros((cnt,) + img.shape)

                        ims[i, :, :, :] = img

        return ims, numpy.repeat(self._label, cnt)
