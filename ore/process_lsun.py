"""
to dump all keys of a lsun dataset. to build the key table is very slow.
"""
import lmdb
import os
import pickle


def process_lsun(dataset_index, data_path):
    """
    """
    path_keys = os.path.join(data_path, 'keys.pkl')

    if os.path.isfile(path_keys):
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

    with open(path_keys, 'w') as kf:
        pickle.Pickler(kf).dump(keys)
