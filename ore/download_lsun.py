"""
downloader method for lsun datasets.
"""
import datasets
import os
import subprocess


def download_lsun(dataset_index, data_path):
    """
    """
    # sanity check
    assert data_path is not None, 'data_path should not be None'

    path_data = os.path.join(data_path, 'data.mdb')
    path_lock = os.path.join(data_path, 'lock.mdb')

    if os.path.isfile(path_data) and os.path.isfile(path_lock):
        # downloaded
        return

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
        datasets.DATASET_LSUN_DINING_ROOM_TRAINING: ('dining_room', 'train'),
        datasets.DATASET_LSUN_DINING_ROOM_VALIDATION: ('dining_room', 'val'),
        datasets.DATASET_LSUN_KITCHEN_TRAINING: ('kitchen', 'train'),
        datasets.DATASET_LSUN_KITCHEN_VALIDATION: ('kitchen', 'val'),
        datasets.DATASET_LSUN_LIVING_ROOM_TRAINING: ('living_room', 'train'),
        datasets.DATASET_LSUN_LIVING_ROOM_VALIDATION: ('living_room', 'val'),
        datasets.DATASET_LSUN_RESTAURANT_TRAINING: ('restaurant', 'train'),
        datasets.DATASET_LSUN_RESTAURANT_VALIDATION: ('restaurant', 'val'),
        datasets.DATASET_LSUN_TOWER_TRAINING: ('tower', 'train'),
        datasets.DATASET_LSUN_TOWER_VALIDATION: ('tower', 'val'),
        datasets.DATASET_LSUN_TEST: ('', 'test'),
    }

    names = table[dataset_index]

    path_temp = os.path.join(data_path, '_.zip')

    if not os.path.isfile(path_temp):
        url = 'http://lsun.cs.princeton.edu/htbin/download.cgi?' \
            'tag=latest&category={}&set={}'.format(*names)

        print('downloading {} to {}'.format(' '.join(names), path_temp))

        cmd = ['curl', url, '-o', path_temp]

        subprocess.call(cmd)

    # unzip
    subprocess.call(['unzip', path_temp, '-d', data_path])

    # move
    name_dir = '_'.join([n for n in names if len(n) > 0]) + '_lmdb'
    path_dir = os.path.join(data_path, name_dir)
    path_data_temp = os.path.join(path_dir, 'data.mdb')
    path_lock_temp = os.path.join(path_dir, 'lock.mdb')

    subprocess.call(['mv', path_data_temp, path_data])
    subprocess.call(['mv', path_lock_temp, path_lock])

    # cleanup
    subprocess.call(['rm', '-r', path_dir])
    subprocess.call(['rm', path_temp])
