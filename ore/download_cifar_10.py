"""
"""
import os
import subprocess

from builtins import range


def download_cifar_10(dataset_index, data_path):
    """
    https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    """
    all_there = True

    for i in range(1, 6):
        file_name = 'data_batch_{}'.format(i)
        file_path = os.path.join(data_path, file_name)

        if not os.path.isfile(file_path):
            all_there = False
            break

    if not os.path.isfile(os.path.join(data_path, 'test_batch')):
        all_there = False

    if all_there:
        # downloaded, unzipped, moved
        return

    path_temp = os.path.join(data_path, 'cifar-10-python.tar.gz')

    # download
    print('downloading {}'.format(path_temp))

    cmd = ['curl',
           'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
           '-o',
           path_temp]

    subprocess.call(cmd)

    # unzip, move
    path_source_dir = os.path.join(data_path, 'cifar-10-batches-py')

    subprocess.call(['unzip', path_temp, '-d', data_path])
    subprocess.call(['mv', os.path.join(path_source_dir, '*'), data_path])

    # cleanup
    subprocess.call(['rm', path_source_dir])
    subprocess.call(['rm', path_temp])
