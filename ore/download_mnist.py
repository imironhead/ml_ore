"""
downloader method for mnist dataset.
"""
import os
import subprocess


def download_mnist(dataset_index, data_path):
    """
    download all 4 *.gz files if necessary.
    """
    # sanity check
    assert data_path is not None, 'data_path should not be None'
    assert os.path.isdir(data_path), 'data_path should be a dir'

    base = 'http://yann.lecun.com/exdb/mnist/'
    names = [
        'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']

    for name in names:
        path_file = os.path.join(data_path, name)

        if os.path.isfile(path_file):
            continue

        url = base + name

        print('downloading {} to {}'.format(name, data_path))

        cmd = ['curl', url, '-o', os.path.join(data_path, name)]

        subprocess.call(cmd)
