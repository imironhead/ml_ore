"""
describe all supported datasets.
"""
import os

from download_lsun import download_lsun
from download_mnist import download_mnist
from process_lsun import process_lsun
from source_lsun import SourceLsun
from source_mnist import SourceMnist


__all__ = [
    'DATASET_LSUN',
    'DATASET_LSUN_BEDROOM_TRAINING',
    'DATASET_LSUN_BEDROOM_VALIDATION',
    'DATASET_LSUN_BRIDGE_TRAINING',
    'DATASET_LSUN_BRIDGE_VALIDATION',
    'DATASET_LSUN_CHURCH_OUTDOOR_TRAINING',
    'DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION',
    'DATASET_LSUN_CLASSROOM_TRAINING',
    'DATASET_LSUN_CLASSROOM_VALIDATION',
    'DATASET_LSUN_CONFERENCE_ROOM_TRAINING',
    'DATASET_LSUN_CONFERENCE_ROOM_VALIDATION',
    'DATASET_LSUN_DINING_ROOM_TRAINING',
    'DATASET_LSUN_DINING_ROOM_VALIDATION',
    'DATASET_LSUN_KITCHEN_TRAINING',
    'DATASET_LSUN_KITCHEN_VALIDATION',
    'DATASET_LSUN_LIVING_ROOM_TRAINING',
    'DATASET_LSUN_LIVING_ROOM_VALIDATION',
    'DATASET_LSUN_RESTAURANT_TRAINING',
    'DATASET_LSUN_RESTAURANT_VALIDATION',
    'DATASET_LSUN_TOWER_TRAINING',
    'DATASET_LSUN_TOWER_VALIDATION',
    'DATASET_LSUN_TEST',

    'DATASET_MNIST',
    'DATASET_MNIST_TRAINING',
    'DATASET_MNIST_TEST',


    'LABEL_INVALID',

    'LABEL_LSUN_BEDROOM', 'LABEL_LSUN_BRIDGE', 'LABEL_LSUN_CHURCH_OUTDOOR',
    'LABEL_LSUN_CLASSROOM', 'LABEL_LSUN_CONFERENCE_ROOM',
    'LABEL_LSUN_DINING_ROOM', 'LABEL_LSUN_KITCHEN', 'LABEL_LSUN_LIVING_ROOM',
    'LABEL_LSUN_RESTAURANT', 'LABEL_LSUN_TOWER',

    'LABEL_MNIST_0', 'LABEL_MNIST_1', 'LABEL_MNIST_2', 'LABEL_MNIST_3',
    'LABEL_MNIST_4', 'LABEL_MNIST_5', 'LABEL_MNIST_6', 'LABEL_MNIST_7',
    'LABEL_MNIST_8', 'LABEL_MNIST_9',
    ]


DATASET_LSUN = 0x00010000
DATASET_LSUN_BEDROOM_TRAINING = DATASET_LSUN + 0x00000000
DATASET_LSUN_BEDROOM_VALIDATION = DATASET_LSUN + 0x00000001
DATASET_LSUN_BRIDGE_TRAINING = DATASET_LSUN + 0x00000002
DATASET_LSUN_BRIDGE_VALIDATION = DATASET_LSUN + 0x00000003
DATASET_LSUN_CHURCH_OUTDOOR_TRAINING = DATASET_LSUN + 0x00000004
DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION = DATASET_LSUN + 0x00000005
DATASET_LSUN_CLASSROOM_TRAINING = DATASET_LSUN + 0x00000006
DATASET_LSUN_CLASSROOM_VALIDATION = DATASET_LSUN + 0x00000007
DATASET_LSUN_CONFERENCE_ROOM_TRAINING = DATASET_LSUN + 0x00000008
DATASET_LSUN_CONFERENCE_ROOM_VALIDATION = DATASET_LSUN + 0x00000009
DATASET_LSUN_DINING_ROOM_TRAINING = DATASET_LSUN + 0x0000000a
DATASET_LSUN_DINING_ROOM_VALIDATION = DATASET_LSUN + 0x0000000b
DATASET_LSUN_KITCHEN_TRAINING = DATASET_LSUN + 0x0000000c
DATASET_LSUN_KITCHEN_VALIDATION = DATASET_LSUN + 0x0000000d
DATASET_LSUN_LIVING_ROOM_TRAINING = DATASET_LSUN + 0x0000000e
DATASET_LSUN_LIVING_ROOM_VALIDATION = DATASET_LSUN + 0x0000000f
DATASET_LSUN_RESTAURANT_TRAINING = DATASET_LSUN + 0x00000010
DATASET_LSUN_RESTAURANT_VALIDATION = DATASET_LSUN + 0x00000011
DATASET_LSUN_TOWER_TRAINING = DATASET_LSUN + 0x00000012
DATASET_LSUN_TOWER_VALIDATION = DATASET_LSUN + 0x00000013
DATASET_LSUN_TEST = DATASET_LSUN + 0x00000014

DATASET_MNIST = 0x00020000
DATASET_MNIST_TRAINING = DATASET_MNIST + 0x00000000
DATASET_MNIST_TEST = DATASET_MNIST + 0x00000000

LABEL_INVALID = -1

# https://github.com/fyu/lsun/blob/master/category_indices.txt
LABEL_LSUN_BEDROOM = 0
LABEL_LSUN_BRIDGE = 1
LABEL_LSUN_CHURCH_OUTDOOR = 2
LABEL_LSUN_CLASSROOM = 3
LABEL_LSUN_CONFERENCE_ROOM = 4
LABEL_LSUN_DINING_ROOM = 5
LABEL_LSUN_KITCHEN = 6
LABEL_LSUN_LIVING_ROOM = 7
LABEL_LSUN_RESTAURANT = 8
LABEL_LSUN_TOWER = 9

LABEL_MNIST_0 = 0
LABEL_MNIST_1 = 1
LABEL_MNIST_2 = 2
LABEL_MNIST_3 = 3
LABEL_MNIST_4 = 4
LABEL_MNIST_5 = 5
LABEL_MNIST_6 = 6
LABEL_MNIST_7 = 7
LABEL_MNIST_8 = 8
LABEL_MNIST_9 = 9


def prepare_source(dataset_index, data_path=None):
    """
    data source virtual constructor. data_path can be None to use default path.
    """
    # use default path if data_path is None.
    data_path = default_data_path(dataset_index, data_path)

    # download the dataset if it can not be found on data_path.
    download_source(dataset_index, data_path)

    # pre-process the dataset (e.g. dump all keys from a lmdb).
    process_source(dataset_index, data_path)

    # create the source.
    sources = [(is_lsun, SourceLsun), (is_mnist, SourceMnist)]

    for source in sources:
        if source[0](dataset_index):
            return source[1](dataset_index, data_path)


def download_source(dataset_index, data_path):
    """
    download the dataset if necessary.
    """
    # sanity check
    assert data_path is not None, 'need a data_path'

    downloaders = [(is_lsun, download_lsun), (is_mnist, download_mnist)]

    for downloader in downloaders:
        if downloader[0](dataset_index):
            downloader[1](dataset_index, data_path)


def process_source(dataset_index, data_path):
    """
    process the dataset if necessary.
    """
    processors = [(is_lsun, process_lsun)]

    for processor in processors:
        if processor[0](dataset_index):
            processor[1](dataset_index, data_path)


def default_data_path(dataset_index, data_path=None):
    """
    default data_path is under ~/data/*
    """
    if data_path is None:
        table = {
            DATASET_LSUN_BEDROOM_TRAINING:
                ('data', 'lsun', 'bedroom_train_lmdb'),
            DATASET_LSUN_BEDROOM_VALIDATION:
                ('data', 'lsun', 'bedroom_val_lmdb'),
            DATASET_LSUN_BRIDGE_TRAINING:
                ('data', 'lsun', 'bridge_train_lmdb'),
            DATASET_LSUN_BRIDGE_VALIDATION:
                ('data', 'lsun', 'bridge_val_lmdb'),
            DATASET_LSUN_CHURCH_OUTDOOR_TRAINING:
                ('data', 'lsun', 'church_outdoor_train_lmdb'),
            DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION:
                ('data', 'lsun', 'church_outdoor_val_lmdb'),
            DATASET_LSUN_CLASSROOM_TRAINING:
                ('data', 'lsun', 'classroom_train_lmdb'),
            DATASET_LSUN_CLASSROOM_VALIDATION:
                ('data', 'lsun', 'classroom_val_lmdb'),
            DATASET_LSUN_CONFERENCE_ROOM_TRAINING:
                ('data', 'lsun', 'conference_room_train_lmdb'),
            DATASET_LSUN_CONFERENCE_ROOM_VALIDATION:
                ('data', 'lsun', 'conference_room_val_lmdb'),
            DATASET_LSUN_DINING_ROOM_TRAINING:
                ('data', 'lsun', 'dining_room_train_lmdb'),
            DATASET_LSUN_DINING_ROOM_VALIDATION:
                ('data', 'lsun', 'dining_room_val_lmdb'),
            DATASET_LSUN_KITCHEN_TRAINING:
                ('data', 'lsun', 'kitchen_train_lmdb'),
            DATASET_LSUN_KITCHEN_VALIDATION:
                ('data', 'lsun', 'kitchen_val_lmdb'),
            DATASET_LSUN_LIVING_ROOM_TRAINING:
                ('data', 'lsun', 'living_room_train_lmdb'),
            DATASET_LSUN_LIVING_ROOM_VALIDATION:
                ('data', 'lsun', 'living_room_val_lmdb'),
            DATASET_LSUN_RESTAURANT_TRAINING:
                ('data', 'lsun', 'restaurant_train_lmdb'),
            DATASET_LSUN_RESTAURANT_VALIDATION:
                ('data', 'lsun', 'restaurant_val_lmdb'),
            DATASET_LSUN_TOWER_TRAINING:
                ('data', 'lsun', 'tower_train_lmdb'),
            DATASET_LSUN_TOWER_VALIDATION:
                ('data', 'lsun', 'tower_val_lmdb'),
            DATASET_LSUN_TEST: ('data', 'lsun', 'test_lmdb'),

            DATASET_MNIST_TRAINING: ('data', 'mnist'),
            DATASET_MNIST_TEST: ('data', 'mnist'),
        }

        path_home = os.path.expanduser('~')

        data_path = os.path.join(*table[dataset_index])

        data_path = os.path.join(path_home, data_path)

        if not os.path.isdir(data_path):
            os.makedirs(data_path)

    return data_path


def is_mnist(index):
    """
    """
    return index in mnist_dataset_indice()


def mnist_dataset_indice():
    """
    """
    return [DATASET_MNIST_TRAINING, DATASET_MNIST_TEST]


def is_lsun(index):
    """
    return True if the index is for a lsun dataset.
    """
    return index in lsun_dataset_indice()


def lsun_dataset_indice():
    """
    list of all lsun dataset index.
    """
    return [
        DATASET_LSUN_BEDROOM_TRAINING,
        DATASET_LSUN_BEDROOM_VALIDATION,
        DATASET_LSUN_BRIDGE_TRAINING,
        DATASET_LSUN_BRIDGE_VALIDATION,
        DATASET_LSUN_CHURCH_OUTDOOR_TRAINING,
        DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION,
        DATASET_LSUN_CLASSROOM_TRAINING,
        DATASET_LSUN_CLASSROOM_VALIDATION,
        DATASET_LSUN_CONFERENCE_ROOM_TRAINING,
        DATASET_LSUN_CONFERENCE_ROOM_VALIDATION,
        DATASET_LSUN_DINING_ROOM_TRAINING,
        DATASET_LSUN_DINING_ROOM_VALIDATION,
        DATASET_LSUN_KITCHEN_TRAINING,
        DATASET_LSUN_KITCHEN_VALIDATION,
        DATASET_LSUN_LIVING_ROOM_TRAINING,
        DATASET_LSUN_LIVING_ROOM_VALIDATION,
        DATASET_LSUN_RESTAURANT_TRAINING,
        DATASET_LSUN_RESTAURANT_VALIDATION,
        DATASET_LSUN_TOWER_TRAINING,
        DATASET_LSUN_TOWER_VALIDATION,
        DATASET_LSUN_TEST,
    ]


def lsun_dataset_index_to_label(index):
    """
    https://github.com/fyu/lsun/blob/master/category_indices.txt

    labels for all subset of lsun except the test set. I can not find the
    labels for the test set.
    """
    table = {
        DATASET_LSUN_BEDROOM_TRAINING: LABEL_LSUN_BEDROOM,
        DATASET_LSUN_BEDROOM_VALIDATION: LABEL_LSUN_BEDROOM,
        DATASET_LSUN_BRIDGE_TRAINING: LABEL_LSUN_BRIDGE,
        DATASET_LSUN_BRIDGE_VALIDATION: LABEL_LSUN_BRIDGE,
        DATASET_LSUN_CHURCH_OUTDOOR_TRAINING: LABEL_LSUN_CHURCH_OUTDOOR,
        DATASET_LSUN_CHURCH_OUTDOOR_VALIDATION: LABEL_LSUN_CHURCH_OUTDOOR,
        DATASET_LSUN_CLASSROOM_TRAINING: LABEL_LSUN_CLASSROOM,
        DATASET_LSUN_CLASSROOM_VALIDATION: LABEL_LSUN_CLASSROOM,
        DATASET_LSUN_CONFERENCE_ROOM_TRAINING: LABEL_LSUN_CONFERENCE_ROOM,
        DATASET_LSUN_CONFERENCE_ROOM_VALIDATION: LABEL_LSUN_CONFERENCE_ROOM,
        DATASET_LSUN_DINING_ROOM_TRAINING: LABEL_LSUN_DINING_ROOM,
        DATASET_LSUN_DINING_ROOM_VALIDATION: LABEL_LSUN_DINING_ROOM,
        DATASET_LSUN_KITCHEN_TRAINING: LABEL_LSUN_KITCHEN,
        DATASET_LSUN_KITCHEN_VALIDATION: LABEL_LSUN_KITCHEN,
        DATASET_LSUN_LIVING_ROOM_TRAINING: LABEL_LSUN_LIVING_ROOM,
        DATASET_LSUN_LIVING_ROOM_VALIDATION: LABEL_LSUN_LIVING_ROOM,
        DATASET_LSUN_RESTAURANT_TRAINING: LABEL_LSUN_RESTAURANT,
        DATASET_LSUN_RESTAURANT_VALIDATION: LABEL_LSUN_RESTAURANT,
        DATASET_LSUN_TOWER_TRAINING: LABEL_LSUN_TOWER,
        DATASET_LSUN_TOWER_VALIDATION: LABEL_LSUN_TOWER,
        DATASET_LSUN_TEST: LABEL_INVALID,
    }

    return table[index]
