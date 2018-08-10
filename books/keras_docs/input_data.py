# @Time    : 2018/7/18 15:35
# @Author  : cap
# @FileName: input_data.py
# @Software: PyCharm Community Edition
# @introduction:
import os

import numpy as np
from keras.datasets.cifar import load_batch
from keras.datasets.mnist import load_data
from keras import backend as K


def input_data(flag):
    if flag == 'mnist':
        #
        path = "D:\\softfiles\\workspace\\data\\tensorflow\\data\\mnist_data\\mnist.npz"
        with np.load(path) as f:
            x_train, y_train = f['x_train'], f['y_train']
            x_test, y_test = f['x_test'], f['y_test']
        return (x_train, y_train), (x_test, y_test)
    elif flag == 'cifar10':
        #'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        path = "D:\\softfiles\\workspace\\data\\tensorflow\\data\\cifar10\\cifar-10-batches-py"
        num_train_samples = 50000

        x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
        y_train = np.empty((num_train_samples,), dtype='uint8')

        for i in range(1, 6):
            fpath = os.path.join(path, 'data_batch_' + str(i))
            (x_train[(i - 1) * 10000: i * 10000, :, :, :],
             y_train[(i - 1) * 10000: i * 10000]) = load_batch(fpath)

        fpath = os.path.join(path, 'test_batch')
        x_test, y_test = load_batch(fpath)

        y_train = np.reshape(y_train, (len(y_train), 1))
        y_test = np.reshape(y_test, (len(y_test), 1))

        if K.image_data_format() == 'channels_last':
            x_train = x_train.transpose(0, 2, 3, 1)
            x_test = x_test.transpose(0, 2, 3, 1)

        return (x_train, y_train), (x_test, y_test)