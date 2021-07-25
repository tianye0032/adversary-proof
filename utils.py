import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from easydict import EasyDict
from keras.datasets import cifar10
import numpy as np

def ld_cifar10(subtract_pixel_mean=False, num_classes=10):
    # Load the CIFAR10 data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    # print('y_train shape:', y_train.shape)
    y_train= y_train.astype(int)
    y_test = y_test.astype(int)
    # Convert class vectors to binary class matrices.
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    return EasyDict(train=(x_train, y_train), test=(x_test, y_test))