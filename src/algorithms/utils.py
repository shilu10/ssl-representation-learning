import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.layers import Dense, Conv2D

def _conv2d(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Conv2D(*args, **kwargs)
    return _func


def _dense(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return Dense(*args, **kwargs)
    return _func
