import tensorflow as tf 
from tensorflow import keras 


import tensorflow as tf


class SmallCNN(tf.keras.Model):
    def __init__(self, 
    			data_format="channels_last", 
    			trainable=True,
    			out_dim=128):

        super(SmallCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=1)
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=1)

        self.activation = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.max_pool(x)

        h = self.global_pool(x)

        return h

# 512 (h) -> 256 -> 128 (z)
class ProjectionHead(tf.keras.Model):

    def __init__(self):
        super(ProjectionHead, self).__init__()
        self.fc1 = tf.keras.layers.Dense(units=256)
        self.bn = tf.keras.layers.BatchNormalization()
        self.fc2 = tf.keras.layers.Dense(units=128)

    def call(self, inp, training=False):
        x = self.fc1(inp)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        x = self.fc2(x)
        return x