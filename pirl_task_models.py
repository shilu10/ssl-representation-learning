import tensorflow as tf 
from tensorflow import keras 
import tensorflow_addons as tfa


class JigsawTask(tf.keras.models.Model):
    def __init__(self, encoding_size, jigsaw_size):
        super(JigsawTask, self).__init__()
        if isinstance(jigsaw_size, int):
            self.jigsaw_size = jigsaw_size**2

        elif isinstance(jigsaw_size, (tuple, list)):
            assert len(jigsaw_size) == 2
            self.jigsaw_size = jigsaw_size[0]*jigsaw_size[1]

        self.encoding_size = encoding_size
        self.pool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.fc1 = tf.keras.Sequential([
                                tf.keras.layers.Dense(encoding_size, use_bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leaky_relu')
        ])

        self.fc2 = tf.keras.Sequential([
                                tf.keras.layers.Dense(encoding_size, use_bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leaky_relu')
        ])
    
    def call(self, x, training=False):
        B = x.shape[0]
        # Input size : [B*Jigsaw_size, Height, Width, Channels]
        x = self.pool(x)
        x = tf.squeeze(tf.squeeze(x, axis=-2), axis=-2)
        # Input size : [B*Jigsaw_size, Channels]
        x = self.fc1(x)
        # Input size : [B*Jigsaw_size, Encoding_size]
        x = tf.reshape(x, (-1, self.jigsaw_size*self.encoding_size))
        #x = x.view(-1, self.jigsaw_size*self.encoding_size)
        # Input size : [B, Jigsaw_size*Encoding_size]
        x = self.fc2(x)
        # Input size : [B, Encoding_size]
        return x


class GenericTask(tf.keras.models.Model):
    def __init__(self, encoding_size):
        super(GenericTask, self).__init__()
        self.pool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.fc = tf.keras.Sequential([
                                tf.keras.layers.Dense(encoding_size, use_bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leaky_relu')
        ])

    def call(self, x, training=False):
        # Input size : [B, Channels, Height, Width]
        x = self.pool(x)
        x = tf.squeeze(tf.squeeze(x, axis=-2), axis=-2)
        # Input size : [B, Channels]
        x = self.fc(x)
        # Input size : [B, Encoding_size]
        return x


class CNN(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                                            input_shape=input_shape, kernel_initializer='glorot_uniform')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(strides=2, pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1,
                                             padding='same', kernel_initializer='glorot_uniform')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, 
                                            padding='same', kernel_initializer='glorot_uniform')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, 
                                            padding='same', kernel_initializer='glorot_uniform')
        self.bn4 = tf.keras.layers.BatchNormalization()


    # @timeit
    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        return x