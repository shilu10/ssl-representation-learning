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
        self.fc1 = tf.keras.Sequential(
                                tf.keras.layers.Dense(encoding_size, bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leakyrelu')
                                )

        self.fc2 = tf.keras.Sequential(
                                tf.keras.layers.Dense(encoding_size, bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leakyrelu')
                                )
    
    def call(self, x):
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
    def __init__(self, input_size, encoding_size):
        super(GenericTask, self).__init__()
        self.pool = tfa.layers.AdaptiveAveragePooling2D((1, 1))
        self.fc = tf.keras.Sequential(
                                tf.keras.layers.Dense(encoding_size, bias=False),
                                tf.keras.layers.BatchNormalization(),
                                tf.keras.layers.Activation('leakyrelu')
                                )

    def call(self, x):
        # Input size : [B, Channels, Height, Width]
        x = self.pool(x)
        x = tf.squeeze(tf.squeeze(x, axis=-2), axis=-2)
        # Input size : [B, Channels]
        x = self.fc(x)
        # Input size : [B, Encoding_size]
        return x