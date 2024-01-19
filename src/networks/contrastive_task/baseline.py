import tensorflow as tf 
from tensorflow import keras 


def simple_cnn(input_shape, width):
  encoder = tf.keras.Sequential(
          [
            tf.keras.layers.Input(shape=(input_shape, input_shape, 3)),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Conv2D(width, kernel_size=3, strides=2, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(width, activation="relu"),
          ],
          name="encoder",
       )

  return encoder


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