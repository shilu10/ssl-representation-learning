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