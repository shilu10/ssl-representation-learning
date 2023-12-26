import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 


class LRNLayer(tf.keras.layers.Layer):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, **kwargs):
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        super(LRNLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.local_size, bias=self.beta, alpha=self.alpha)

    def from_config(self)


class ConvLayer(tf.keras.layers.Layer):
  def __init__(self, 
              kernel_size: tuple = (3, 3), 
              filters: int = 10, 
              strides: Union[tuple, None] = (1, 1), 
              padding: str = "valid",
              groups=1,
              use_act: bool = True, 
              act_type: str = "relu", 
              use_pooling: bool = True,
              pooling_type: str = "max",
              pool_size: tuple = (2, 2),
              pool_strides: Union[tuple, None] = None,
              pool_padding: str= "valid",
              *args, 
              **kwargs):

    super(ConvLayer, self).__init__(*args, **kwargs)

    # conv
    self.conv = tf.keras.layers.Conv2D(kernel_size=kernel_size, 
                                  filters=filters, padding=padding, strides=strides, groups=groups)

    # activation
    if use_act:
      self.activation = tf.keras.layers.Activation(act_type)

    else:
      self.activation = tf.identity

    # pooling
    if use_pooling:
      if pooling_type == "max":
        self.pooling = tf.keras.layers.MaxPooling2D(pool_size=pool_size, 
                                              padding=pool_padding, strides=pool_strides)

      elif pooling_type == "avg":
        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=pool_size, 
                                              padding=pool_padding, strides=pool_strides)
        
    else:
      self.pooling = tf.identity

  def call(self, inputs):
    x = self.conv(inputs)
    x = self.activation(x)
    x = self.pooling(x)

    return x 


class LinearLayer(tf.keras.layers.Layer):
  def __init__(self, 
              units: int = 1028, 
              use_act: bool = True, 
              act_type: str = 'relu', 
              use_dropout: bool = True, 
              dropout_rate: float = 0.5, 
              *args, **kwargs):

    super(LinearLayer, self).__init__(*args, **kwargs)

    self.linear = tf.keras.layers.Dense(units=units)

    # activaton
    if use_act:
      self.activation = tf.keras.layers.Activation(act_type)

    else:
      self.activation = tf.identity

    # dropout
    if use_dropout:
      self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    else:
      self.dropout = tf.identity


  def call(self, inputs: tf.Tensor):
    x = self.linear(inputs)

    x = self.activation(x)
    x = self.dropout(x)

    return x 


class AlexNet(tf.keras.models.Model):
  def __init__(self, n_classes=1000, *args, **kwargs):
    super(AlexNet, self).__init__(*args, **kwargs)

    self.conv_1 = ConvLayer(kernel_size=11, 
                              strides=2, 
                              filters=96,
                              padding="valid", 
                              use_act=True, 
                              act_type='relu', 
                              use_pooling=True, 
                              pooling_type='max', 
                              pool_size=(3, 3), 
                              pool_strides=(2, 2)
                        )

    self.conv_2 = ConvLayer(kernel_size=(5, 5), 
                              groups=2,
                              filters=256,
                              padding='same', 
                              use_act=True, 
                              act_type='relu', 
                              use_pooling=True, 
                              pooling_type='max', 
                              pool_size=(3, 3), 
                              pool_strides=(2, 2)
                        )
 
    self.conv_3 = ConvLayer(kernel_size=(3, 3), 
                              filters=384,
                              padding='same', 
                              use_act=True, 
                              act_type='relu', 
                              use_pooling=False, 
                        )

    self.conv_4 = ConvLayer(kernel_size=(3, 3), 
                              groups=2,
                              filters=384,
                              padding='same', 
                              use_act=True, 
                              act_type='relu', 
                              use_pooling=False, 
                        )

    self.conv_5 = ConvLayer(kernel_size=(3, 3), 
                              groups=2,
                              filters=256,
                              padding='same', 
                              use_act=True, 
                              act_type='relu', 
                              use_pooling=True, 
                              pooling_type='max', 
                              pool_size=(3, 3), 
                              pool_strides=(2, 2)
                        )

    self.fc_1 = LinearLayer(units=1024, 
                              use_act=True, 
                              act_type='relu', 
                              use_dropout=True, 
                              dropout_rate=0.5
                        )

    self.fc_2 = LinearLayer(units=4096, 
                              use_act=True,
                              act_type='relu', 
                              use_dropout=True, 
                              dropout_rate=0.5
                        )

    self.lrn_1 = LRNLayer(local_size=5, alpha=0.0001, beta=0.75)
    self.lrn_2 = LRNLayer(local_size=5, alpha=0.0001, beta=0.75)

    self.flatten = tf.keras.layers.Flatten()

    self.out = tf.keras.layers.Dense(n_classes)

  def call(self, inputs, training=False):
      # B-batch, T-tile, H-height, W-width, C-channels
      B, T, H, W, C = inputs.shape 
      inputs = tf.transpose(inputs, perm=(1, 0, 2, 3, 4))

      x_list = []
      for i in range(9):
        # conv layer and lrn layers
        x = self.conv_1(inputs[i])
        x = self.lrn_1(x)

        x = self.conv_1(x)
        x = self.lrn_2(x)

        x = self.conv_1(x)
        x = self.conv_1(x)
        x = self.conv_1(x)

        x = self.flatten(x)
        x = self.fc_1(x)

        x_list.append(x)

      # linear layers
      x = tf.concat(x_list, axis=1)
      x = self.fc_1(x)

      # output layer
      out = self.out(x)

      return out 