# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model definition compatible with TensorFlow's eager execution.

Reference [Deep Residual Learning for Image
Recognition](https://arxiv.org/abs/1512.03385)

Adapted from tf.keras.applications.ResNet50. A notable difference is that the
model here outputs logits while the Keras model outputs probability.
"""
import functools
import tensorflow as tf
from typing import Union


layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
  """_IdentityBlock is the block that has no conv layer at shortcut.

  Args:
    kernel_size: the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    data_format: data_format for the input ('channels_first' or
      'channels_last').
  """

  def __init__(self, kernel_size, filters, stage, block, data_format):
    super(_IdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(
        filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        data_format=data_format,
        name=conv_name_base + '2b')
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv2D(
        filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
  """_ConvBlock is the block that has a conv layer at shortcut.

  Args:
      kernel_size: the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      data_format: data_format for the input ('channels_first' or
        'channels_last').
      strides: strides for the convolution. Note that from stage 3, the first
       conv layer at main path is with strides=(2,2), and the shortcut should
       have strides=(2,2) as well.
  """

  def __init__(self,
               kernel_size,
               filters,
               stage,
               block,
               data_format,
               strides=(2, 2)):
    super(_ConvBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    bn_axis = 1 if data_format == 'channels_first' else 3

    self.conv2a = layers.Conv2D(
        filters1, (1, 1),
        strides=strides,
        name=conv_name_base + '2a',
        data_format=data_format)
    self.bn2a = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2a')

    self.conv2b = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        name=conv_name_base + '2b',
        data_format=data_format)
    self.bn2b = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2b')

    self.conv2c = layers.Conv2D(
        filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
    self.bn2c = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '2c')

    self.conv_shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        name=conv_name_base + '1',
        data_format=data_format)
    self.bn_shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.relu(x)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    shortcut = self.conv_shortcut(input_tensor)
    shortcut = self.bn_shortcut(shortcut, training=training)

    x += shortcut
    return tf.nn.relu(x)


# pylint: disable=not-callable
class ResNet50(tf.keras.Model):
  """Instantiates the ResNet50 architecture.

  Args:
    data_format: format for the image. Either 'channels_first' or
      'channels_last'.  'channels_first' is typically faster on GPUs while
      'channels_last' is typically faster on CPUs. See
      https://www.tensorflow.org/performance/performance_guide#data_formats
    name: Prefix applied to names of variables created in the model.
    trainable: Is the model trainable? If true, performs backward
        and optimization after call() method.
    include_top: whether to include the fully-connected layer at the top of the
      network.
    pooling: Optional pooling mode for feature extraction when `include_top`
      is `False`.
      - `None` means that the output of the model will be the 4D tensor
          output of the last convolutional layer.
      - `avg` means that global average pooling will be applied to the output of
          the last convolutional layer, and thus the output of the model will be
          a 2D tensor.
      - `max` means that global max pooling will be applied.
    block3_strides: whether to add a stride of 2 to block3 to make it compatible
      with tf.slim ResNet implementation.
    average_pooling: whether to do average pooling of block4 features before
      global pooling.
    classes: optional number of classes to classify images into, only to be
      specified if `include_top` is True.

  Raises:
      ValueError: in case of invalid argument for data_format.
  """

  def __init__(self,
               data_format,
               name='',
               trainable=True,
               include_top=True,
               pooling=None,
               block3_strides=False,
               average_pooling=True,
               classes=1000):
    super(ResNet50, self).__init__(name=name)

    valid_channel_values = ('channels_first', 'channels_last')
    if data_format not in valid_channel_values:
      raise ValueError('Unknown data_format: %s. Valid values: %s' %
                       (data_format, valid_channel_values))
    self.include_top = include_top
    self.block3_strides = block3_strides
    self.average_pooling = average_pooling
    self.pooling = pooling

    def conv_block(filters, stage, block, strides=(2, 2)):
      return _ConvBlock(
          3,
          filters,
          stage=stage,
          block=block,
          data_format=data_format,
          strides=strides)

    def id_block(filters, stage, block):
      return _IdentityBlock(
          3, filters, stage=stage, block=block, data_format=data_format)

    self.conv1 = layers.Conv2D(
        64, (7, 7),
        strides=(2, 2),
        data_format=data_format,
        padding='same',
        name='conv1')
    bn_axis = 1 if data_format == 'channels_first' else 3
    self.bn_conv1 = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')
    self.max_pool = layers.MaxPooling2D((3, 3),
                                        strides=(2, 2),
                                        data_format=data_format)

    self.l2a = conv_block([64, 64, 256], stage=2, block='a', strides=(1, 1))
    self.l2b = id_block([64, 64, 256], stage=2, block='b')
    self.l2c = id_block([64, 64, 256], stage=2, block='c')

    self.l3a = conv_block([128, 128, 512], stage=3, block='a')
    self.l3b = id_block([128, 128, 512], stage=3, block='b')
    self.l3c = id_block([128, 128, 512], stage=3, block='c')
    self.l3d = id_block([128, 128, 512], stage=3, block='d')

    self.l4a = conv_block([256, 256, 1024], stage=4, block='a')
    self.l4b = id_block([256, 256, 1024], stage=4, block='b')
    self.l4c = id_block([256, 256, 1024], stage=4, block='c')
    self.l4d = id_block([256, 256, 1024], stage=4, block='d')
    self.l4e = id_block([256, 256, 1024], stage=4, block='e')
    self.l4f = id_block([256, 256, 1024], stage=4, block='f')

    # Striding layer that can be used on top of block3 to produce feature maps
    # with the same resolution as the TF-Slim implementation.
    if self.block3_strides:
      self.subsampling_layer = layers.MaxPooling2D((1, 1),
                                                   strides=(2, 2),
                                                   data_format=data_format)
      self.l5a = conv_block([512, 512, 2048],
                            stage=5,
                            block='a',
                            strides=(1, 1))
    else:
      self.l5a = conv_block([512, 512, 2048], stage=5, block='a')
    self.l5b = id_block([512, 512, 2048], stage=5, block='b')
    self.l5c = id_block([512, 512, 2048], stage=5, block='c')

    self.avg_pool = layers.AveragePooling2D((7, 7),
                                            strides=(7, 7),
                                            data_format=data_format)

    if self.include_top:
      self.flatten = layers.Flatten()
      self.fc1000 = layers.Dense(classes, name='fc1000')
    else:
      reduction_indices = [1, 2] if data_format == 'channels_last' else [2, 3]
      reduction_indices = tf.constant(reduction_indices)
      if pooling == 'avg':
        self.global_pooling = functools.partial(
            tf.reduce_mean,
            axis=reduction_indices,
            keepdims=False)
      elif pooling == 'max':
        self.global_pooling = functools.partial(
            tf.reduce_max, reduction_indices=reduction_indices, keep_dims=False)
      else:
        self.global_pooling = None

  def call(self, inputs, training=True, intermediates_dict=None):
    """Call the ResNet50 model.

    Args:
      inputs: Images to compute features for.
      training: Whether model is in training phase.
      intermediates_dict: `None` or dictionary. If not None, accumulate feature
        maps from intermediate blocks into the dictionary.
        ""

    Returns:
      Tensor with featuremap.
    """

    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = tf.nn.relu(x)
    if intermediates_dict is not None:
      intermediates_dict['block0'] = x

    x = self.max_pool(x)
    if intermediates_dict is not None:
      intermediates_dict['block0mp'] = x

    # Block 1 (equivalent to "conv2" in Resnet paper).
    x = self.l2a(x, training=training)
    x = self.l2b(x, training=training)
    x = self.l2c(x, training=training)
    if intermediates_dict is not None:
      intermediates_dict['block1'] = x

    # Block 2 (equivalent to "conv3" in Resnet paper).
    x = self.l3a(x, training=training)
    x = self.l3b(x, training=training)
    x = self.l3c(x, training=training)
    x = self.l3d(x, training=training)
    if intermediates_dict is not None:
      intermediates_dict['block2'] = x

    # Block 3 (equivalent to "conv4" in Resnet paper).
    x = self.l4a(x, training=training)
    x = self.l4b(x, training=training)
    x = self.l4c(x, training=training)
    x = self.l4d(x, training=training)
    x = self.l4e(x, training=training)
    x = self.l4f(x, training=training)

    if self.block3_strides:
      x = self.subsampling_layer(x)
      if intermediates_dict is not None:
        intermediates_dict['block3'] = x
    else:
      if intermediates_dict is not None:
        intermediates_dict['block3'] = x

    x = self.l5a(x, training=training)
    x = self.l5b(x, training=training)
    x = self.l5c(x, training=training)

    if self.average_pooling:
      x = self.avg_pool(x)
      if intermediates_dict is not None:
        intermediates_dict['block4'] = x
    else:
      if intermediates_dict is not None:
        intermediates_dict['block4'] = x

    if self.include_top:
      return self.fc1000(self.flatten(x))
    elif self.global_pooling:
      return self.global_pooling(x)
    else:
      return x


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

    self.conv_layer_1 = ConvLayer(kernel_size=11, 
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

    self.conv_layer_2 = ConvLayer(kernel_size=(5, 5), 
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

 

    self.linear_layer_1 = LinearLayer(units=1024, 
                                use_act=True, 
                                act_type='relu', 
                                use_dropout=True, 
                                dropout_rate=0.5
                        )

    self.linear_layer_2 = LinearLayer(units=4096, 
                                use_act=True,
                                act_type='relu', 
                                use_dropout=True, 
                                dropout_rate=0.5
                        )

    self.flatten = tf.keras.layers.Flatten()

    self.out = tf.keras.layers.Dense(n_classes)

  def call(self, inputs, training=False):
      # B-batch, T-tile, H-height, W-width, C-channels
      B, T, H, W, C = inputs.shape 

      inputs = tf.transpose(inputs, perm=(1, 0, 2, 3, 4))

      x_list = []
      for i in range(9):
       
        # conv layer and lrn layers
        x = self.conv_layer_1(inputs[i])
        #x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)

        x = self.conv_layer_2(x)
        #x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)

        #x = self.conv_layer_3(x)
        #x = self.conv_layer_4(x)
        #x = self.conv_layer_5(x)

        x = self.flatten(x)
        x = self.linear_layer_1(x)

        x_list.append(x)

      # linear layers
      x = tf.concat(x_list, axis=1)
      x = self.linear_layer_2(x)

      # output layer
      out = self.out(x)

      return out 



class AlexnetV1(tf.keras.models.Model):
  def __init__(self, n_classes):
    super(AlexnetV1, self).__init__()

    self.n_classes = n_classes

    self.conv1 = tf.keras.models.Sequential([])

    self.conv1.add(
      tf.keras.layers.Conv2D(filters=96,
                               kernel_size=(11, 11),
                               strides=4,
                               padding="valid",
                               activation=tf.keras.activations.relu,
                               input_shape=(64, 64, 3))
    )

    self.conv1.add(
      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="valid"),
    )

    self.conv1.add(
      tf.keras.layers.BatchNormalization(),
    )

    self.conv2 = tf.keras.models.Sequential([])

    self.conv2.add(
      tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(5, 5),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
    )

    self.conv2.add(
      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
      
    )

    self.conv2.add(
      tf.keras.layers.BatchNormalization(),
    )

    self.conv3 = tf.keras.models.Sequential([])

    self.conv3.add(
        tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
    )

    self.conv4 = tf.keras.Sequential([])

    self.conv4.add(
      tf.keras.layers.Conv2D(filters=384,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
    )

    self.conv5 = tf.keras.models.Sequential([])

    self.conv5.add(
      tf.keras.layers.Conv2D(filters=256,
                               kernel_size=(3, 3),
                               strides=1,
                               padding="same",
                               activation=tf.keras.activations.relu),
    )

    self.conv5.add(

      tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                  strides=2,
                                  padding="same"),
    )

    self.conv5.add(
      tf.keras.layers.BatchNormalization(),
    )

    self.flatten = tf.keras.layers.Flatten()

    self.fc1 = tf.keras.Sequential([])

    self.fc1.add(
        tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),
    )

    self.fc1.add(
        tf.keras.layers.Dropout(rate=0.2),
    )

    self.fc2 = tf.keras.models.Sequential([])

    self.fc2.add(tf.keras.layers.Dense(units=4096,
                              activation=tf.keras.activations.relu),)
    
    self.fc2.add(tf.keras.layers.Dropout(rate=0.2),)

    self.out = tf.keras.layers.Dense(units=self.n_classes,
                             )


  def call(self, inputs, training=False):
    # B-batch, T-tile, H-height, W-width, C-channels
    B, T, H, W, C = inputs.shape 

    inputs = tf.transpose(inputs, perm=(1, 0, 2, 3, 4))

    print(inputs.shape)

    x_list = []
    for i in range(9):
       
      # conv layer and lrn layers
      x = self.conv1(inputs[i])
      #x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)

      x = self.conv2(x)
      #x = tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.0001, beta=0.75, bias=1.0)

      ##x = self.conv3(x)
      #x = self.conv4(x)
      #x = self.conv5(x)

      x = self.flatten(x)
      x = self.fc1(x)

      x_list.append(x)

    # linear layers
    x = tf.concat(x_list, axis=1)
    x = self.fc2(x)

    # output layer
    out = self.out(x)

    return out 


class LRNLayer(tf.keras.layers.Layer):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, **kwargs):
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        super(LRNLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.local_size, bias=self.beta, alpha=self.alpha)

class Network(tf.keras.Model):
    def __init__(self, classes=1000):
        super(Network, self).__init__()

        self.conv = tf.keras.Sequential([
            layers.Conv2D(96, kernel_size=11, strides=2, padding='valid'),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2),
            LRNLayer(local_size=5, alpha=0.0001, beta=0.75),
            layers.Conv2D(256, kernel_size=5, padding='same', groups=2),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2),
            LRNLayer(local_size=5, alpha=0.0001, beta=0.75),
            layers.Conv2D(384, kernel_size=3, padding='same'),
            layers.ReLU(),
            layers.Conv2D(384, kernel_size=3, padding='same', groups=2),
            layers.ReLU(),
            layers.Conv2D(256, kernel_size=3, padding='same', groups=2),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=3, strides=2)
        ])

        self.fc6 = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(1024),
            layers.ReLU(),
            layers.Dropout(0.5),
        ])

        self.fc7 = tf.keras.Sequential([
            layers.Dense(4096),
            layers.ReLU(),
            layers.Dropout(0.5),
        ])

        self.flatten = tf.keras.layers.Flatten()

        self.classifier = tf.keras.Sequential([
            layers.Dense(classes),
        ])

    def call(self, x, training=False):
      B,T,C,H,W = x.shape

      x = tf.transpose(x, (1, 0, 2, 3, 4))

      x_list = []
      for i in range(9):
        z = self.conv(x[i])
        z = self.flatten(z)
        z = self.fc6(z)

        x_list.append(z)

      x = tf.concat(x_list, axis=1)
      x = self.fc7(x)
      x = self.classifier(x)
      return x


class BasicBlock(tf.keras.Model):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self._layer = tf.keras.Sequential([
            layers.Conv2D(out_planes, kernel_size=kernel_size, strides=1, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

    def call(self, x):
        return self.layer(x)

class GlobalAveragePooling(tf.keras.Model):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def call(self, feat):
        return tf.reduce_mean(feat, axis=[1, 2])

class NetworkInNetwork(tf.keras.Model):
    def __init__(self, num_classes, num_stages, use_avg_on_conv3):
        super(NetworkInNetwork, self).__init__()

      #  num_classes = opt['num_classes']
       # num_inchannels = opt['num_inchannels'] if ('num_inchannels' in opt) else 3
       # num_stages = opt['num_stages'] if ('num_stages' in opt) else 3
       # use_avg_on_conv3 = opt['use_avg_on_conv3'] if ('use_avg_on_conv3' in opt) else True

        assert num_stages >= 3
        nChannels = 192
        nChannels2 = 160
        nChannels3 = 96

        self.blocks = []
        # 1st block
        self.blocks.append(tf.keras.Sequential([
            BasicBlock(3, nChannels, 5),
            BasicBlock(nChannels, nChannels2, 1),
            BasicBlock(nChannels2, nChannels3, 1),
            layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        ]))

        # 2nd block
        self.blocks.append(tf.keras.Sequential([
            BasicBlock(nChannels3, nChannels, 5),
            BasicBlock(nChannels, nChannels, 1),
            BasicBlock(nChannels, nChannels, 1),
            layers.AvgPool2D(pool_size=3, strides=2, padding='same')
        ]))

        # 3rd block
        self.blocks.append(tf.keras.Sequential([
            BasicBlock(nChannels, nChannels, 3),
            BasicBlock(nChannels, nChannels, 1),
            BasicBlock(nChannels, nChannels, 1),
            layers.AvgPool2D(pool_size=3, strides=2, padding='same') if num_stages > 3 and use_avg_on_conv3 else tf.identity # This is only added conditionally in PyTorch code
        ]))

        for s in range(3, num_stages):
            self.blocks.append(tf.keras.Sequential([
                BasicBlock(nChannels, nChannels, 3),
                BasicBlock(nChannels, nChannels, 1),
                BasicBlock(nChannels, nChannels, 1)
            ]))

        # Global average pooling and classifier
        self.blocks.append(tf.keras.Sequential([
            GlobalAveragePooling(),
            layers.Dense(num_classes)
        ]))

        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]

    def call(self, x, out_feat_keys=None):
        out_feats = []

        for f, block in enumerate(self.blocks):
            x = block(x)
            key = f'conv{f+1}' if f < len(self.blocks) - 1 else 'classifier'
            if out_feat_keys is None or key in out_feat_keys:
                out_feats.append(x)

        return out_feats[0] if len(out_feats) == 1 else out_feats

def create_model(num_classes=4, num_stages=5):
    return NetworkInNetwork(num_classes=num_classes, num_stages=num_stages, use_avg_on_conv3=True)

'''
if __name__ == '__main__':
    size = 32
    model = create_model()

    x = tf.random.uniform((1, size, size, 3), -1, 1)
    print(x.shape)
    out = model(x, None) #out_feat_keys=[f'conv{i+1}' for i in range(5)]
    for f, feat in enumerate(out):
        print(f'Output feature conv{f+1} - size {feat.shape}')

    out = model(x)
    print(f'Final output: {out}')
