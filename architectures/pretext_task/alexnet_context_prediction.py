import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from typing import Union


@tf.keras.saving.register_keras_serializable()
class LRNLayer(tf.keras.layers.Layer):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, **kwargs):
        super(LRNLayer, self).__init__(**kwargs)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        super(LRNLayer, self).__init__(**kwargs)

    def call(self, x):
        return tf.nn.lrn(x, depth_radius=self.local_size, bias=self.beta, alpha=self.alpha)

    def get_config(self):
      config = super().get_config()

      config.update({
            'local_size': self.local_size,
            'alpha': self.alpha,
            'beta': self.beta
        })

      return config 

    @classmethod
    def from_config(cls, config):
      return cls(**config)


@tf.keras.saving.register_keras_serializable()
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
    self.kernel_size = kernel_size
    self.filters = filters 
    self.strides = strides
    self.padding = padding
    self.groups = groups
    self.use_act = use_act
    self.act_type = act_type
    self.use_pooling = use_pooling
    self.pooling_type = pooling_type
    self.pool_size = pool_size
    self.pool_strides = pool_strides
    self.pool_padding = pool_padding

    # conv
    self.conv = tf.keras.layers.Conv2D(kernel_size=kernel_size, filters=filters, 
                                      padding=padding, strides=strides, groups=groups)

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

  def get_config(self):
    config = super().get_config()

    config.update({

            'kernel_size': self.kernel_size,
            'filters': self.filters,
            'padding': self.padding,
            'groups': self.groups,
            'strides': self.strides,
            'use_act': self.use_act,
            'act_type': self.act_type,
            'use_pooling': self.use_pooling,
            'pooling_type': self.pooling_type,
            'pool_size': self.pool_size,
            'pool_strides': self.pool_strides,
            'pool_padding': self.pool_padding
            
      })

    return config 

  @classmethod
  def from_config(cls, config):
    return cls(**config)


@tf.keras.saving.register_keras_serializable()
class LinearLayer(tf.keras.layers.Layer):
  def __init__(self, 
              units: int = 1028, 
              use_act: bool = True, 
              act_type: str = 'relu', 
              use_dropout: bool = True, 
              dropout_rate: float = 0.5, 
              *args, **kwargs):

    super(LinearLayer, self).__init__(*args, **kwargs)
    self.units = units
    self.use_act = use_act
    self.act_type = act_type
    self.use_dropout = use_dropout
    self.dropout_rate = dropout_rate

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

  def get_config(self):
    config = super().get_config()

    config.update({
          'units': self.units,
          'use_act': self.use_act,
          'act_type': self.act_type,
          'use_dropout': self.use_dropout,
          'dropout_rate': self.dropout_rate
      })

    return config 

  @classmethod
  def from_config(cls, config):
    return cls(**config)


class AlexNetContextPrediction(tf.keras.models.Model):
  def __init__(self, config, n_classes=1000, *args, **kwargs):
    super(AlexNetContextPrediction, self).__init__(*args, **kwargs)
    self.config = config 
    self.n_classes = n_classes

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

    self.all_feat_names = ['conv'+str(s+1) for s in range(5)] + ['fc'+str(s+1) for s in range(2)] +['classifier',]

  def _parse_out_keys_arg(self, out_feat_keys):
    # By default return the features of the last layer / module.
    out_feat_keys = [self.all_feat_names[-1], ] if out_feat_keys is None else out_feat_keys

    if len(out_feat_keys) == 0:
      raise ValueError('Empty list of output feature keys.')

    for f, key in enumerate(out_feat_keys):
      if key not in self.all_feat_names:
        raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))

      elif key in out_feat_keys[:f]:
        raise ValueError('Duplicate output feature key: {0}.'.format(key))

    # Find the highest output feature in `out_feat_keys
    max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

    return out_feat_keys, max_out_feat

  def forward_once(self, inputs, out_feat_keys, out_feats):

    x = self.conv_1(inputs)

    #if 'conv1' in out_feat_keys:
     # out_feats[out_feat_keys.index('conv1')] = x

    x = self.lrn_1(x)
    x = self.conv_2(x)

   # if 'conv2' in out_feat_keys:
    #  out_feats[out_feat_keys.index('conv2')] = x

    x = self.lrn_2(x)
    x = self.conv_3(x)

   # if 'conv3' in out_feat_keys:
    #  out_feats[out_feat_keys.index('conv3')] = x

    x = self.conv_4(x)

    #if 'conv4' in out_feat_keys:
     # out_feats[out_feat_keys.index('conv4')] = x

    x = self.conv_5(x)

   # if 'conv5' in out_feat_keys:
    #  out_feats[out_feat_keys.index('conv5')] = x

    x = self.flatten(x)
    x = self.fc_1(x)

    #if 'fc1' in out_feat_keys:
     # out_feats[out_feat_keys.index('fc1')] = x

    return x

  def call(self, inputs, training=False, out_feat_keys=None):
    out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
    out_feats = [None] * len(out_feat_keys)
    
    # B-batch, 2, H-height, W-width, C-channels
    B, T, H, W, C = inputs.shape
    uniform_inputs, random_inputs = tf.unstack(inputs, axis=1) 

    fc1_uniform_feats = self.forward_once(uniform_inputs)
    fc1_random_feats = self.forward_once(random_inputs)

    combined_feats = tf.concat([fc1_uniform_feats, fc1_random_feats], -1)

    x = self.fc_2(combined_feats)

   # if 'fc2' in out_feat_keys:
   #   out_feats[out_feat_keys.index('fc2')] = x

    out = self.out(x)

    #if 'classifer' in out_feat_keys:
     # out_feats[out_feat_keys.index('classifer')] = out

    #out_feats = out_feats[0] if len(out_feats)==1 else out_feats

    return out


  def get_config(self):
    config = super().get_config()

    config.update({

            'n_classes': self.n_classes

      })

    return config 

  @classmethod
  def from_config(cls, config):
    return cls(**config)


def create_model(config, n_classes):
    return AlexNetContextPrediction(config=config, n_classes=n_classes)


if __name__ == '__main__':
    
    class Config:
        def __init__(self):
            self.num_stages = num_stages
            self.use_avg_on_conv3 = use_avg_on_conv3

    config = Config()

    model = create_model(config, 4)

    x = tf.random.uniform((1, size, size, 3), -1, 1)
    out = model(x, None)

    for f, feat in enumerate(out):
        print(f'Output feature conv{f+1} - size {feat.shape}')
