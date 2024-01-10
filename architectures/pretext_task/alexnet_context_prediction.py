import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from typing import Union
from common import ConvLayer


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
