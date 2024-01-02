import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil


@tf.keras.saving.register_keras_serializable()
class BasicBlock(tf.keras.Model):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(BasicBlock, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) // 2
        self._layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_planes, kernel_size=kernel_size, strides=1, padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU()
        ])

    def call(self, x):
        return self._layers(x)

    def get_config(self):
        config = super().get_config()

        config.update({

                'in_planes': self.in_planes,
                'out_planes': self.out_planes,
                'kernel_size': self.kernel_size

            })

        return config 

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.saving.register_keras_serializable()
class GlobalAveragePooling(tf.keras.Model):
    def __init__(self):
        super(GlobalAveragePooling, self).__init__()

    def call(self, feat):
        return tf.reduce_mean(feat, axis=[1, 2])

    def get_config(self):
        config = super().get_config()

        return config 

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class NetworkInNetwork(tf.keras.Model):
    def __init__(self, config, n_classes, *args, **kwargs):
        super(NetworkInNetwork, self).__init__(*args, **kwargs)
        self.config = config
        self.n_classes = n_classes

        num_stages = config.num_stages
        use_avg_on_conv3 = config.use_avg_on_conv3
        #self.num_stages = num_stages
        #self.use_avg_on_conv3 = use_avg_on_conv3

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
            tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        ]))

        # 2nd block
        self.blocks.append(tf.keras.Sequential([
            BasicBlock(nChannels3, nChannels, 5),
            BasicBlock(nChannels, nChannels, 1),
            BasicBlock(nChannels, nChannels, 1),
            tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='same')
        ]))

        # 3rd block
        self.blocks.append(tf.keras.Sequential([
            BasicBlock(nChannels, nChannels, 3),
            BasicBlock(nChannels, nChannels, 1),
            BasicBlock(nChannels, nChannels, 1),
            
        ]))
        
        if use_avg_on_conv3 and num_stages>3:
            avgpool_layer = tf.keras.layers.AvgPool2D(pool_size=3, strides=2, padding='same')
            self.blocks[-1].append(avgpool_layer)

        for s in range(3, num_stages):
            self.blocks.append(tf.keras.Sequential([
                BasicBlock(nChannels, nChannels, 3),
                BasicBlock(nChannels, nChannels, 1),
                BasicBlock(nChannels, nChannels, 1)
            ]))

        # Global average pooling and classifier
        self.blocks.append(tf.keras.Sequential([
            GlobalAveragePooling(),
            tf.keras.layers.Dense(self.n_classes)
        ]))

        self.all_feat_names = ['conv'+str(s+1) for s in range(num_stages)] + ['classifier',]

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

    def call(self, x,training=False, out_feat_keys=None):
        """Forward an image `x` through the network and return the asked output features.

            Args:
              x: input image.
              out_feat_keys: a list/tuple with the feature names of the features
                    that the function should return. By default the last feature of
                    the network is returned.

            Return:
                out_feats: If multiple output features were asked then `out_feats`
                    is a list with the asked output features placed in the same
                    order as in `out_feat_keys`. If a single output feature was
                    asked then `out_feats` is that output feature (and not a list).
        """
        out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
        out_feats = [None] * len(out_feat_keys)
          
          #out_feats = []

        feat = x 
        for f in range(max_out_feat+1):
          feat = self.blocks[f](feat)
          key = self.all_feat_names[f]

          if key in out_feat_keys:
            out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats

        return out_feats

    def get_config(self):
        config = super().get_config()

        config.update({

                'num_classes': self.num_classes,
                'config': self.config,

            })

        return config 

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_model(config, n_classes):
    return NetworkInNetwork(config=config, n_classes=n_classes)


if __name__ == '__main__':
    
    class Config:
        def __init__(self):
            self.num_stages = 5
            self.use_avg_on_conv3 = False

    config = Config()

    model = create_model(config, 4)

    x = tf.random.uniform((10, 224, 224, 3), -1, 1)
    
    out_feats = ['conv1', 'conv2']
    out = model(x, out_feat_keys=None)
    
    if isinstance(out, list):
        for f, feat in enumerate(out):
            print(f'Output feature conv{f+1} - size {feat.shape}')
    
    else:
        print('classifier output size: ', out.shape)
