import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil
from typing import * 


@tf.keras.saving.register_keras_serializable()
class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, out_dims: int, kernel_size: int, strides: int, padding: str, *args, **kwargs):
        super(BasicBlock, self).__init__(*args, **kwargs)
        self.out_dims = out_dims
        self.kernel_size = kernel_size 
        self.strides = strides 
        self.padding = padding 
        
        self.conv = tf.keras.layers.Conv2D(kernel_size=kernel_size, 
                                           filters=out_dims, 
                                           strides=strides, 
                                           padding=padding)
        
        self.norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('leaky_relu')
        
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.norm(x)
        x = self.activation(x)
        
        return x 
    
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'out_dims': self.out_dims,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
@tf.keras.saving.register_keras_serializable()   
class TransposeBlock(tf.keras.layers.Layer):
    def __init__(self, out_dims: int, kernel_size: int, strides: int, padding: str, *args, **kwargs):
        super(TransposeBlock, self).__init__(*args, **kwargs)
        self.out_dims = out_dims
        self.kernel_size = kernel_size 
        self.strides = strides 
        self.padding = padding 
        
        self.transpose_conv = tf.keras.layers.Conv2DTranspose(kernel_size=kernel_size, 
                                                             filters=out_dims, 
                                                             strides=strides, 
                                                             padding=padding)
        
        self.norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('relu')
        
    def call(self, inputs):
        x = self.transpose_conv(inputs)
        
        x = self.norm(x)
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'out_dims': self.out_dims,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
    
class ContextEncoder(tf.keras.models.Model):
    def __init__(self, bottleneck_dim, img_size, in_channels, *args, **kwargs):
        super(ContextEncoder, self).__init__(*args, **kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.img_size = img_size 
        self.in_channels = in_channels
       
        len_channel_sizes = 4 if img_size in (64, 128) else 3
        n_channels = [64 * 2 ** i for i in range(len_channel_sizes)]
        
        self.conv1 = tf.keras.layers.Conv2D(
                                    kernel_size=4,
                                    filters=n_channels[0],
                                    strides=2,
                                    padding='same', 
                                    use_bias=False)

        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        blocks = []
        if img_size == 128:
            blocks.append(
                BasicBlock(
                    n_channels[0], kernel_size=4, strides=2, padding='same'
                )
            )
            
        blocks.extend(
            [
                BasicBlock(
                    n_channels[i + 1], kernel_size=4, strides=2, padding='same'
                )
                for i in range(len_channel_sizes - 1)
            ]
        )
        
        self.blocks = tf.keras.Sequential(blocks)
        
        self.conv_bottleneck = tf.keras.layers.Conv2D(filters=bottleneck_dim, kernel_size=4)
    
    def call(self, inputs, training=False):
        x = self.relu1(self.conv1(inputs))
        x = self.blocks(x)
        out = self.conv_bottleneck(x)
        
        return out