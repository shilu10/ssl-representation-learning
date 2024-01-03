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
    

class ContextDeocder(tf.keras.models.Model):
    def __init__(self, bottleneck_dim, out_size, out_channels, *args, **kwargs):
        super(ContextDeocder, self).__init__(*args, **kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.out_channels = out_channels 
        self.out_size = out_size
        
        len_channel_sizes = 4 if out_size in (64, 128) else 3
        n_channels = [64 * 2 ** i for i in range(len_channel_sizes - 1, -1, -1)]
        
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.bottleneck_block = TransposeBlock(
                n_channels[0], kernel_size=4, strides=1, padding='valid'
        )
        
        blocks = [
            TransposeBlock(
                 n_channels[i + 1], kernel_size=4, strides=2, padding='same'
            )
            for i in range(len_channel_sizes - 1)
        ]
        
        if out_size == 128:
            blocks.append(
                TransposeBlock(
                    n_channels[-1], kernel_size=4, strides=2, padding='same'
                )
            )
            
        self.blocks = tf.keras.Sequential(blocks)

        self.final_conv = tf.keras.layers.Conv2DTranspose(
                out_channels, kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.tanh = tf.keras.layers.Activation('tanh')
        
    def call(self, inputs, training=False):
        x = self.relu1(self.bn1(inputs))
        x = self.bottleneck_block(x)
        x = self.blocks(x)
        out = self.tanh(self.final_conv(x))
        
        return out
    

class ContextGenerator(tf.keras.models.Model):
    def __init__(self, bottleneck_dim, img_size, out_size, channels, *args, **kwargs):
        super(ContextGenerator, self).__init__(*args, **kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.img_size = img_size 
        self.out_size = out_size 
        self.channels = channels 
        
        self.context_encoder = ContextEncoder(bottleneck_dim, img_size, channels)
        self.context_decoder = ContextDeocder(bottleneck_dim, out_size, channels)
        
    def call(self, inputs, training=False):
        x = self.context_encoder(inputs)
        x = self.context_decoder(x)
        
        return x 
    
    
class ContextDiscriminator(tf.keras.Model):
    def __init__(self, input_size: int = 128, in_channels=3, *args, **kwargs):
        super(ContextDiscriminator, self).__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.input_size = input_size 
        
        len_channel_sizes = 4 if input_size in (64, 128) else 3
        n_channels = [64 * 2 ** i for i in range(len_channel_sizes)]
        
        self.conv1 = tf.keras.layers.Conv2D(
            n_channels[0], kernel_size=4, strides=2, padding='same', use_bias=False
        )
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        
        blocks = []
        if input_size == 128:
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

        self.final_conv = tf.keras.layers.Conv2D(n_channels[-1], padding='same', kernel_size=4, use_bias=False)
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.out = tf.keras.layers.Dense(1, activation='sigmoid')
        
    def call(self, inputs, training=False):
        x = self.relu1(self.conv1(inputs))
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.gap(x)
        x = self.out(x)
        
        return x


if __name__ == '__main__':

	inputs = tf.random.uniform((2, 128, 128, 3))

	context_generator = ContextGenerator(1024, 128, 128, 3)
	context_discriminator = ContextDiscriminator(128, 3)

	c_gen_output = context_generator(inputs)
	c_dis_output = context_discriminator(c_gen_output)

	print(f'Discrimnator Ouput: {c_dis_output}')