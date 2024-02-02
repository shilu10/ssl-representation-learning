import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from typing import Union
from .common import ConvLayer, LinearLayer, LRNLayer


class AlexNetJigSaw(tf.keras.models.Model):
    def __init__(self, config, n_classes=1000, *args, **kwargs):
        super(AlexNetJigSaw, self).__init__(*args, **kwargs)
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

    def forward_once(self, inputs, training=False):
        B, T, H, W, C = inputs.shape 

        x = self.conv_1(inputs[i])
        x = self.lrn_1(x)

        x = self.conv_2(x)
        x = self.lrn_2(x)

        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)

        x = self.flatten(x)
        x = self.fc_1(x)

        return x 

    def call(self, inputs, training=False):
        # B-batch, T-tile, H-height, W-width, C-channels
        B, T, H, W, C = inputs.shape 
        inputs = tf.transpose(inputs, perm=(1, 0, 2, 3, 4))

        x_list = []
        for i in range(9):
            # conv layer and lrn layers
            x = self.conv_1(inputs[i])
            x = self.lrn_1(x)

            x = self.conv_2(x)
            x = self.lrn_2(x)

            x = self.conv_3(x)
            x = self.conv_4(x)
            x = self.conv_5(x)

            x = self.flatten(x)
            x = self.fc_1(x)

            x_list.append(x)

        # linear layers
        x = tf.concat(x_list, axis=1)
        x = self.fc_2(x)

        # output layer
        out = self.out(x)

        return out 

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                n_classes: self.n_classes
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_model(config, n_classes):
    return AlexNetJigSaw(config=config, n_classes=n_classes)


if __name__ == '__main__':
    
    class Config:
        def __init__(self):
            pass 

    config = Config()

    model = create_model(config, 10)

    size = 128

    x = tf.random.uniform((1, size, size, 3), -1, 1)
    out = model(x)

    print(f"output shape: {out.shape}")
