import tensorflow as tf
import numpy as np
import collections, os, datetime
from src.utils.progbar_utils import Progbar


class MemoryBank:
    def __init__(self, shape, weight=0.5):
        self.weight = weight
        self.shape = shape

        memory_initializer = tf.random.truncated_normal(shape)

        self.memory = tf.Variable(memory_initializer, trainable=False)

    def initialize(self, encoder, f, train_loader, steps_per_epoch, sep_init=True):

        bar = Progbar(steps_per_epoch, stateful_metrics=[])
        for step, batch in enumerate(train_loader):

            if sep_init:
                indices, images = batch 

            else:
                data, indices = batch
                images = data['original']

            output = encoder(images, training=False)
            values = f(output, training=False)

            self.memory = tf.tensor_scatter_nd_update(self.memory, tf.expand_dims(indices, 1), values)
            
            bar.update(step, values=[])

            if step == steps_per_epoch:
                break

    def update_memory_repr(self, indices, features):
        # perform batch update to the representations
        memory_value = tf.gather(self.memory, indices)
        values = (self.weight * memory_value + (1 - self.weight) * features)

        update_indices = tf.expand_dims(indices, 1)
        self.memory = tf.tensor_scatter_nd_update(self.memory, update_indices, values)

    def sample_negatives(self, positive_indices, batch_size):
        positive_indices = tf.expand_dims(positive_indices, axis=1)
        updates = tf.zeros(positive_indices.shape[0], dtype=tf.int32)

        mask = tf.ones([self.shape[0]], dtype=tf.int32)
        mask = tf.tensor_scatter_nd_update(mask, positive_indices, updates)

        p = tf.ones(self.shape[0])
        p = p * tf.cast(mask, tf.float32)
        p = p / tf.reduce_sum(p)

        candidate_negative_indices = tf.random.categorical(tf.math.log(tf.reshape(p, (1, -1))),
                                                           batch_size)  # note log-prob
        embeddings = tf.nn.embedding_lookup(self.memory, tf.squeeze(candidate_negative_indices))
        return embeddings

    def sample_by_indices(self, batch_indices):
        return tf.nn.embedding_lookup(self.memory, batch_indices)
