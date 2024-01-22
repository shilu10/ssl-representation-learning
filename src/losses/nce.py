import tensorflow as tf 
from tensorflow import keras 
import numpy as np 



class NCE(tf.keras.losses.Loss):
    def __init__(self, config, batch_size, **kwargs):
        super(NCE, self).__init__(**kwargs)
        self.config = config 
        self.temp = config.criterion.get("temp")
        self.batch_size = batch_size
        

    def call(self, f_vi, g_vit, negatives):
        assert f_vi.shape == g_vit.shape, "Shapes do not match" + str(f_vi.shape) + ' != ' + str(g_vit.shape)
        #  predicted input values of 0 and 1 are undefined (hence the clip by value)

        batch_size = f_vi.shape[0]
        return self.n_way_softmax(f_vi, g_vit, negatives) - tf.math.log(
            1 - tf.math.exp(-self.n_way_softmax(g_vit, negatives[:batch_size, :], negatives)))


    def n_way_softmax(self, vi_feat, vit_feat, mem_feat):
    
        pos_sim = tf.reshape(tf.einsum('nc,nc->n', vi_feat, vit_feat), (-1, 1))  # nx1 
        pos_sim /= self.temp

        neg_sim = tf.einsum('nc,ck->nk', vit_feat, tf.transpose(mem_feat))  # nxK
        neg_sim /= self.temp

        logits = tf.concat([pos_sim, neg_sim], axis=1)

        labels = tf.zeros(self.batch_size, dtype=tf.int32)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, name='nce-loss')

        return loss
