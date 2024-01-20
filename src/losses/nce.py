import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class NCE(tf.keras.losses.Loss):
    def __init__(self, t, **kwargs):
        super(NCE, self).__init__(**kwargs)
        self.t = t 
        self.cross_entropy = tf.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.cosine_similarity_dim1 = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=1)
        self.cosine_similarity_dim2 = tf.keras.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE, axis=2)

    def call(self, vis, negatives):
        f_vi, g_vit = vis 

        assert f_vi.shape == g_vit.shape, "Shapes do not match" + str(f_vi.shape) + ' != ' + str(g_vit.shape)

        l_pos = tf.expand_dims(sim_func_dim1(f_vi, g_vit), 1)
        l_pos /= self.t

        l_neg = sim_func_dim2(tf.expand_dims(v_it, axis=1), tf.expand_dims(negatives, axis=0))
        l_neg /= self.t

        logits = tf.concat([l_pos, l_neg], axis=1)
        labels = tf.zeros(v_i.shape[0], dtype=tf.int32)

        h_loss = cross_entropy(y_true=labels, y_pred=logits)
        return h_loss
