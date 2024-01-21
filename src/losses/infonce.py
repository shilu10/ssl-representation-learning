import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class InfoNCE(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, config, **kwargs):
        """ 
            Calculates the contrastive loss of the input data using NT_Xent. The
            equation can be found in the paper: https://arxiv.org/pdf/2002.05709.pdf
            (This is the Tensorflow implementation of the standard numpy version found
            in the NT_Xent function).
            
            Args:
                zi: One half of the input data, shape = (batch_size, feature_1, feature_2, ..., feature_N)
                zj: Other half of the input data, must have the same shape as zi
                tau: Temperature parameter (a constant), default = 1.

            Returns:
                loss: The complete NT_Xent constrastive loss
        """
        super(InfoNCE, self).__init__(**kwargs)

        self.temp = config.model.get("temp")
        self.batch_size = config.model.get("batch_size")
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def __call__(self, q_feat, key_feat, queue):
        # calculating the positive similarities
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, tf.stop_gradient(key_feat)), (-1, 1))  # nx1

        # calculating the negative similarites
        l_neg = tf.einsum('nc,ck->nk', q_feat, queue)  # nxK
        
        # combining l_pos and l_neg for logits
        logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
        logits /= self.temp 

        # pseduo labels
        labels = tf.zeros(self.batch_size, dtype=tf.int64)  # n

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        return loss 
