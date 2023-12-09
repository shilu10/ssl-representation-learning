from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
import os,sys,shutil 
import tensorflow.keras.backend as K


class NTXent(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, tau=1, **kwargs):
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
        super(NTXent, self).__init__(**kwargs)
        self.cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.criterion = tf.keras.losses.CategoricalCrossentropy(from_logits=True) 
        self.tau = tau

    def call(self, zi, zj):
        z = tf.cast(tf.concat([zi, zj], 0), dtype=tf.float32)
        loss = 0 

        for k in range(zi.shape[0]):
            # Numerator (compare i,j & j,i)
            i = k
            j = k + zi.shape[0]

            sim = tf.squeeze(- self.cosine_sim(tf.reshape(z[i], (1, -1)), tf.reshape(z[j], (1, -1))))
            numerator = tf.math.exp(sim / self.tau)

            # Denominator (compare i & j to all samples apart from themselves)
            sim_ik = - self.cosine_sim(tf.reshape(z[i], (1, -1)), z[tf.range(z.shape[0]) != i])
            sim_jk = - self.cosine_sim(tf.reshape(z[j], (1, -1)), z[tf.range(z.shape[0]) != j])
            denominator_ik = tf.reduce_sum(tf.math.exp(sim_ik / self.tau))
            denominator_jk = tf.reduce_sum(tf.math.exp(sim_jk / self.tau))

            # Calculate individual and combined losses
            loss_ij = - tf.math.log(numerator / denominator_ik)
            loss_ji = - tf.math.log(numerator / denominator_jk)
            loss += loss_ij + loss_ji

        # Divide by the total number of samples
        loss /= z.shape[0]

        return loss 


class InfoNCE(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, **kwargs):
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
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def __call__(self, q, k, queue):
        l_pos = tf.squeeze(tf.matmul(q, k), axis=-1)
        l_neg = tf.matmul(tf.squeeze(q), tf.transpose(queue))
        # logits = softmax(tf.concat([l_pos, l_neg], axis=1))
        logits = tf.concat([l_pos, l_neg], axis=1)
        ###### keras-fashion version ######
        # return logits
        ###### gradient-tape version ###### 
        labels = tf.zeros(tf.shape(q)[0])
        loss = K.mean(self.criterion(labels, logits))
        l2 = tf.reduce_mean(tf.math.l2_normalize(q))
        # print(K.max(logits, axis=1).numpy())
        hits = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, 'int64'))
        return loss + 0.1 * l2