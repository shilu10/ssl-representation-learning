from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
import os,sys,shutil 
import tensorflow.keras.backend as K
from utils import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2, get_negative_mask


class NTXent(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, tau=1, batch_size=32, **kwargs):
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
        self.batch_size = batch_size
        self.cosine_sim = tf.keras.losses.CosineSimilarity(axis=-1, reduction=tf.keras.losses.Reduction.NONE)
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                        reduction=tf.keras.losses.Reduction.SUM)
        self.tau = tau

        self.logits = None 
        self.labels = None 

    def call_1(self, zi, zj):
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

    def call(self, zis, zjs):
        
        # calculate the positive samples similarities
        l_pos = sim_func_dim1(zis, zjs)
        negative_mask = get_negative_mask(self.batch_size)

        l_pos = tf.reshape(l_pos, (self.batch_size, 1))
        l_pos /= self.tau
        # assert l_pos.shape == (config['batch_size'], 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        # combine all the zis and zijs and consider as negatives 
        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(self.batch_size, dtype=tf.int64)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (self.batch_size, -1))
            l_neg /= self.tau

            # assert l_neg.shape == (
            #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
            #     l_neg.shape)
            logits = tf.concat([l_pos, l_neg], axis=1)  # [N, K+1]
            loss += self.criterion(y_pred=logits, y_true=labels)
        
        loss = loss / (2 * self.batch_size)

        #self.logits = logits
        #self.labels = labels

        return loss


class InfoNCE(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, temp=0.07, batch_size, **kwargs):
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
        self.temp = temp
        self.batch_size = batch_size
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def __call__(self, q_feat, key_feat, queue):
        # calculating the positive similarities
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1

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



