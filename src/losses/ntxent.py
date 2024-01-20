import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


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

    def call(self, zis, zjs):
        # calculate the positive samples similarities
        l_pos = sim_func_dim1(zis, zjs)
        negative_mask = get_negative_mask(self.batch_size)

        l_pos = tf.reshape(l_pos, (self.batch_size, 1))
        l_pos /= self.tau
        assert l_pos.shape == (self.batch_size, 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        # combine all the zis and zijs and consider as negatives 
        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(self.batch_size, dtype=tf.int64)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (self.batch_size, -1))
            l_neg /= self.tau

            assert l_neg.shape == (
                 self.batch_size, 2 * (self.batch_size - 1)), "Shape of negatives not expected." + str(
                 l_neg.shape)

            logits = tf.concat([l_pos, l_neg], axis=1)  # [N, K+1]
            loss += self.criterion(y_pred=logits, y_true=labels)
        
        loss = loss / (2 * self.batch_size)

        return loss