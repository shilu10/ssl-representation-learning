import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class BYOL(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, *args, **kwargs):
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
        super(BYOL, self).__init__(*args, **kwargs)
        

    def call(self, prediction, target):
        prediction = tf.math.l2_normalize(prediction, axis=1)  # (2*bs, 128)
        target = tf.math.l2_normalize(target, axis=1)  # (2*bs, 128)

        similarities = tf.reduce_sum(tf.multiply(prediction, target), axis=1)
        return 2 - 2 * tf.reduce_mean(similarities)