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


class InfoNCE(tf.keras.losses.Loss):
    """ Normalized temperature-scaled CrossEntropy loss [1]
        [1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” arXiv. 2020, Accessed: Jan. 15, 2021. [Online]. Available: https://github.com/google-research/simclr.
    """
    def __init__(self, temp=0.07, batch_size=32, **kwargs):
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

#https://github.com/aniket03/pirl_pytorch/blob/master/pirl_loss.py
def get_img_pair_probs(vi_batch, vi_t_batch, mn_arr, temp_parameter):
    """
    Returns the probability that feature representation for image I and I_t belong to same distribution.
    :param vi_batch: Feature representation for batch of images I
    :param vi_t_batch: Feature representation for batch containing transformed versions of I.
    :param mn_arr: Memory bank of feature representations for negative images for current batch
    :param temp_parameter: The temperature parameter
    """

    # Define constant eps to ensure training is not impacted if norm of any image rep is zero
    eps = 1e-6

    # L2 normalize vi, vi_t and memory bank representations
    #vi_norm_arr = tf.normalize(vi_batch, axis=1)
    vi_norm_arr = tf.norm(vi_batch, ord='euclidean', axis=1, keepdims=True)
    vi_t_norm_arr = tf.norm(vi_t_batch, ord='euclidean', axis=1, keepdims=True)
    mn_norm_arr = tf.norm(mn_arr, ord='euclidean', axis=1, keepdims=True)
    

    vi_batch = vi_batch / (vi_norm_arr + eps)
    vi_t_batch = vi_t_batch/ (vi_t_norm_arr + eps)
    mn_arr = mn_arr / (mn_norm_arr + eps)

    # Find cosine similarities
   # sim_vi_vi_t_arr = (vi_batch @ vi_t_batch.t()).diagonal()
    sim_vi_vi_t_arr = tf.linalg.diag_part(vi_batch @ tf.transpose(vi_t_batch)) # positive similarity
    sim_vi_t_mn_mat = vi_t_batch @ tf.transpose(mn_arr) # negative sim
    

    # Fine exponentiation of similarity arrays
    #exp_sim_vi_vi_t_arr = torch.exp(sim_vi_vi_t_arr / temp_parameter)
    #exp_sim_vi_t_mn_mat = torch.exp(sim_vi_t_mn_mat / temp_parameter)
    exp_sim_vi_vi_t_arr = tf.math.exp(sim_vi_vi_t_arr / temp_parameter)
    exp_sim_vi_t_mn_mat = tf.math.exp(sim_vi_t_mn_mat / temp_parameter)

    # Sum exponential similarities of I_t with different images from memory bank of negatives
    sum_exp_sim_vi_t_mn_arr = tf.math.reduce_sum(exp_sim_vi_t_mn_mat, axis = 1)

    # Find batch probabilities arr
    batch_prob_arr = exp_sim_vi_vi_t_arr / (exp_sim_vi_vi_t_arr + sum_exp_sim_vi_t_mn_arr + eps)

    return batch_prob_arr


def loss_pirl(img_pair_probs_arr, img_mem_rep_probs_arr):
    """
    Returns the average of [-log(prob(img_pair_probs_arr)) - log(prob(img_mem_rep_probs_arr))]
    :param img_pair_probs_arr: Prob vector of batch of images I and I_t to belong to same data distribution.
    :param img_mem_rep_probs_arr: Prob vector of batch of I and mem_bank_rep of I to belong to same data distribution
    """

    # Get 1st term of loss
    neg_log_img_pair_probs = -1 * tf.math.log(img_pair_probs_arr)
    loss_i_i_t = tf.math.reduce_sum(neg_log_img_pair_probs) / neg_log_img_pair_probs.shape[0]

    # Get 2nd term of loss
    neg_log_img_mem_rep_probs_arr = -1 * tf.math.log(img_mem_rep_probs_arr)
    
    loss_i_mem_i = tf.math.reduce_sum(neg_log_img_mem_rep_probs_arr) / neg_log_img_mem_rep_probs_arr.shape[0]

    loss = (loss_i_i_t + loss_i_mem_i) / 2

    return  loss
