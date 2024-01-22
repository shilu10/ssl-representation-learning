# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from src.networks import contrastive_task as networks
from .common import ContrastiveLearning


def _dense(**custom_kwargs):
    def _func(*args, **kwargs):
        kwargs.update(**custom_kwargs)
        return tf.keras.layers.Dense(*args, **kwargs)
    return _func


class MoCo(ContrastiveLearning):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, config, **kwargs):

        super(MoCo, self).__init__(dynamic=True)
        self.config = config 

        DEFAULT_ARGS = {
            "use_bias": False,
            "kernel_regularizer": tf.keras.regularizers.l2()}

        self.m = config.model.get("m")
        self.version = config.model.get("version")
        self.temp = config.model.get("temp")

        self.feature_dims = config.model.get("feature_dims")   # num_classes 
        self.queue_len = config.model.get("queue_len")         # dictionary len

        def set_encoder(name):
            img_size = self.config.model.get("img_size")
            backbone = getattr(networks, config.networks.get("encoder_type"))(
                include_top=False,
                input_shape=(img_size, img_size, 3),
                pooling='avg')
            
            x = backbone.output
            x = _dense(**DEFAULT_ARGS)(self.feature_dims, name='proj_fc1')(x)

            if config.model.get("version") == "v2":
                x = tf.keras.layers.Activation('relu', name='proj_relu1')(x)
                x = _dense(**DEFAULT_ARGS)(self.feature_dims, name='proj_fc2')(x)

            encoder = tf.keras.models.Model(backbone.input, x, name=name)
            return encoder

        # encoder q and k
        self.encoder_q = set_encoder(name="query_encoder")
        self.encoder_k = set_encoder(name="key_encoder")

        self.initialize_queue(self.feature_dims, self.queue_len)

        queue_ptr = tf.zeros((1, ), dtype=tf.int32)
        self.queue_ptr = tf.Variable(queue_ptr)

        self.initialize_momentum_networks()

        
        #self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

    def call(self, inputs):
        outputs = self.encoder_q(inputs)

        return outputs

    def initialize_queue(self, feature_dims, queue_len):
        _queue = np.random.normal(size=(feature_dims, queue_len))
        _queue /= np.linalg.norm(_queue, axis=0)
        self.queue = self.add_weight(
            name='queue',
            shape=(feature_dims, queue_len),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False)

    def _dequeue_and_enqueue(self, keys):
        # batch size
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size].assign(tf.transpose(keys)) 
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0].assign(ptr) 

    def batch_shuffle(self, tensor):  # nx...
        batch_size = tf.shape(tensor)[0]
        inds = tf.range(batch_size)
        idx_shuffle = tf.random.shuffle(inds)
        
        idx_unshuffle = tf.argsort(idx_shuffle)

        return tf.gather(tensor, idx_shuffle), idx_unshuffle

    def batch_unshuffle(self, key_feat, idx_unshuffle):
        batch_size = tf.shape(key_feat)[0]
        return tf.gather(key_feat, idx_unshuffle)   

    def compile(self, optimizer, loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metrics = metrics

    def train_step(self, inputs):
        
        x_q = inputs[0]
        x_k = inputs[0]

        batch_size = x_q.shape[0]

        # shuffling the batch before encoding(key)
        im_k, idx_unshuffle = self.batch_shuffle(x_k)
        key_feat = self.encoder_k(im_k, training=False)
        key_feat = tf.cast(key_feat, dtype=tf.float32)
        key_feat = tf.math.l2_normalize(key_feat, axis=1)  # NxC

        # unshuffling the batch after encoding(key)
        key_feat = self.batch_unshuffle(key_feat, idx_unshuffle)

        with tf.GradientTape() as tape:
            # embedding representation
            q_feat = self.encoder_q(x_q, training=True)
            q_feat = tf.cast(q_feat, dtype=tf.float32)
            q_feat = tf.math.l2_normalize(q_feat, axis=1)

            # infonce loss
            loss, labels, logits = self.loss(q_feat, key_feat, self.queue)

        # dequeue and enqueue
        self._dequeue_and_enqueue(key_feat)

        trainable_params = self.encoder_q.trainable_weights #+ projection_head_params

        grads = tape.gradient(loss, trainable_params)

        self.optimizer.apply_gradients(
            zip(
                grads,
                trainable_params
            )
        )

        self.update_contrastive_accuracy(q_feat, key_feat)
        self.update_correlation_accuracy(q_feat, key_feat)

        accs = [metric.update_state(labels, logits) for metric in self.acc_metrics]

        self._momentum_update_key_encoder()

        # probe layer
        """
        preprocessed_images = self.classification_augmenter(labeled_X)
        with tf.GradientTape() as tape:
            features = self.encoder(preprocessed_images)
            class_logits = self.linear_probe(features)
            probe_loss = self.probe_loss(labeled_y, class_logits)
        
        gradients = tape.gradient(probe_loss, self.linear_probe.trainable_weights)

        self.probe_optimizer.apply_gradients(
            zip(gradients, self.linear_probe.trainable_weights)
        )
        self.probe_accuracy.update_state(labeled_y, class_logits)
        """
        #results = {}
        results = {m.name: m.result() for m in self.acc_metrics}

        results.update({
            "c_loss": loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
        })

        return results

    def test_step(self, inputs):
        labeled_images, labels = inputs

        preprocessed_images = self.classification_augmenter(
            labeled_images, training=False
        )
        features = self.encoder(preprocessed_images, training=False)
        class_logits = self.linear_probe(features, training=False)
        probe_loss = self.probe_loss(labels, class_logits)

        self.probe_accuracy.update_state(labels, class_logits)
        return {"p_loss": probe_loss, "p_acc": self.probe_accuracy.result()} 


    def con_loss(self, q_feat, key_feat, batch_size):
        # calculating the positive similarities
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1

        # calculating the negative similarites
        l_neg = tf.einsum('nc,ck->nk', q_feat, self.queue)  # nxK
        
        # combining l_pos and l_neg for logits
        logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
        logits /= self.temp 

        # pseduo labels
        labels = tf.zeros(batch_size, dtype=tf.int64)  # n

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        return loss, labels, logits

    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        # the momentum networks are updated by exponential moving average
        """
        encoder_weights = self.encoder.weights
        mom_encoder_weights = self.m_encoder.weights

        for indx in range(len(encoder_weights)):
            weight = encoder_weights[indx]
            m_weight = mom_encoder_weights[indx]

            mom_encoder_weights[indx] = self.m * m_weight + (1 - self.m) * weight

        self.m_encoder.set_weights(mom_encoder_weights)
        """
        encoder_q_weights = self.encoder_q.get_weights()
        encoder_k_weights = self.encoder_k.get_weights()

        for i in range(len(encoder_q_weights)):
            encoder_k_weights[i] = self.m * encoder_k_weights[i] + (1 - self.m) * encoder_q_weights[i]
        
        self.encoder_k.set_weights(encoder_k_weights)

    def initialize_momentum_networks(self):
        self.encoder_k.set_weights(self.encoder_q.get_weights())

        self.encoder_k.trainable = False 

    def one_step(self, input_shape):
        inputs = tf.zeros(input_shape, dtype=tf.float32)

        x = self.encoder_q(inputs)

    @property
    def get_all_trainable_params(self):
        encoder_q_params = self.encoder_q.trainable_variables
        
        return encoder_q_params