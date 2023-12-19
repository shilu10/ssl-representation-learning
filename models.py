# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from utils import _cosine_simililarity_dim1 as sim_func_dim1, _cosine_simililarity_dim2 as sim_func_dim2
from utils import get_negative_mask


# https://github.com/drkostas?tab=repositories

'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''

class SimCLR(tf.keras.models.Model):
    
    def __init__(self, encoder, projection_head, **kwargs):
        super(SimCLR, self).__init__(dynamic=True, **kwargs)
        self.encoder = encoder 
        self.projection_head = projection_head 

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # loss function
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, 
                                                                        reduction=tf.keras.losses.Reduction.SUM)
         
    def compile(self, contrastive_optimizer,
                 probe_optimizer, contrastive_loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.probe_optimizer = probe_optimizer
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss = contrastive_loss
        self.acc_metrics = metrics

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def train_step(self, inputs):
  
        xi = inputs['query']
        xj = inputs['key']

        batch_size = xi.shape[0]

        with tf.GradientTape() as tape:
            # embedding representation
            hi = self.encoder(xi)
            hj = self.encoder(xj)

            # the representations are passed through a projection mlp
            zi = self.projection_head(hi)
            zj = self.projection_head(hj)

            # apply l2 normalization on zis and zjs 
            zi = tf.math.l2_normalize(zi, axis=1)
            zj = tf.math.l2_normalize(zj, axis=1)

            loss, labels, logits = self._loss(zi, zj, batch_size)

        encoder_params, proj_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights
        trainable_params = encoder_params + proj_head_params

        grads = tape.gradient(loss, trainable_params)

        self.contrastive_optimizer.apply_gradients(
            zip(
                grads,
                trainable_params,
            )
        )

        self.update_contrastive_accuracy(hi, hj)
        self.update_correlation_accuracy(hi, hj)

        accs = [metric.update_state(labels, logits) for metric in self.acc_metrics]

        # for probe layer (lncls model parallel computation)
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

        results = {m.name: m.result() for m in self.acc_metrics}

        results.update(
            {
                "c_loss": loss,
                "c_acc": self.contrastive_accuracy.result(),
                "r_acc": self.correlation_accuracy.result(),
            }
        ) 

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

    def _loss(self, zis, zjs, batch_size):
        
        # calculate the positive samples similarities
        l_pos = sim_func_dim1(zis, zjs)
        negative_mask = get_negative_mask(batch_size)

        l_pos = tf.reshape(l_pos, (batch_size, 1))
        l_pos /= 0.07
        assert l_pos.shape == (batch_size, 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        # combine all the zis and zijs and consider as negatives 
        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0
        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(batch_size, dtype=tf.int64)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (batch_size, -1))
            l_neg /= 0.07

            assert l_neg.shape == (
                 batch_size, 2 * (batch_size - 1)), "Shape of negatives not expected." + str(
                 l_neg.shape)

            logits = tf.concat([l_pos, l_neg], axis=1)  # [N, K+1]
            loss += self.criterion(y_pred=logits, y_true=labels)
        
        loss = loss / (2 * batch_size)

        return loss, labels, logits


class MoCo(tf.keras.models.Model):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, encoder, projection_head, m=0.999, queue_len=6500, version, **kwargs):

        super(MoCo, self).__init__(dynamic=True)
        self.m = m
        self.version = version
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        
        feature_dimensions = encoder.output_shape[1]
        self.temp = 0.07

        self.encoder = encoder 
        self.projection_head = projection_head 

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)
        
        self.m_encoder.set_weights(self.encoder.get_weights())

        _queue = np.random.normal(size=(feature_dimensions, queue_len))
        _queue /= np.linalg.norm(_queue, axis=0)
        self.queue = self.add_weight(
            name='queue',
            shape=(feature_dimensions, queue_len),
            initializer=tf.keras.initializers.Constant(_queue),
            trainable=False)

        queue_ptr = tf.zeros((1, ), dtype=tf.int32)
        self.queue_ptr = tf.Variable(queue_ptr)
        self.queue_len = queue_len

        for layer in self.m_encoder.layers:
            layer.trainable = False  

        for layer in self.m_projection_head.layers:
            layer.trainable = False        


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

    def compile(self, contrastive_optimizer,
                 probe_optimizer, contrastive_loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.probe_optimizer = probe_optimizer
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss = contrastive_loss
        self.acc_metrics = metrics

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def train_step(self, inputs):
        
        x_q = inputs['query']
        x_k = inputs['key']

        batch_size = x_q.shape[0]

        with tf.GradientTape() as tape:
            # embedding representation
            q_feat = self.encoder(x_q)

            if self.version == 'mocov2':
                q_feat = self.projection_head(q_feat)
            q_feat = tf.math.l2_normalize(q_feat, axis=1)

            # shuffling the batch before encoding(key)
            im_k, idx_unshuffle = self.batch_shuffle(x_k)
            key_feat = self.m_encoder(im_k)
            if self.version == 'mocov2':
                key_feat = self.m_projection_head(key_feat)
            key_feat = tf.math.l2_normalize(key_feat, axis=1)  # NxC

            # unshuffling the batch after encoding(key)
            key_feat = self.batch_unshuffle(key_feat, idx_unshuffle)

            # infonce loss
            loss, labels, logits = self.con_loss(q_feat, key_feat, batch_size)

        # dequeue and enqueue
        self._dequeue_and_enqueue(key_feat)

        encoder_params, projection_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights

        trainable_params = encoder_params #+ projection_head_params
        if self.version == 'mocov2':
            trainable_params = encoder_params + projection_head_params

        grads = tape.gradient(loss, trainable_params)

        self.contrastive_optimizer.apply_gradients(
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
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.m * m_weight + (1 - self.m) * weight
            )

        if self.version == 'mocov2':

            for weight, m_weight in zip(
                self.projection_head.weights, self.m_projection_head.weights
            ):
                m_weight.assign(
                    self.m * m_weight + (1 - self.m) * weight
                )


class PIRL(tf.keras.models.Model):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, encoder, f, g, **kwargs):

        super(MoCo, self).__init__(dynamic=True)
        self.g = g
        self.f = f
        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
        
        self.temp = 0.07

        self.encoder = encoder 

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  

    def compile(self, contrastive_optimizer,
                 probe_optimizer, contrastive_loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.probe_optimizer = probe_optimizer
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss = contrastive_loss
        self.acc_metrics = metrics

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()

    def update_contrastive_accuracy(self, features_1, features_2):
        features_1 = tf.math.l2_normalize(features_1, axis=1)
        features_2 = tf.math.l2_normalize(features_2, axis=1)
        similarities = tf.matmul(features_1, features_2, transpose_b=True)

        batch_size = tf.shape(features_1)[0]
        contrastive_labels = tf.range(batch_size)
        self.contrastive_accuracy.update_state(
            tf.concat([contrastive_labels, contrastive_labels], axis=0),
            tf.concat([similarities, tf.transpose(similarities)], axis=0),
        )

    def update_correlation_accuracy(self, features_1, features_2):
        features_1 = (
            features_1 - tf.reduce_mean(features_1, axis=0)
        ) / tf.math.reduce_std(features_1, axis=0)
        features_2 = (
            features_2 - tf.reduce_mean(features_2, axis=0)
        ) / tf.math.reduce_std(features_2, axis=0)

        batch_size = tf.shape(features_1)[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)
        cross_correlation = (
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size
        )

        feature_dim = tf.shape(features_1)[1]
        correlation_labels = tf.range(feature_dim)
        self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )

    def train_step(self, inputs):
        
        original = inputs['original']
        transformed = inputs['transformed']

        batch_size = x_q.shape[0]
        transformed = tf.concat([*transformed], axis=0)

        with tf.GradientTape() as tape:
            # embedding representation
            q_feat = self.encoder(x_q)
            #q_feat = self.projection_head(q_feat)
            q_feat = tf.math.l2_normalize(q_feat, axis=1)

            # shuffling the batch before encoding(key)
            im_k, idx_unshuffle = self.batch_shuffle(x_k)
            key_feat = self.m_encoder(im_k)
            # key_feat = self.m_projection_head(key_feat)
            key_feat = tf.math.l2_normalize(key_feat, axis=1)  # NxC

            # unshuffling the batch after encoding(key)
            key_feat = self.batch_unshuffle(key_feat, idx_unshuffle)

            # infonce loss
            loss, labels, logits = self.con_loss(q_feat, key_feat, batch_size)

        # dequeue and enqueue
        self._dequeue_and_enqueue(key_feat)

        encoder_params, projection_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights
        #encoder_params = self.encoder.trainable_weights

        trainable_params = encoder_params #+ projection_head_params

        #grads = tape.gradient(loss, encoder_params + projection_head_params)
        grads = tape.gradient(loss, trainable_params)

        self.contrastive_optimizer.apply_gradients(
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
        for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
            m_weight.assign(
                self.m * m_weight + (1 - self.m) * weight
            )

        for weight, m_weight in zip(
            self.projection_head.weights, self.m_projection_head.weights
        ):
            m_weight.assign(
                self.m * m_weight + (1 - self.m) * weight
            )
