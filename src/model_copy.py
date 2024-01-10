# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K


'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''




# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 

'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''

class SimCLR(tf.keras.models.Model):
    
    def __init__(self, encoder, projection_head,
         contrastive_augmenter, classification_augmenter, linear_probe, **kwargs):
        super(SimCLR, self).__init__(**kwargs)
        self.encoder = encoder 
        self.projection_head = projection_head 
        self.contrastive_augmenter = contrastive_augmenter 
        self.classification_augmenter = classification_augmenter
        self.linear_probe = linear_probe

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # loss function
        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
         
    def compile(self, contrastive_optimizer,
                 probe_optimizer, contrastive_loss, **kwargs):
        super().compile(**kwargs)
        self.probe_optimizer = probe_optimizer
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss = contrastive_loss

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

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

        batch_size = tf.shape(features_1, out_type=tf.float32)[0]
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
        # unlabeled images and labeled i'mages
        (unlabeled_X, _), (labeled_X, labeled_y) = inputs
        # combining both labeled and unlabeled images
        X = tf.concat([unlabeled_X, labeled_X], axis=0)
        xi = self.contrastive_augmenter(X) 
        xj = self.contrastive_augmenter(X)

        with tf.GradientTape() as tape:
            # embedding representation
            hi = self.encoder(xi)
            hj = self.encoder(xj)

            # the representations are passed through a projection mlp
            zi = self.projection_head(hi)
            zj = self.projection_head(hj)

            contrastive_loss = self.contrastive_loss(zi, zj)

        encoder_params, proj_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights

        grads = tape.gradient(contrastive_loss, encoder_params + proj_head_params)

        self.contrastive_optimizer.apply_gradients(
            zip(
                grads,
                encoder_params + proj_head_params,
            )
        )

        self.update_contrastive_accuracy(hi, hj)
        self.update_correlation_accuracy(hi, hj)

        # probe layer
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

        return {
            "c_loss": contrastive_loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
            "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
        }

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




class MoCo(tf.keras.models.Model):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, encoder, projection_head,
         linear_probe,contrastive_augmenter, m=0.999, queue_len=65000, **kwargs):

        super(MoCo, self).__init__(dynamic=True)
       # hvd.init()
        self.m = m
        self.queue_len = queue_len


        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

        self.encoder = encoder 
        feature_dimensions = encoder.output_shape[1]
        queue_init = tf.math.l2_normalize(
            tf.random.normal([queue_len, feature_dimensions]), axis=1)

        queue = tf.compat.v1.get_variable('queue', initializer=queue_init, trainable=False)

        queue_ptr = tf.compat.v1.get_variable(
            'queue_ptr',
            [], initializer=tf.zeros_initializer(),
            dtype=tf.int64, trainable=False)

        self.temp = 0.07

        self.queue = queue
        self.queue_ptr = queue_ptr

        #self.queue = tf.zeros((1, feature_dimensions))
        #self.queue_len = queue_len
            
        self.projection_head = projection_head 
        self.contrastive_augmenter = contrastive_augmenter 
        #self.classification_augmenter = classification_augmenter
        self.linear_probe = linear_probe

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.probe_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        # loss function
        self.probe_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)
        
        self.m_encoder.set_weights(self.encoder.get_weights())


        #self.register_buffer("queue", torch.randn(dim, K))
        #self.queue = nn.functional.normalize(self.queue, dim=0)

        #self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        #queue = tf.zeros((feature_dimensions, queue_len))
        #self.queue = tf.math.l2_normalize(queue, axis=0)

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
        #if self.queue == None:
        #    self.queue = k
        #elif len(self.queue) >= self.queue_len:
         #   batch_len = tf.shape(k)[0]
          #  self.queue = self.queue[batch_len:]
           # self.queue = tf.concat([self.queue, k], axis=0)
        #else:
         #   self.queue = tf.concat([self.queue, k], axis=0)

        # from pytorch impl
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.K % batch_size == 0  # for simplicity

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
        
    def push_queue(self, queue, queue_ptr, item):
        # queue: KxC
        # item: NxC
        
        #item = allgather(item, 'queue_gather')  # GN x C
        batch_size = tf.shape(item, out_type=tf.int64)[0]
        end_queue_ptr = queue_ptr + batch_size

        inds = tf.range(queue_ptr, end_queue_ptr, dtype=tf.int64)
        with tf.control_dependencies([inds]):
            queue_ptr_update = tf.compat.v1.assign(queue_ptr, end_queue_ptr % 128)
        queue_update = tf.compat.v1.scatter_update(queue, inds, item)
        return tf.group(queue_update, queue_ptr_update)




    def compile(self, contrastive_optimizer,
                 probe_optimizer, contrastive_loss, **kwargs):
        super().compile(**kwargs)
        self.probe_optimizer = probe_optimizer
        self.contrastive_optimizer = contrastive_optimizer
        self.contrastive_loss = contrastive_loss
        self.acc_metrics = [
            tf.keras.metrics.SparseTopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)
        ]

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.probe_accuracy.reset_states()

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
        
        # unlabeled images and labeled i'mages
        (unlabeled_X, _), (labeled_X, labeled_y) = inputs
        
        # combining both labeled and unlabeled images
        #X = tf.concat([unlabeled_X, labeled_X], axis=0)
        batch_size = unlabeled_X.shape[0]
        x_q = self.contrastive_augmenter(unlabeled_X) 
        x_k = self.contrastive_augmenter(unlabeled_X)

        #shuffled_key, shuffle_idxs = self.batch_shuffle(x_k)
        #shuffled_key.set_shape([batch_size, None, None, None])
        
        with tf.GradientTape() as tape:
            # embedding representation
            q_feat = self.encoder(x_q)
            q_feat = tf.math.l2_normalize(q_feat, axis=1)

            #key_feat = self.m_encoder(shuffled_key)
            im_k, idx_unshuffle = self.batch_shuffle(x_k)

            key_feat = self.m_encoder(im_k)
            key_feat = tf.math.l2_normalize(key_feat, axis=1)  # NxC
          #  key_feat = self.batch_unshuffle(key_feat, shuffle_idxs)

            key_feat = self.batch_unshuffle(key_feat, idx_unshuffle)
            
            #key_feat = tf.stop_gradient(key_feat)
            l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1
            l_neg = tf.einsum('nc,ck->nk', q_feat, self.queue)  # nxK
            
            #l_neg = tf.einsum('nc,ck->nk', q_feat, self.queue)  # nxK
            logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
            logits /= self.temp 
            labels = tf.zeros(batch_size, dtype=tf.int64)  # n

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss = tf.reduce_mean(loss, name='xentropy-loss')


        self._dequeue_and_enqueue(key_feat)
        #self.push_queue(self.queue, self.queue_ptr, key_feat)

        #encoder_params, projection_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights
        encoder_params = self.encoder.trainable_weights

        #grads = tape.gradient(loss, encoder_params + projection_head_params)
        grads = tape.gradient(loss, encoder_params)

        self.contrastive_optimizer.apply_gradients(
            zip(
                grads,
                encoder_params #+ projection_head_params,
            )
        )

        self.update_contrastive_accuracy(q_feat, key_feat)
        self.update_correlation_accuracy(q_feat, key_feat)

        accs = [metric.update_state(labels, logits) for metric in self.acc_metrics]

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


        # the momentum networks are updated by exponential moving average
        encoder_weights = self.encoder.weights
        mom_encoder_weights = self.m_encoder.weights

        for indx in range(len(encoder_weights)):
            weight = encoder_weights[indx]
            m_weight = mom_encoder_weights[indx]

            mom_encoder_weights[indx] = self.m * m_weight + (1 - self.m) * weight

        self.m_encoder.set_weights(mom_encoder_weights)

        #for weight, m_weight in zip(self.encoder.weights, self.m_encoder.weights):
         #   m_weight.assign(
                
          #  )
        for weight, m_weight in zip(
            self.projection_head.weights, self.m_projection_head.weights
        ):
            m_weight.assign(
                self.m * m_weight + (1 - self.m) * weight
            )

        
        results = {m.name: m.result() for m in self.acc_metrics}
        results.update(
            {
            "c_loss": loss,
            "c_acc": self.contrastive_accuracy.result(),
            "r_acc": self.correlation_accuracy.result(),
           # "p_loss": probe_loss,
            "p_acc": self.probe_accuracy.result(),
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


    def con_loss(self, q_feat, key_feat, queue, temp, batch_size):
        l_pos = tf.reshape(tf.einsum('nc,nc->n', q_feat, key_feat), (-1, 1))  # nx1
        l_neg = tf.einsum('nc,kc->nk', q_feat, queue)  # nxK
        logits = tf.concat([l_pos, l_neg], axis=1)  # nx(1+k)
        logits = logits * (1 / temp)
        labels = tf.zeros(batch_size, dtype=tf.int64)  # n
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        loss = tf.reduce_mean(loss, name='xentropy-loss')

        return loss 

    def con_loss_1(self, q, k, batch_size):

        q = tf.reshape(q, (tf.shape(q)[0], 1, -1))
        k = tf.reshape(k, (tf.shape(k)[0], -1, 1))
       
        l_pos = tf.squeeze(tf.matmul(q, k), axis=-1)
        l_neg = tf.matmul(tf.squeeze(q), tf.transpose(self.queue))
        # logits = softmax(tf.concat([l_pos, l_neg], axis=1))
        logits = tf.concat([l_pos, l_neg], axis=1)
       #self.queue_them(tf.squeeze(k))
        ###### keras-fashion version ######
        # return logits
        ###### gradient-tape version ###### 
        labels = tf.zeros(batch_size)
        loss = K.mean(sparse_categorical_crossentropy(labels, logits, from_logits=True))
        l2 = tf.reduce_mean(tf.math.l2_normalize(q))
        # print(K.max(logits, axis=1).numpy())
        hits = tf.equal(tf.argmax(logits, axis=1), tf.cast(labels, 'int64'))
        acc = tf.reduce_mean(tf.cast(hits, 'float64'))

        return loss + 0.1 * l2

    # somewhat working
    def con_loss_2(self):
        # loss
            #loss = self.con_loss(q_feat, key_feat, self.queue, self.temp, batch_size)
            #loss = self.con_loss_1(q_feat, key_feat, batch_size)
            #loss = tf.cast(loss, dtype=tf.float32)
            labels = tf.zeros(batch_size, dtype=tf.int64)  # n
            l_pos = tf.einsum('nc,nc->n', q_feat, tf.stop_gradient(key_feat))[:,None]
            l_neg = tf.einsum('nc,ck->nk', q_feat, self.queue)
            logits = tf.concat((l_pos, l_neg), axis=1)
            logits /= self.temp

            loss_moco = sparse_categorical_crossentropy(labels, logits, from_logits=True)
            loss  = tf.reduce_mean(loss_moco)