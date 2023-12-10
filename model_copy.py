# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K


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

    def save_weights(self, epoch=0, loss=None):
        self.encoder.save_weights(f"simclr_weights_epoch_{epoch}_loss_{loss}.h5")



class MoCo(tf.keras.models.Model):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, encoder, projection_head,
         contrastive_augmenter, classification_augmenter, 
         linear_probe, m=0.1, queue_len=128, **kwargs):

        super(MoCo, self).__init__(dynamic=True)
        self.m = m
        self.queue_len = queue_len

        self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 

        self.encoder = encoder 
        feature_dimensions = encoder.output_shape[1]
        self.queue = tf.zeros((1, feature_dimensions))
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

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)
        
        self.m_encoder.set_weights(self.encoder.get_weights())

        #for layer in self.m_encoder.layers:
         #   layer.trainable = False  

        #for layer in self.m_projection_head.layers:
         #   layer.trainable = False        
    
    def queue_them(self, k):
        if self.queue == None:
            self.queue = k
        elif len(self.queue) >= self.queue_len:
            batch_len = tf.shape(k)[0]
            self.queue = self.queue[batch_len:]
            self.queue = tf.concat([self.queue, k], axis=0)
        else:
            self.queue = tf.concat([self.queue, k], axis=0)

    def save_weights(self, epoch=0, loss=None):
        self.q_enc.save_weights(f"moco_weights_epoch_{epoch}_loss_{loss}.h5")


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
        x_q = self.contrastive_augmenter(X) 
        x_k = self.contrastive_augmenter(X)
        
        q_temp = None
        k_temp = None
        with tf.GradientTape() as tape:
            # embedding representation
            q = self.encoder(x_q)
            k = self.m_encoder(x_k)

            q_temp = q
            k_temp = k 

            q = self.projection_head(q)
            k = self.m_projection_head(k)

            q = tf.reshape(q, (tf.shape(q)[0], 1, -1))
            k = tf.reshape(k, (tf.shape(k)[0], -1, 1))

            contrastive_loss = self.con_loss(q, k, self.queue)

        self.queue_them(tf.squeeze(k))

        encoder_params, proj_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights

        grads = tape.gradient(contrastive_loss, encoder_params + proj_head_params)

        self.contrastive_optimizer.apply_gradients(
            zip(
                grads,
                encoder_params + proj_head_params,
            )
        )

        #self.update_contrastive_accuracy(q_temp, k_temp)
        #self.update_correlation_accuracy(q_temp, k_temp)

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


        # the momentum networks are updated by exponential moving average
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

    def save_weights(self, epoch=0, loss=None):
        self.encoder.save(f"simclr_weights_epoch_{epoch}_loss_{loss}", save_format="tf")

    def con_loss(self, q, k, queue):
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