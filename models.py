# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 


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
    def __init__(self, m=0.1, queue_len=128):
        super(MoCo, self).__init__(dynamic=True)
        self.m = m
        self.queue_len = queue_len

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

        # the momentum networks are initialized from their online counterparts
        self.m_encoder = keras.models.clone_model(self.encoder)
        self.m_projection_head = keras.models.clone_model(self.projection_head)
        
        self.m_encoder.set_weights(self.encoder.get_weights())

        for layer in self.k_enc.layers:
            layer.trainable = False
        
    def call(self, inputs):
        # update key extractor weights
        k_weights = [self.m*w_k + (1-self.m)*w_q for w_k, w_q \
                     in zip(self.k_enc.get_weights(), self.q_enc.get_weights())]
        self.k_enc.set_weights(k_weights)
        # get two versions of same batch data
        x_q, x_k = self.rand_aug(inputs)
        
        # save the key and query
        # sample_q = tf.io.encode_jpeg(tf.cast(x_q[0]*255, tf.uint8))
        # ts = ''.join(map(str, list(time.localtime())))
        # tf.io.write_file(f'{ts}_sample_q.jpeg', sample_q)
        # sample_k = tf.io.encode_jpeg(tf.cast(x_k[0]*255, tf.uint8))
        # tf.io.write_file(f'{ts}_sample_k.jpeg', sample_k)

        # forward
        q = self.g(self.q_enc(x_q))
        q = tf.reshape(q, (tf.shape(q)[0], 1, -1))
        k = self.g(self.k_enc(x_k))
        k = tf.reshape(k, (tf.shape(k)[0], -1, 1))


        
    
    def queue_them(self, k):
        if self.queue == None:
            self.queue = k
        elif len(self.queue) >= self.queue_len:
            batch_len = tf.shape(k)[0]
            self.queue = self.queue[batch_len:]
            self.queue = tf.concat([self.queue, k], axis=0)
        else:
            self.queue = tf.concat([self.queue, k], axis=0)
    
    def rand_aug(self, batch, 
                 resize_min=1, 
                 resize_max=2, 
                 jitter_delta=0.5):
        batch_shape = tf.shape(batch)
        # random reszie
        img_size = batch_shape[1:-1].numpy()
        resize_delta = tf.random.uniform([2, 1], resize_min, resize_max).numpy()
                            
        x_k = tf.image.resize(batch, size=img_size*resize_delta[0])
        x_q = tf.image.resize(batch, size=img_size*resize_delta[1])
        # random crop
        crop_size = (*batch_shape.numpy()[:1], *self.img_shape)
        x_k = tf.image.random_crop(x_k, size=crop_size)
        x_q = tf.image.random_crop(x_q, size=crop_size)
        # random jitter
        x_k = tf.image.random_brightness(x_k, jitter_delta)
        x_k = tf.image.random_contrast(x_k, 1-jitter_delta, 1+jitter_delta)
        x_k = tf.image.random_saturation(x_k, 1-jitter_delta, 1+jitter_delta)
        x_q = tf.image.random_brightness(x_q, jitter_delta)
        x_q = tf.image.random_contrast(x_q, 1-jitter_delta, 1+jitter_delta)
        x_q = tf.image.random_saturation(x_q, 1-jitter_delta, 1+jitter_delta)
        # random horizontal flip
        x_k = tf.image.random_flip_left_right(x_k)
        x_q = tf.image.random_flip_left_right(x_q)
        # random grayscale
        grayscale_or_not = tf.random.uniform([2])
        if grayscale_or_not[0] > 0.5:
            x_k = tf.tile(tf.image.rgb_to_grayscale(x_k), (1, 1, 1, 3))
        if grayscale_or_not[1] > 0.5:
            x_q = tf.tile(tf.image.rgb_to_grayscale(x_q), (1, 1, 1, 3))
        return K.clip(x_k, 0, 255) / 255, K.clip(x_q, 0, 255) / 255
    
    def compute_output_shape(self, input_shape):
        ###### keras-fashion version ######
        # return (input_shape[0], self.queue_len+1)
        ###### gradient-tape version ###### 
        return (1)
    
    def compute_output_shape(self, input_shape):
        ###### keras-fashion version ######
        # return (input_shape[0], self.queue_len+1)
        ###### gradient-tape version ###### 
        return (1)

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