# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from ..networks import contrastive_task as networks

# https://github.com/drkostas?tab=repositories

'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''

class BYOL(tf.keras.models.Model):
    
    def __init__(self, config, *args, **kwargs):
        super(BYOL, self).__init__(dynamic=True, *args, **kwargs)
        self.config = config 
        self.m = 0.99

        # online encoder
        f_online = getattr(networks, 
                          config.model.get("encoder_type"))             # encoder_online
        self.f_online = f_online(data_format="channels_last",
                                trainable=True)                  
        
        # online projection head 1
        g_online = getattr(networks,
                           config.model.get("projectionhead_1_type"))   # projection head 1 
        self.g_online = g_online()                                     
       
        # online projection head 2 
        q_online = getattr(networks,
                           config.model.get("projectionhead_2_type"))   # projection head 2
        self.q_online = q_online()                         

        # target encoder
        f_target = getattr(networks, 
                          config.model.get("encoder_type"))             # encoder_target
        self.f_target = f_target(data_format="channels_last",
                                trainable=True)                  

        
        # target projection head 
        g_target = getattr(networks,
                           config.model.get("projectionhead_1_type"))   # projection head 1 target 
        self.g_target = g_target()                             

        self._initialize_target_network()

        # metric function 
        self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

         
    def compile(self, optimizer, loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metrics = metrics

    def reset_metrics(self):
        self.contrastive_accuracy.reset_states()
        self.correlation_accuracy.reset_states()
        self.loss_tracker.reset_states()

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
  
        x1 = inputs['view1']    # (bs, img_size, img_size, 3)
        x2 = inputs['view2']    # (bs, img_size, img_size, 3)

        batch_size = x1.shape[0]

        # pass first view of image through target network
        h_target_1 = self.f_target(x1, training=True)
        z_target_1 = self.g_target(h_target_1, training=True)

        # pass second view of image through target network
        h_target_2 = self.f_target(x2, training=True)
        z_target_2 = self.g_target(h_target_2, training=True)

        with tf.GradientTape(persistent=True) as tape:
            # pass first view of image through online network
            h_online_1 = self.f_online(x1, training=True)
            z_online_1 = self.g_online(h_online_1, training=True)
            p_online_1 = self.q_online(z_online_1, training=True)

            # pass first view of image through online network
            h_online_2 = self.f_online(x2, training=True)
            z_online_2 = self.g_online(h_online_2, training=True)
            p_online_2 = self.q_online(z_online_2, training=True)

            p_online = tf.concat(p_online_1, p_online_2, axis=0)
            z_target = tf.concat(z_target_1, z_target_2, axis=0)

            loss = self.loss(p_online, z_target)


        # Backward pass (update online networks)
        f_params = self.f_online.trainable_variables
        g_params = self.g_online.trainable_variables
        q_params = self.q_online.trainable_variables

        grads = tape.gradient(loss, f_params)
        self.optimizer.apply_gradients(zip(grads, f_params))

        grads = tape.gradient(loss, g_params)
        self.optimizer.apply_gradients(zip(grads, g_params))

        grads = tape.gradient(loss, q_params)
        self.optimizer.apply_gradients(zip(grads, q_params))

        self.update_contrastive_accuracy(p_online, z_target)
        self.update_correlation_accuracy(p_online, z_target)

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

        # EMA for target network for each batch.
        self._update_target_network()

        results = {}

        results.update(
            {
                "loss": loss,
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

    def _initialize_target_network(self):
        f_online_weights = self.f_online.get_weights()
        g_online_weights = self.g_online.get_weights()

        self.f_target.set_weights(f_online_weights)
        self.g_target.set_weights(g_online_weights)

    def _update_target_network(self):
        # update target encoder and projection head 
        f_online_weights = self.f_online.get_weights()
        f_target_weights = self.f_target.get_weights()

        g_online_weights = self.g_online.get_weights()
        g_target_weights = self.g_target.get_weights()

        for i in range(len(f_online_weights)):
            f_target_weights[i] = self.m * f_target_weights[i] + (1 - self.m) * f_online_weights[i]

        for i in range(len(g_online_weights)):
            g_target_weights[i] = self.m * g_target_weights[i] + (1 - self.m) * g_online_weights[i] 

        self.f_target.set_weights(f_target_weights)
        self.q_target.set_weights(g_target_weights)


