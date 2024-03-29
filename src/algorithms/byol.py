# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from src.networks import contrastive_task as networks
from .common import ContrastiveLearning
from .utils import _dense, _conv2d

# https://github.com/drkostas?tab=repositories

'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''

class BYOL(ContrastiveLearning):
    
    def __init__(self, config, *args, **kwargs):
        super(BYOL, self).__init__(dynamic=True, *args, **kwargs)
        self.config = config 
        self.m = config.model.get('m')

        img_size = config.model.get("img_size")
        hidden_dims = config.model.get("hidden_dims")
        projection_dims = config.model.get("projection_dims")

        DEFAULT_ARGS = {
            "use_bias": False,
            "kernel_regularizer": tf.keras.regularizers.l2()}

        # online encoder
        self.f_online = getattr(networks, config.networks.get("encoder_type"))(
                                            include_top=False,
                                            input_shape=(img_size, img_size, 3),
                                            pooling='avg')           # encoder_online
        
        # online projection head 1
        self.g_online = tf.keras.models.Sequential([
                                        _dense(**DEFAULT_ARGS)(hidden_dims, name="fc1"), 
                                        tf.keras.layers.BatchNormalization(), 
                                        tf.keras.layers.Activation("relu"), 
                                        _dense(**DEFAULT_ARGS)(projection_dims, name="fc2")])

       
        # online projection head 2 
        self.q_online = tf.keras.models.Sequential([
                                        _dense(**DEFAULT_ARGS)(hidden_dims, name="fc1"), 
                                        tf.keras.layers.BatchNormalization(), 
                                        tf.keras.layers.Activation("relu"), 
                                        _dense(**DEFAULT_ARGS)(projection_dims, name="fc2")])

        # target encoder
        self.f_target = getattr(networks, config.networks.get("encoder_type"))(
                                            include_top=False,
                                            input_shape=(img_size, img_size, 3),
                                            pooling='avg')            # encoder_target
        

        self.g_target = tf.keras.models.Sequential([
                                        _dense(**DEFAULT_ARGS)(hidden_dims, name="fc1"), 
                                        tf.keras.layers.BatchNormalization(), 
                                        tf.keras.layers.Activation("relu"), 
                                        _dense(**DEFAULT_ARGS)(projection_dims, name="fc2")])

        self.one_step((1, img_size, img_size, 3))

        self._initialize_target_network()

         
    def compile(self, optimizer, loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metrics = metrics

    def call(self, inputs):
        x = self.f_online(inputs)
        x = self.g_online(x)

        return x 
   
    def train_step(self, inputs):

  
        x1 = inputs[0]    # (bs, img_size, img_size, 3)
        x2 = inputs[1]   # (bs, img_size, img_size, 3)

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

            p_online = tf.concat([p_online_1, p_online_2], axis=0)
            z_target = tf.concat([z_target_1, z_target_2], axis=0)

            loss = self.loss(p_online, z_target)

        trainable_params = self.get_all_trainable_params
        grads = tape.gradient(loss, trainable_params)
        
        self.optimizer.apply_gradients(
            zip(
                grads,
                trainable_params,
            )
        )

        # Delete the tape to free up resources
        del tape
        
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
        self.g_target.set_weights(g_target_weights)

    def one_step(self, input_shape):
        inputs = tf.zeros(input_shape, dtype=tf.float32)

        x = self.f_online(inputs)
        x = self.g_online(x)
        x = self.q_online(x)

        x = self.f_target(inputs)
        x = self.g_target(x)

    @property
    def get_all_trainable_params(self):
        f_online_params = self.f_online.trainable_variables
        g_online_params = self.g_online.trainable_variables
        q_online_params = self.q_online.trainable_variables

        all_online_params = f_online_params + g_online_params + q_online_params

        return all_online_params

    def save_encoder_weights(self, filepath):
        self.f_online.save_weights(filepath)

    def load_encoder_weights(self, filepath):
        self.f_online.load_weights(filepath, overwrite=True)
