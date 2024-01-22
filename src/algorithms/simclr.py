# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from src.utils.contrastive_task import _cosine_simililarity_dim1 as sim_func_dim1, _cosine_simililarity_dim2 as sim_func_dim2
from src.utils.contrastive_task import get_negative_mask
from src.networks import contrastive_task as networks
from .common import ContrastiveLearning
from .utils import _dense, _conv2d


# https://github.com/drkostas?tab=repositories

'''Contrastive accuracy: self-supervised metric, the ratio of cases in which the representation of an image is more similar to its differently augmented version's one, than to the representation of any other image in the current batch. Self-supervised metrics can be used for hyperparameter tuning even in the case when there are no labeled examples.
Linear probing accuracy: linear probing is a popular metric to evaluate self-supervised classifiers. It is computed as the accuracy of a logistic regression classifier trained on top of the encoder's features. In our case, this is done by training a single dense layer on top of the frozen encoder. Note that contrary to traditional approach where the classifier is trained after the pretraining phase, in this example we train it during pretraining. This might slightly decrease its accuracy, but that way we can monitor its value during training, which helps with experimentation and debugging.
'''

class SimCLR(ContrastiveLearning):
    
    def __init__(self, config, *args, **kwargs):
        super(SimCLR, self).__init__(dynamic=True, *args, **kwargs)
        hidden_dims = config.model.get("hidden_dims")
        projection_dims = config.model.get("projection_dims")

        img_size = config.model.get("img_size")
        self.encoder = getattr(networks, config.networks.get("encoder_type"))(
                        include_top=False,
                        input_shape=(img_size, img_size, 3),
                        pooling='avg') 

        DEFAULT_ARGS = {
            "use_bias": False,
            "kernel_regularizer": tf.keras.regularizers.l2()}

        self.projection_head = tf.keras.models.Sequential([
                _dense(**DEFAULT_ARGS)(hidden_dims, name='proj_fc1'), 
                tf.keras.layers.Activation("relu"),
                 _dense(**DEFAULT_ARGS)(projection_dims, name='proj_fc2'),
            ])

    def compile(self, optimizer, loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metrics = metrics

    def call(self, inputs):
        x = self.encoder(inputs, training=False)
        x = self.projection_head(x)

        return x 

    def train_step(self, inputs):
  
        xi = inputs[0]
        xj = inputs[1]

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

            loss, labels, logits = self.loss(zi, zj)

        encoder_params, proj_head_params = self.encoder.trainable_weights, self.projection_head.trainable_weights
        trainable_params = encoder_params + proj_head_params

        grads = tape.gradient(loss, trainable_params)

        self.optimizer.apply_gradients(
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

    def one_step(self, input_shape):
        inputs = tf.zeros(input_shape)

        x = self.encoder(inputs)
        x = self.projection_head(x)

    @property
    def get_all_trainable_params(self):
        encoder_params = self.encoder.trainable_variables

        projection_head_params = self.projection_head.trainable_variables

        return encoder_params + projection_head_params