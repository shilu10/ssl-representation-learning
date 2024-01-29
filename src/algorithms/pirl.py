# https://keras.io/examples/vision/nnclr/
from tensorflow import keras 
import tensorflow as tf 
import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import numpy as np 
from src.utils.contrastive_task import _cosine_simililarity_dim1 as sim_func_dim1, _cosine_simililarity_dim2 as sim_func_dim2
from src.utils.contrastive_task import get_negative_mask
from src.networks import contrastive_task as networks
from src import init_memory_bank 
from .common import ContrastiveLearning
from .utils import _dense, _conv2d
from src import memory_bank


class PIRL(ContrastiveLearning):
    """Momentum Contrastive Feature Learning"""
    def __init__(self, config, *args, **kwargs):

        super(PIRL, self).__init__(dynamic=True, *args, **kwargs)
        self.config = config
        img_size = config.model.get('img_size')
        self.projection_dims = config.model.get("projection_dims")
        self.pretext_task_type = config.model.get("pretext_task_type")
        
        self.encoder = getattr(networks, config.networks.get("encoder_type"))(
                                            include_top=False,
                                            input_shape=(None, None, 3),
                                            pooling=None)

        self.f = getattr(networks, config.networks.get("generic_type"))(
                                            self.projection_dims)

        if self.pretext_task_type == "JigSaw":
            n_patches = self.config.transformations.get("n_patches")
            n_patches = n_patches if isinstance(n_patches, tuple) else (n_patches, n_patches)
            self.g = getattr(networks, config.networks.get("transformed_type"))(
                                            encoding_size=self.projection_dims, 
                                            jigsaw_size=n_patches)

        else:
            self.g = getattr(networks, config.networks.get("transformed_type"))(
                                            encoding_size=self.projection_dims)

        self.initialize_memory_bank()

    def compile(self, optimizer, loss, metrics, **kwargs):
        super().compile(**kwargs)
        self.optimizer = optimizer
        self.loss = loss
        self.acc_metrics = metrics

    def train_step(self, inputs):
        indices, original, transformed = inputs

        batch_size = original.shape[0]

        if self.pretext_task_type == "JigSaw":
            transformed = tf.concat([*transformed], axis=0)

        with tf.GradientTape() as tape:
            # original image feats 
            original_image_feats = self.encoder(original)
            original_image_feats = self.f(original_image_feats)

            #transformed image feats
            transformed_image_feats = self.encoder(transformed)
            transformed_image_feats = self.g(transformed_image_feats)

            mem_repr = self.memory_bank.sample_by_indices(indices)
            mem_arr = self.memory_bank.sample_negatives(indices, 1000)

            loss_1 = self.loss(mem_repr, transformed_image_feats, mem_arr)
            loss_2 = self.loss(mem_repr, original_image_feats, mem_arr)

            loss =  tf.reduce_mean(0.5 * loss_1 + (1 - 0.5) * loss_2)

            del mem_arr

        trainable_params = self.get_all_trainable_params

        grads = tape.gradient(loss, trainable_params)

        # update representation memory
        self.memory_bank.update_memory_repr(indices, original_image_feats)

        self.optimizer.apply_gradients(
            zip(
                grads,
                trainable_params
            )
        )

        self.update_contrastive_accuracy(original_image_feats, transformed_image_feats)
        self.update_correlation_accuracy(original_image_feats, transformed_image_feats)

        #accs = [metric.update_state(labels, logits) for metric in self.acc_metrics]

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
        results = {}
        #results = {m.name: m.result() for m in self.acc_metrics}

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

    def initialize_memory_bank(self):
        #self.memory_bank = init_memory_bank.main(self.config)

        self.memory_bank = memory_bank.MemoryBank((100000, 128))

    def one_step(self, input_shape):
        inputs = tf.zeros(input_shape, dtype=tf.float32)

        encoder_out = self.encoder(inputs) 
        f_out = self.f(encoder_out)
        g_out = self.g(encoder_out)

    @property
    def get_all_trainable_params(self):
        encoder_params = self.encoder.trainable_weights
        f_params = self.f.trainable_weights
        g_params = self.g.trainable_weights

        all_trainable_params = encoder_params + f_params + g_params

        return all_trainable_params 

    def save_encoder_weights(self, filepath):
        self.encoder.save_weights(filepath)

    def load_encoder_weights(self, filepath):
        self.encoder.load_weights(filepath, overwrite=True)
