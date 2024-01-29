import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 


class ContrastiveLearning(tf.keras.models.Model):
	def __init__(self, *args, **kwargs):
		super(ContrastiveLearning, self).__init__(*args, **kwargs)
		# metric function 
		self.contrastive_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
		self.correlation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
 

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
            tf.matmul(features_1, features_2, transpose_a=True) / batch_size)

		feature_dim = tf.shape(features_1)[1]
		correlation_labels = tf.range(feature_dim)

		self.correlation_accuracy.update_state(
            tf.concat([correlation_labels, correlation_labels], axis=0),
            tf.concat([cross_correlation, tf.transpose(cross_correlation)], axis=0),
        )
     
    def save_encoder_weights(self, filepath):
    	pass 

    def laod_encoder_weights(self, filepath):
    	pass 
