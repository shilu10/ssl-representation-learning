import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
import imutils, random 
from src.augmentations import contrastive_task as augments


class Common:
	def __init__(self, config, image_files_path, batch_size=32, split="train", shuffle=True):
		self.config = config 
		self.batch_size = batch_size
		self.split = split
		self.shuffle = shuffle
		self.image_files_path = image_files_path

	def __parse_image(self, image_path):
		raw = tf.io.read_file(image_path)
		image = tf.io.decode_jpeg(raw)

		image = tf.cast(image, dtype=tf.float32)
		return image 

	def __augment(self, image):
		augmenter = getattr(augments, self.config.dataloader.get("augmentations_type"))
		augmenter = augmenter(self.config)

		view1 = augmenter.transform(image)
		view2 = augmenter.transform(image)

		#inputs = {"inputs": (view1, view2)}

		return view1, view2 

	def create_dataset(self):
		image_files_tensor = tf.constant(self.image_files_path)

		dataset = tf.data.Dataset.from_tensor_slices(image_files_tensor)
		
		# Shuffle if needed
		if self.shuffle:
			dataset = dataset.shuffle(buffer_size=len(self.image_files_path), reshuffle_each_iteration=True)

		# parallely parse image
		dataset = dataset.map(self.__parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)


		# parallely augment
		dataset = dataset.map(self.__augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)

		# batch
		dataset = dataset.batch(self.batch_size, drop_remainder=True)


		# prefetch
		dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)


		return dataset