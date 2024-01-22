import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
import matplotlib.pyplot as plt 
import random 
from .functions import GaussianBlur, RandomResizedCrop, ColorJitter, GrayScale, JigSaw, Rotate


class BYOL:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.transformations.get("scales"),
										ratio=config.transformations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.transformations.get('brightness'), 
								contrast=config.transformations.get('contrast'), 
								saturation=config.transformations.get('saturation'), 
								hue=config.transformations.get('hue'),
								p=config.transformations.get("color_jitter_prob"))

		self.gaussian_blur = GaussianBlur(
								kernel_size=config.transformations.get('kernel_size'))

		self.grayscale = GrayScale(
								p=config.transformations.get("grayscale_prob"))

	def transform(self, image):
		# random resized crop
		image = self.random_resized_crop(image)

		# random horizontal flip
		image = tf.image.random_flip_left_right(image)

		# random color jitter with random grayscale
		image = self.color_jitter(image)

		# gaussian blur
		image = self.gaussian_blur(image)

		return image 


class MoCoV1:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.transformations.get("scales"),
										ratio=config.transformations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.transformations.get('brightness'), 
								contrast=config.transformations.get('contrast'), 
								saturation=config.transformations.get('saturation'), 
								hue=config.transformations.get('hue'), 
								p=config.transformations.get("color_jitter_prob"))

		self.grayscale = GrayScale(
								p=config.transformations.get("grayscale_prob"))

	def transform(self, image):
		# random resized crop
		image = self.random_resized_crop(image)

		# random grayscale
		image = self.grayscale(image)

		# random color jitter with random grayscale
		image = self.color_jitter(image)

		# random horizontal flip
		image = tf.image.random_flip_left_right(image)

		return image 


class MoCoV2:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.transformations.get("scales"),
										ratio=config.transformations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.transformations.get('brightness'), 
								contrast=config.transformations.get('contrast'), 
								saturation=config.transformations.get('saturation'), 
								hue=config.transformations.get('hue'), 
								p=config.transformations.get("color_jitter_prob"))

		self.grayscale = GrayScale(
								p=config.transformations.get("grayscale_prob"))

	def transform(self, image):
		# random resized crop
		image = self.random_resized_crop(image)

		# random grayscale
		image = self.grayscale(image)

		# random color jitter with random grayscale
		image = self.color_jitter(image)

		# random horizontal flip
		image = tf.image.random_flip_left_right(image)

		return image 


class SimCLR:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.transformations.get("scales"),
										ratio=config.transformations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.transformations.get('brightness'), 
								contrast=config.transformations.get('contrast'), 
								saturation=config.transformations.get('saturation'), 
								hue=config.transformations.get('hue'), 
								p=config.transformations.get("color_jitter_prob"))

		self.grayscale = GrayScale(
								p=config.transformations.get("grayscale_prob"))

	def transform(self, image):
		# random resized crop
		image = self.random_resized_crop(image)

		# random horizontal flip
		image = tf.image.random_flip_left_right(image)

		# random color jitter with random grayscale
		image = self.color_jitter(image)

		# random grayscale
		image = self.grayscale(image)

		return image 



