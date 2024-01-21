import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
import matplotlib.pyplot as plt 
import random 
from .functions import GaussianBlur, RandomResizedCrop, ColorJitter, GrayScale


class BYOL:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.augmentations.get("scales"),
										ratio=config.augmentations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.augmentations.get('brightness'), 
								contrast=config.augmentations.get('contrast'), 
								saturation=config.augmentations.get('saturation'), 
								hue=config.augmentations.get('hue'),
								p=config.augmentations.get("color_jitter_prob"))

		self.gaussian_blur = GaussianBlur(
								kernel_size=config.augmentations.get('kernel_size'))

		self.grayscale = GrayScale(
								p=config.augmentations.get("grayscale_prob"))

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


class MOCOV1:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.augmentations.get("scales"),
										ratio=config.augmentations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.augmentations.get('brightness'), 
								contrast=config.augmentations.get('contrast'), 
								saturation=config.augmentations.get('saturation'), 
								hue=config.augmentations.get('hue'), 
								p=config.augmentations.get("color_jitter_prob"))

		self.grayscale = GrayScale(
								p=config.augmentations.get("grayscale_prob"))

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