import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
import matplotlib.pyplot as plt 
import random 
from .functions import GaussianBlur, RandomResizedCrop, ColorJitter


class Byol:
	def __init__(self, config):
		self.random_resized_crop = RandomResizedCrop(
										scales=config.augmentations.get("scales"),
										ratio=config.augmentations.get("ratio"))


		self.color_jitter = ColorJitter(
								brightness=config.augmentations.get('brightness'), 
								contrast=config.augmentations.get('contrast'), 
								saturation=config.augmentations.get('saturation'), 
								hue=config.augmentations.get('hue'))

		self.gaussian_blur = GaussianBlur(kernel_size=config.augmentations.get('kernel_size'))

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
