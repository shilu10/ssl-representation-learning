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


class MoCoV1:
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


class MoCoV2:
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


class SimCLR:
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

		# random horizontal flip
		image = tf.image.random_flip_left_right(image)

		# random color jitter with random grayscale
		image = self.color_jitter(image)

		# random grayscale
		image = self.grayscale(image)

		return image 


# pirl 

mean_std = [[0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]]


def jigsaw(img):
    mean, std = mean_std
    img = tf.cast(img, tf.float32)
    img /= 255.
    #img -= mean
    #img /= std

    copy_img = tf.image.resize(img, (225, 225), method='bilinear')

    imgclips = []
    for i in range(3):
        for j in range(3):
            clip = copy_img[i * 75: (i + 1) * 75, j * 75: (j + 1) * 75, :]
            randomx = tf.experimental.numpy.random.randint(0, 10)
            randomy = tf.experimental.numpy.random.randint(0, 10)
            clip = clip[randomx: randomx+64, randomy:randomy+64, :]

            imgclips.append(clip)

    imgclips = tf.convert_to_tensor(imgclips)
    imgclips = tf.random.shuffle(imgclips)

    return img, imgclips


class JigSaw(object):
    def __init__(self, n_patches):
        self.n_patches = n_patches
    
    def __call__(self, img):
        assert isinstance(img, tf.Tensor)
        img = tf.expand_dims(img, axis=0)
        transformed_image = img.numpy()  # Convert tf.Tensor to NumPy array
        batch_size, img_size_1, img_size_2, channels = transformed_image.shape
        self.patch_size_1 = img_size_1 // self.n_patches[0]
        self.patch_size_2 = img_size_2 // self.n_patches[1]

        transformed_image = tf.image.extract_patches(
            transformed_image,
            sizes=[1, self.patch_size_1, self.patch_size_2, 1],
            strides=[1, self.patch_size_1, self.patch_size_2, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )

        transformed_image = tf.reshape(transformed_image, [batch_size, -1, self.patch_size_1, self.patch_size_2, channels])

        rand_perm = tf.random.shuffle(tf.range(tf.shape(transformed_image)[1]))
        transformed_image = tf.gather(transformed_image, rand_perm, axis=1)

        # Convert NumPy array back to tf.Tensor
        img = tf.convert_to_tensor(img)
        transformed_image = tf.convert_to_tensor(transformed_image)

        return img[0], transformed_image[0]


class Rotate(object):
    def __init__(self, num_positions=4, return_image=False):
        self.return_image = return_image
        self.num_positions = num_positions

    def __call__(self, img):
        image = img
        ind = tf.random.uniform(shape=(), minval=0, maxval=self.num_positions, dtype=tf.int32)

        transformed_image = tf.image.rot90(image, k=ind)

        if self.return_image:
            return img, transformed_image

        return transformed_image
