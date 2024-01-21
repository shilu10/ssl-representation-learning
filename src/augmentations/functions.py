import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


class GaussianBlur(tf.keras.layers.Layer):
    def __init__(self, kernel_size=3, sigma=1.0, **kwargs):
        super(GaussianBlur, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.filter = self._gaussian_filter()

    def _gaussian_filter(self):
        kernel = self._gaussian_kernel(self.kernel_size, self.sigma)
        kernel = tf.constant(kernel, dtype=tf.float32)
        kernel = tf.reshape(kernel, (self.kernel_size, 1, 1, 1))
        return tf.tile(kernel, [1, 1, 3, 1])  # Replicate for each channel

    def _gaussian_kernel(self, size, sigma):
        coords = np.arange(size) - (size - 1) / 2.0
        coords = np.exp(-(coords**2) / (2 * sigma**2))
        coords /= np.sum(coords)
        return coords

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, axis=0)
        
        # Apply depthwise convolution
        return tf.nn.depthwise_conv2d(inputs, self.filter, strides=[1, 1, 1, 1], padding='SAME')[0]


class RandomResizedCrop:
    def __init__(self, scales, ratio, **kwargs):
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scales = scales
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def __call__(self, images, training=True):
        if training:
           # batch_size = tf.shape(images)[0]
            height = tf.shape(images)[0]
            width = tf.shape(images)[1]
            channel = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (1,), self.scales[0], self.scales[1]
            )
            random_ratios = tf.exp(
                tf.random.uniform((1,), self.log_ratio[0], self.log_ratio[1])
            )

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((1,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((1,), 0, 1 - new_widths)

            # assemble bounding boxes and crop
            bounding_boxes = tf.stack(
                [
                    height_offsets,
                    width_offsets,
                    height_offsets + new_heights,
                    width_offsets + new_widths,
                ],
                axis=1,
            )
            images = tf.expand_dims(images, axis=0)
            images = tf.image.crop_and_resize(
                images, bounding_boxes, tf.range(1), (height, width)
            )

        return images[0]


class ColorJitter:
	def __init__(self,
				brightness: float,
				contrast: float, 
				saturation: float, 
				hue: float, 
				p: float):

		self.brightness_vals = (max(0, 1 - brightness), 1 + brightness)
		self.contrast_vals = (max(0, 1 - contrast), 1 + contrast)
		self.saturation_vals = (max(0, 1 - saturation), 1 + saturation)

		self.hue_vals = np.random.uniform(low = 0, high = min(hue, 0.5))

	def __call__(self, image):
    	# Random color jittering (strength 0.5)
	    color_jitter = np.random.uniform(low=0.0, high=1.0)
	    if color_jitter < p:
	        image = tf.image.random_brightness(image,
	        								   max_delta=self.brightness_vals[1])

	        image = tf.image.random_contrast(image,
	        								 lower=self.contrast_vals[0],
	        								 upper=self.contrast_vals[1])

	        image = tf.image.random_saturation(image,
	        								  lower=self.saturation_vals[0],
	        								  upper=self.saturation_vals[1])

	        image = tf.image.random_hue(image,
	        						   max_delta=self.hue_vals)

	        image = tf.clip_by_value(image, 0, 255)

	    return image


class GrayScale:

	def __init__(self, p=0.2):
		self.p = p 

	def __call__(self, image):
		# Color dropping
	    color_drop = np.random.uniform(low=0.0, high=1.0)
	    if color_drop < self.p:
	        image = tf.image.rgb_to_grayscale(image)
	        image = tf.tile(image, [1, 1, 3])

	    return image 