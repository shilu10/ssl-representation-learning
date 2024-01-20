import tensorflow as tf 
from tensorflow import keras 
import numpy as np 


# the implementation of these image augmentations follow the torchvision library:
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/functional_tensor.py


mean_std = [[0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]]


class GaussianBlur(tf.keras.layers.Layer):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def call(self, image, training=True):
        if training:
            #sample = np.array(image)

            # blur the image with a 50% chance
            rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
            if rand_ < 0.5:

                sigma = (self.max - self.min) * np.random.random_sample() + self.min
                #image = cv2.GaussianBlur(image, (self.kernel_size, self.kernel_size), sigma)
                image = tfa.image.gaussian_filter2d(image, sigma=sigma)

        return image


class RandomColorDisortion(tf.keras.layers.Layer):
    def __init__(self, s=1.0, **kwargs):
        super().__init__(**kwargs)
        self.s = s 
    
    # image is a tensor with value range in [0, 1].
    # s is the strength of color distortion.

    def color_jitter(self, x):
        # one can also shuffle the order of following augmentations
        # each time they are applied.
        x = tf.image.random_brightness(x, max_delta=0.8 * self.s)
        x = tf.image.random_contrast(x, lower=1 - 0.8 * self.s, upper=1 + 0.8 * self.s)
        x = tf.image.random_saturation(x, lower=1 - 0.8 * self.s, upper=1 + 0.8 * self.s)
        x = tf.image.random_hue(x, max_delta=0.2 * self.s)
        x = tf.clip_by_value(x, 0, 1)
        return x

    def color_drop(self, x):
        x = tf.image.rgb_to_grayscale(x)
        x = tf.tile(x, [1, 1, 1, 3])
        return x

    def call(self, image, training=True):
        if training:
            rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
            # randomly apply transformation with probability p.
            if rand_ < 0.8:
                image = self.color_jitter(image)

            rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
            if rand_ < 0.2:
                image = self.color_drop(image)
        return image


# crops and resizes part of the image to the original resolutions
class RandomResizedCrop(layers.Layer):
    def __init__(self, scale, ratio, **kwargs):
        super().__init__(**kwargs)
        # area-range of the cropped part: (min area, max area), uniform sampling
        self.scale = scale
        # aspect-ratio-range of the cropped part: (log min ratio, log max ratio), log-uniform sampling
        self.log_ratio = (tf.math.log(ratio[0]), tf.math.log(ratio[1]))

    def call(self, images, training=True):
        if training:
            batch_size = tf.shape(images)[0]
            height = tf.shape(images)[1]
            width = tf.shape(images)[2]

            # independently sampled scales and ratios for every image in the batch
            random_scales = tf.random.uniform(
                (batch_size,), self.scale[0], self.scale[1]
            )
            random_ratios = tf.exp(
                tf.random.uniform((batch_size,), self.log_ratio[0], self.log_ratio[1])
            )

            # corresponding height and widths, clipped to fit in the image
            new_heights = tf.clip_by_value(tf.sqrt(random_scales / random_ratios), 0, 1)
            new_widths = tf.clip_by_value(tf.sqrt(random_scales * random_ratios), 0, 1)

            # random anchors for the crop bounding boxes
            height_offsets = tf.random.uniform((batch_size,), 0, 1 - new_heights)
            width_offsets = tf.random.uniform((batch_size,), 0, 1 - new_widths)

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
            images = tf.image.crop_and_resize(
                images, bounding_boxes, tf.range(batch_size), (height, width)
            )

        return images


# distorts the color distibutions of images
class RandomColorJitter(layers.Layer):
    def __init__(self, brightness, contrast, saturation, hue, **kwargs):
        super().__init__(**kwargs)

        # color jitter ranges: (min jitter strength, max jitter strength)
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

        # list of applicable color augmentations
        self.color_augmentations = [
            self.random_brightness,
            self.random_contrast,
            self.random_saturation,
            self.random_hue,
        ]

        # the tf.image.random_[brightness, contrast, saturation, hue] operations
        # cannot be used here, as they transform a batch of images in the same way

    def blend(self, images_1, images_2, ratios):
        # linear interpolation between two images, with values clipped to the valid range
        return tf.clip_by_value(ratios * images_1 + (1.0 - ratios) * images_2, 0, 1)

    def random_brightness(self, images):
        # random interpolation/extrapolation between the image and darkness
        return self.blend(
            images,
            0,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.brightness, 1 + self.brightness
            ),
        )

    def random_contrast(self, images):
        # random interpolation/extrapolation between the image and its mean intensity value
        mean = tf.reduce_mean(
            tf.image.rgb_to_grayscale(images), axis=(1, 2), keepdims=True
        )
        return self.blend(
            images,
            mean,
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.contrast, 1 + self.contrast
            ),
        )

    def random_saturation(self, images):
        # random interpolation/extrapolation between the image and its grayscale counterpart
        return self.blend(
            images,
            tf.image.rgb_to_grayscale(images),
            tf.random.uniform(
                (tf.shape(images)[0], 1, 1, 1), 1 - self.saturation, 1 + self.saturation
            ),
        )

    def random_hue(self, images):
        # random shift in hue in hsv colorspace
        images = tf.image.rgb_to_hsv(images)
        images += tf.random.uniform(
            (tf.shape(images)[0], 1, 1, 3), (-self.hue, 0, 0), (self.hue, 0, 0)
        )
        # tf.math.floormod(images, 1.0) should be used here, however in introduces artifacts
        images = tf.where(images < 0.0, images + 1.0, images)
        images = tf.where(images > 1.0, images - 1.0, images)
        images = tf.image.hsv_to_rgb(images)
        return images

    def call(self, images, training=True):
        if training:
            # applies color augmentations in random order
            rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
            # randomly apply transformation with probability p.
            if rand_ < 0.8:
                for color_augmentation in random.sample(self.color_augmentations, 4):
                    images = color_augmentation(images)
        return images
