import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from imutils import paths 
import imutils 
import datetime
from datetime import datetime


AUTO = tf.data.experimental.AUTOTUNE


class RotateNetDataLoader(object):
    def __init__(self, 
                image_files_path, 
                labels, 
                rotations=[0, 90, 180, 270], 
                use_all_rotations=False, 
                split_type='train',
                batch_size=32,
                shuffle=True):

        self.image_files_path = image_files_path
        self.labels = labels
        self.rotations = rotations
        self.use_all_rotations = use_all_rotations
        self.split_type = split_type
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.image_files_path))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def parse_file(self, image_path, label=None):
        raw = tf.io.read_file(image_path)

        if label is not None:
          return tf.data.Dataset.from_tensors((raw, label))

        return tf.data.Dataset.from_tensors(raw)

    def preprocess_image(self, value, rotation_index):
        shape = tf.image.extract_jpeg_shape(value)

        image = tf.image.decode_jpeg(value, channels=3)

        # augmentation
        image = self.augmentation(image)
        rotation_value = self.rotations[rotation_index]

        # rotate the image
        transformed_image = tf.image.rot90(image, k=rotation_index)

        return transformed_image, rotation_index

    def augmentation(self, image):
        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        if self.split_type == "train":
            image = self.center_crop(image, 224, 224)

        else:
            image = tf.image.random_crop(image, (224, 224))
            # horizontal flip
            image = tf.image.random_flip_left_right(image)

        return image

    def center_crop(self, image, crop_height, crop_width):
        height, width = tf.shape(image)[0], tf.shape(image)[1]

        # Calculate the crop coordinates
        start_y = (height - crop_height) // 2
        start_x = (width - crop_width) // 2

        # Perform cropping
        cropped_image = tf.image.crop_to_bounding_box(
            image,
            start_y,
            start_x,
            crop_height,
            crop_width
        )

        return cropped_image

    def get_dataset(self):
        # Convert file paths and labels to TensorFlow tensors
        image_files_tensor = tf.constant(self.image_files_path)
        labels_tensor = tf.constant(self.labels)

        if self.use_all_rotations:
            dataset = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor([], dtype=tf.string))
            indices = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor([], dtype=tf.int32))

            dataset = tf.dat.Dataset.zip((dataset, indices))

            for _ in range(len(self.rotations)):

                inner_dataset = tf.data.Dataset.from_tensor_slices((image_files_tensor, labels_tensor))
                dataset = dataset.concatenate(inner_dataset)
        
        else:
            dataset = tf.data.Dataset.from_tensor_slices((image_files_tensor, labels_tensor))

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_files_path), reshuffle_each_iteration=True)

        dataset = dataset.interleave(self.parse_file, num_parallel_calls=AUTO)
        dataset = dataset.map(lambda x, y:  tf.py_function(self.preprocess_image, [x, y], [tf.float32, tf.int32]))
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)

        return dataset