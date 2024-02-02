import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from imutils import paths 
import imutils 
import datetime
from datetime import datetime


AUTO = tf.data.experimental.AUTOTUNE


class ContextPredictionDataLoader:
    def __init__(self, 
    			args, 
    			image_files_path, 
    			labels, 
    			split_type='train', 
    			batch_size=32, 
    			shuffle=True):

        self.args = args
        self.image_files_path = image_files_path
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split_type = split_type

    def preprocess_image(self, image_path, label):
        try:

            patch_dim, gap = self.args.model.get('patch_dim'), self.args.model.get('gap')

            raw = tf.io.read_file(image_path)
            image = tf.image.decode_image(raw, channels=3)
            image = tf.image.resize(image, (64, 64))

            offset_x = image.shape[0] - (patch_dim*3 + gap*2)
            offset_y = image.shape[1] - (patch_dim*3 + gap*2)

            start_grid_x = tf.maximum(0, tf.random.uniform(shape=(), minval=1, maxval=offset_x, dtype=tf.int32))
            start_grid_y = tf.maximum(0, tf.random.uniform(shape=(), minval=1, maxval=offset_y, dtype=tf.int32))

            patch_loc_arr = tf.constant([(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)], dtype=tf.int32)
            loc = tf.random.uniform(shape=(), minval=0, maxval=tf.shape(patch_loc_arr)[0], dtype=tf.int32)
            random_patch_loc = tf.gather(patch_loc_arr, loc)
            tempx, tempy = random_patch_loc

            patch_x_pt = start_grid_x + patch_dim * (tempx-1) + gap * (tempx-1)
            patch_y_pt = start_grid_y + patch_dim * (tempy-1) + gap * (tempy-1)

            random_patch = tf.image.crop_to_bounding_box(image, patch_x_pt, patch_y_pt, patch_dim, patch_dim)
            uniform_patch = tf.image.crop_to_bounding_box(image, start_grid_x, start_grid_y, patch_dim, patch_dim)

            random_patch_label = loc

            uniform_patch = tf.image.resize(uniform_patch, (96, 96))
            random_patch = tf.image.resize(random_patch, (96, 96))

            uniform_patch /= 255.0
            random_patch /= 255.0

            return tf.stack([uniform_patch, random_patch]), random_patch_label

        except Exception as err:
            print("err in train loader context prediction")
            return err 


    def create_dataset(self):
        try:
            # Convert file paths and labels to TensorFlow tensors
            image_files_tensor = tf.constant(self.image_files_path)
            labels_tensor = tf.constant(self.labels)

            # Create a tf.data.Dataset from the tensors
            dataset = tf.data.Dataset.from_tensor_slices((image_files_tensor, labels_tensor))

            # Shuffle if needed
            if self.shuffle:
                dataset = dataset.shuffle(buffer_size=len(self.image_files_path), reshuffle_each_iteration=True)

            # Apply the preprocess_image function to each element of the dataset
            # works with py_function, not with numpy_function
            dataset = dataset.map(lambda x, y: tuple(tf.py_function(self.preprocess_image, [x, y], [tf.float32, tf.int32])))

            # Batch the dataset
            dataset = dataset.batch(self.batch_size, drop_remainder=True)

            dataset = dataset.prefetch(AUTO)

            return dataset

        except Exception as err:
            print("err in create_dataset")
            return err