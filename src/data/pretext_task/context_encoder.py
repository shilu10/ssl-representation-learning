import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from imutils import paths 
import imutils 
import datetime
from datetime import datetime


AUTO = tf.data.experimental.AUTOTUNE


class ContextEncoderDataLoader:
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

        raw = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.image.resize(image, (128, 128))
        image /= 255.0
        #image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        #image = tf.cast(image, dtype=tf.float32)

        return image, label

    def create_dataset(self):
        # Convert file paths and labels to TensorFlow tensors
        image_files_tensor = tf.constant(self.image_files_path)
        labels_tensor = tf.constant(self.labels)

        # Create a tf.data.Dataset from the tensors
        dataset = tf.data.Dataset.from_tensor_slices((image_files_tensor, labels_tensor))

        # Shuffle if needed
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_files_path), reshuffle_each_iteration=True)

        # Apply the preprocess_image function to each element of the dataset
        dataset = dataset.map(self.preprocess_image, num_parallel_calls=AUTO)

        # Batch the dataset
        dataset = dataset.batch(self.batch_size, drop_remainder=True)

        dataset = dataset.prefetch(AUTO)

        return dataset