import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
import imutils, random 
import src.transforms as transforms


class PIRL:
    def __init__(self, config, image_files_path, batch_size=32, split="train", shuffle=True):
        self.config = config 
        self.batch_size = batch_size
        self.split = split
        self.shuffle = shuffle
        self.image_files_path = image_files_path

    def __parse_image(self, index, image_path):
        raw = tf.io.read_file(image_path)
        image = tf.io.decode_jpeg(raw)

        image = tf.cast(image, dtype=tf.float32)
        return index, image 

    def __augment(self, index, image):
        pretext_task_type = self.config.model.get("pretext_task_type")

        augmenter = getattr(transforms, pretext_task_type)

        _, transformed_image = augmenter(self.config)(image)

        image /= 255.0
        transformed_image /= 255.0

        #inputs = {"inputs": (view1, view2)}

        return index, image, transformed_image

    def create_dataset(self):
        image_files_tensor = tf.constant(self.image_files_path)
        indices_tensor = tf.range(len(image_files_tensor))

        dataset = tf.data.Dataset.from_tensor_slices((indices_tensor, image_files_tensor))
        
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