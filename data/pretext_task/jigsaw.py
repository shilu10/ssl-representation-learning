import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil 
from imutils import paths 
import imutils 
import datetime
from datetime import datetime


AUTO = tf.data.experimental.AUTOTUNE


class JigSawDataLoader:
    def __init__(self,
                args, 
                image_files_path, 
                labels, 
                split_type='train',
                batch_size=32, 
                shuffle=True):

        self.args = args 
        self.permutations = self.__retrive_permutations(args.num_classes, args.permutation_path)
        self.num_classes = args.num_classes
        self.split_type = split_type

        self.image_files_path = image_files_path
        self.labels = labels 
        self.batch_size = batch_size
        self.shuffle = shuffle

    def preprocess_image(self, image_path, label):
        value = tf.io.read_file(image_path)
        shape = tf.image.extract_jpeg_shape(value)

        image = tf.image.decode_jpeg(value, channels=3)

        result = tf.py_function(self.transform, [image, label], [tf.float32, tf.int32])

        tiles, perm_label = result

        return tiles, perm_label 

    def transform(self, image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.
        #image -= mean
        #image /= std

        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        image = tf.clip_by_value(image, 0, 1)

        cropped_image = tf.image.random_crop(image, size=(225, 225, 3))

        # grid size or grid dim
        grid_size = self.args.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        height, width, channels = cropped_image.shape
        
        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                coordinates.append((i * height // grid_size[0], j * width // grid_size[1]))

        grids = []
        for coordinate in coordinates:
            grid = cropped_image[coordinate[0]:coordinate[0] + height // grid_size[0], coordinate[1]:coordinate[1] + width // grid_size[1], :]
           
            randomx = tf.experimental.numpy.random.randint(0, 10)
            randomy = tf.experimental.numpy.random.randint(0, 10)
            #clip = grid[randomx: randomx+64, randomy:randomy+64, :]
            grids.append(grid)
            
        grids = tf.convert_to_tensor(grids)

        # extract 65*65 tile from the grid
        n_grid = grid_size[0] * grid_size[1]

        #print(grids.shape)
        tiles = tf.image.random_crop(grids, (grids.shape[0], 64, 64, 3))
        
        r_index = tf.random.uniform([], minval=0, maxval=self.num_classes, dtype=tf.dtypes.int32)
        selected_permutation = self.permutations[r_index.numpy()]

        shuffled_tiles = tf.gather(tiles, selected_permutation, axis=0)

        return shuffled_tiles, r_index


    def __retrive_permutations(self, num_classes, permutation_path):
        all_perm = np.load(permutation_path)

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def create_dataset(self):
        
        # Convert file paths and labels to TensorFlow tensors
        image_files_tensor = tf.constant(self.image_files_path)
        labels_tensor = tf.constant(self.labels)

        # Create a tf.data.Dataset from the tensors
        dataset = tf.data.Dataset.from_tensor_slices((image_files_tensor, labels_tensor))   
        
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.image_files_path), reshuffle_each_iteration=True)


        # for parallel preprocessing
        dataset = dataset.map(lambda x, y:  tf.py_function(self.preprocess_image, [x, y], [tf.float32, tf.int32]))

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)

        return dataset