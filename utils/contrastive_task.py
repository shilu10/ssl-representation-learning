import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil


cosine_sim_1d = tf.keras.metrics.CosineSimilarity(axis=1)
cosine_sim_2d = tf.keras.metrics.CosineSimilarity(axis=2)


AUTO = tf.data.experimental.AUTOTUNE


def convert_x_to_y_image_format(dir_path, x_format='png', y_format='jpg'):
    files = glob('/kaggle/input/cifar10/cifar10/train/**/*.png', recursive=True)
    for file in files:
        # Load .png image
        image = cv2.imread(file)

        # Save .jpg image
        splitted = file.split('/')
        dir_name = splitted[-2]
        file_name = splitted[-1]

        #out = cv2.imwrite(j[:-3] + 'jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        out = cv2.imwrite(f"cifar10/{dir_name}/{file_name[: -3]}" + 'jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


## negative mask used in simclr loss
def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def _cosine_simililarity_dim1(x, y):
    v = tf.abs(tf.keras.losses.cosine_similarity(x, y, axis=1))
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = tf.abs(tf.keras.losses.cosine_similarity(tf.expand_dims(x, 1), tf.expand_dims(y, 0), axis=2))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v