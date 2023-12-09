import tensorflow as tf 
from tensorflow import keras 
import tensorflow_datasets as tfds


def prepare_dataset(dataset_name="stl10", steps_per_epoch=200):
    unlabeled_batch_size = unlabelled_images // steps_per_epoch
    labeled_batch_size = labelled_train_images // steps_per_epoch
    batch_size = unlabeled_batch_size + labeled_batch_size

    unlabeled_train_dataset = (
        tfds.load(
            dataset_name, split="unlabelled", as_supervised=True, shuffle_files=True
        )
        .shuffle(buffer_size=shuffle_buffer)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )
    labeled_train_dataset = (
        tfds.load(dataset_name, split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=shuffle_buffer)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    test_dataset = (
        tfds.load(dataset_name, split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=AUTOTUNE)

    return batch_size, train_dataset, labeled_train_dataset, test_dataset

