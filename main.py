import tensorflow as tf 
from tensorflow import keras 
import os, sys, shutil
import numpy as np 
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs 
from tensorflow.keras.layers.experimental import preprocessing

from dataloader import prepare_dataset
from augmentations import RandomResizedCrop, RandomColorJitter
from models import SimCLR
from losses import NTXent

tf.get_logger().setLevel("WARN")  # suppress info-level logs


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, default=30) 
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--width', type=int, default=128)

    return parser.parse_args()


def main(args):
    # load STL10 dataset
    batch_size, train_dataset, labeled_train_dataset, test_dataset = prepare_dataset(
        args.steps_per_epoch
    )

    contrastive_augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(96, 96, 3)),
                preprocessing.Rescaling(1 / 255),
                preprocessing.RandomFlip("horizontal"),
                RandomResizedCrop(scale=(0.2, 1.0), ratio=(3 / 4, 4 / 3)),
                RandomColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            ],
            name="contrastive_augmenter",
        )

    classification_augmenter = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(96, 96, 3)),
                preprocessing.Rescaling(1 / 255),
                preprocessing.RandomFlip("horizontal"),
                RandomResizedCrop(scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)),
                RandomColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ],
            name="classification_augmenter",
        )

    encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(96, 96, 3)),
                tf.keras.layers.Conv2D(args.width, kernel_size=3, strides=2, activation="relu"),
                tf.keras.layers.Conv2D(args.width, kernel_size=3, strides=2, activation="relu"),
                tf.keras.layers.Conv2D(args.width, kernel_size=3, strides=2, activation="relu"),
                tf.keras.layers.Conv2D(args.width, kernel_size=3, strides=2, activation="relu"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(args.width, activation="relu"),
            ],
            name="encoder",
        )

    projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(args.width,)),
                tf.keras.layers.Dense(args.width, activation="relu"),
                tf.keras.layers.Dense(args.width),
            ],
            name="projection_head",
        )

    linear_probe = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(args.width,)),
                tf.keras.layers.Dense(10),
            ],
            name="linear_probe",
        )

    contrastive_optimizer = tf.keras.optimizers.Adam()
    probe_optimizer = tf.keras.optimizers.Adam()

    contrastive_loss = NTXent()

    model = SimCLR(
        encoder = encoder,
        projection_head = projection_head,
        contrastive_augmenter = contrastive_augmenter,
        classification_augmenter = classification_augmenter,
        linear_probe = linear_probe
    )

    model.compile(
        contrastive_optimizer=contrastive_optimizer, 
        probe_optimizer = probe_optimizer,
        contrastive_loss = contrastive_loss,
    )

    history = model.fit(train_dataset, epochs=args.num_epochs, validation_data=test_dataset)


if __name__ == '__main__':
    args = get_args()
    main(args)