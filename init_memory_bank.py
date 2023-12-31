from memory_bank import MemoryBank
import tensorflow as tf 
from tensorflow import keras 
import argparse 
import os, sys, shutil
import numpy as np 
from imutils import paths
from pirl_task_models import GenericTask, CNN


AUTO = tf.data.experimental.AUTOTUNE
mean_std = [[0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]]


def parse_image(indices, image_path):
    raw = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(raw)

    return tf.data.Dataset.from_tensors((indices, image))


def map_func(indices, image):
    image = tf.cast(image, tf.float32)

    mean, std = mean_std
    image /= 255.
    image -= mean
    image /= std

    return indices, image


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--unlabeled_datapath", type=str, default="./stl10/unlabeled_images/")
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--image_shape", type=tuple, default=(96, 96, 3))

    return parser.parse_args()


def main(args):
    datapath = args.unlabeled_datapath
    image_files = list(paths.list_images(datapath))

    dataset_size = len(image_files)
    dataset = tf.data.Dataset.from_tensor_slices(image_files)
    indices = tf.data.Dataset.from_tensor_slices(tf.range(dataset_size))

    dataset = tf.data.Dataset.zip((indices, dataset))
    dataset = dataset.shuffle(dataset_size)

    # parallel extraction
    dataset = dataset.interleave(parse_image, num_parallel_calls=AUTO)
    dataset = dataset.map(map_func, num_parallel_calls=AUTO)

    dataset = dataset.batch(batch_size=10)
    dataset = dataset.prefetch(AUTO)

    steps_per_epoch = int(dataset_size / 10)

    memory_bank = MemoryBank(shape=(dataset_size, args.out_dim), weight=args.weight)
    encoder = CNN(input_shape=args.image_shape, output_dim=args.out_dim)
    f = GenericTask(encoding_size=args.out_dim)

    memory_bank.initialize(encoder=encoder, 
                           f=f, 
                           train_loader=dataset, 
                           steps_per_epoch=steps_per_epoch, 
                           sep_init=True)

    print('Completed the memory bank initialization')

if __name__ == '__main__':
    args = parse_args()

    main(args)