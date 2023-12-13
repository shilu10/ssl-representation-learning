import tensorflow as tf
import tensorflow_datasets as tfds
from augment import Augment 
import imutils 
from imutils import paths
import os, sys, shutil
import numpy as np 


AUTO = tf.data.experimental.AUTOTUNE


def prepare_dataset(steps_per_epoch):
    # labeled and unlabeled samples are loaded synchronously
    # with batch sizes selected accordingly
    unlabeled_batch_size = 100000 // steps_per_epoch
    labeled_batch_size = 5000 // steps_per_epoch
    batch_size = unlabeled_batch_size + labeled_batch_size
    print(
        "batch size is {} (unlabeled) + {} (labeled)".format(
            unlabeled_batch_size, labeled_batch_size
        )
    )

    unlabeled_train_dataset = (
        tfds.load("stl10", split="unlabelled", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .batch(unlabeled_batch_size, drop_remainder=True)
    )
    labeled_train_dataset = (
        tfds.load("stl10", split="train", as_supervised=True, shuffle_files=True)
        .shuffle(buffer_size=5000)
        .batch(labeled_batch_size, drop_remainder=True)
    )
    test_dataset = (
        tfds.load("stl10", split="test", as_supervised=True)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    )

    # labeled and unlabeled datasets are zipped together
    train_dataset = tf.data.Dataset.zip(
        (unlabeled_train_dataset, labeled_train_dataset)
    ).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return batch_size, train_dataset, labeled_train_dataset, test_dataset



class DataLoader:
    def __init__(self, args, batch_size, shuffle, num_workers):
         self.args = args 
         self.num_workers = num_workers
         self.shuffle = shuffle
         self.batch_size=batch_size
         self.num_image_files = None

         self.augmenter = Augment(args)

    def augmentation(self, image, shape):
        augmented_images = [] 
        model_type = self.args.model_type
        
        if self.args.task == "pretraining":
            # 2 -> two augmented views of images
            for _ in range(2):
                try:

                    if model_type == 'simclr':
                        radius = np.random.choice([3, 5])
                        aug_img = self.augmenter._augment_simclr(image, shape, radius=radius)

                    elif model_type == 'mocov1':
                        aug_img = self.augmenter._augment_mocov1(image, shape)

                    elif model_type == 'mocov2':
                        aug_img = self.augmenter._augment_mocov2(image, shape)

                    augmented_images.append(aug_img)

                except Exception as err:
                    print(err)

            return augmented_images

        return self.augmenter._augment_lincls(image, shape)


    def parse_file(self, file_path, y=None):
        raw = tf.io.read_file(file_path)

         # for supervised learning
        if y is not None:
            return tf.data.Dataset.from_tensors((raw, y))

        # for ssl pretext task 
        return tf.data.Dataset.from_tensors(raw)

    def prepare_images(self, value, label=None):
        shape = tf.image.extract_jpeg_shape(value)
        #shape = tf.shape(value)
        img = tf.io.decode_png(value, channels=3)
        if label is None:
            # moco
            query, key = self.augmentation(img, shape)
            inputs = {'query': query, 'key': key}

            return inputs
            
        else:
            # lincls
            inputs = self.augmentation(img, shape)
            labels = tf.one_hot(label, self.args.classes)
            return (inputs, labels)

    def prepare_files(self, mode='unlabeled'):
        if mode == 'unlabeled':
            datapath = self.args.unlabeled_datapath
            image_file_paths = list(paths.list_images(datapath))

            return image_file_paths

        else:
            datapath = self.args.train_datapath
            all_classes = [(cls_name, indx) for indx, cls_name in enumerate(os.listdir(datapath))]
            all_classes_dict = dict(all_classes)

            image_file_paths = list(paths.list_images(datapath))
            label_list = [all_classes_dict[image_path.split('/')[-2]] for image_path in image_file_paths]

            return image_file_paths, label_list

    def __call__(self):

        if self.args.task == 'lincls':
            image_file_paths, label_list = self.prepare_files(mode="labeled")

            dataset = tf.data.Dataset.from_tensor_slices((image_file_paths, label_list))

        else:
            image_file_paths = self.prepare_files(mode='unlabeled')
            dataset = tf.data.Dataset.from_tensor_slices(image_file_paths)

        # for calculating steps_per_epoch
        self.num_image_files = len(image_file_paths)
        
        dataset = dataset.repeat()
        if self.shuffle:
            dataset = dataset.shuffle(len(image_file_paths))


        dataset = dataset.interleave(self.parse_file, num_parallel_calls=AUTO)
        dataset = dataset.map(self.prepare_images, num_parallel_calls=AUTO)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(AUTO)

        return dataset 