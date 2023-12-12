import tensorflow as tf
import tensorflow_datasets as tfds
from augment import Augment 
import imutils 
from imutils import paths


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
    def __init__(self, args, num_workers):
         self.args = args 
         self.num_workers

         self.image_file_paths = paths.list_images(args.datapath)
         self.augmenter = Augment(args)


    def augment(self, image):
        augmented_images = [] 
        model_type = args.model_type
        
        if model_type in ['simclr', 'mocov1', 'mocov2']:
            # 2 -> two augmented views of images
            for _ in range(2):

                if model_type == 'simclr':
                    aug_img = self.augmenter._augment_simclr(image)

                elif model_type == 'mocov1':
                    aug_img = self.augmenter__augment_mocov1(image)

                elif model_type == 'mocov2':
                    aug_img = self.augmenter__augment_mocov2(image)

                augmented_images.append(aug_img)

            return augmented_images


    def parse_file(self, file_path, y=None):
         raw = tf.io.read_file(file_path)

         # for supervised learning
         if y is not None:
            return tf.data.Dataset.from_tensors((raw, y))

        # for ssl pretext task 
        return tf.data.Dataset.from_tensors((raw))

    def prepare_images(self):
        pass 

    def __call__(self):
        pass 