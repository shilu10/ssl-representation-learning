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




args:
    
     model_type 
     num_classes
     image_size 
     unlabelled_datapath 
     train_datapath


class DataLoader:
    def __init__(self, args, num_workers):
         self.args = args 
         self.num_workers

         self.augmenter = Augment(args)

    def augmentation(self, image):
        augmented_images = [] 
        model_type = self.args.model_type
        
        if model_type in ['simclr', 'mocov1', 'mocov2']:
            # 2 -> two augmented views of images
            for _ in range(2):
                try:

                    if model_type == 'simclr':
                        aug_img = self.augmenter._augment_simclr(image)

                    elif model_type == 'mocov1':
                        aug_img = self.augmenter__augment_mocov1(image)

                    elif model_type == 'mocov2':
                        aug_img = self.augmenter__augment_mocov2(image)

                    augmented_images.append(aug_img)

                except Exception as err:
                    print(err)

            return augmented_images

        return self.augmenter._augment_lincls(img, shape)


    def parse_file(self, file_path, y=None):
         raw = tf.io.read_file(file_path)

         # for supervised learning
         if y is not None:
            return tf.data.Dataset.from_tensors((raw, y))

        # for ssl pretext task 
        return tf.data.Dataset.from_tensors((raw))

    def prepare_images(self, value, label=None):
        shape = tf.image.extract_jpeg_shape(value)
        img = tf.io.decode_jpeg(value, channels=3)
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
            datapath = self.args.unlabelled_datapath
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

        if self.args.model_type != 'lincls':
            dataset = tf.data.Dataset.from_tensor_slices(self.image_file_paths)

        else:
            dataset = tf.data.Dataset.from_tensor_slices(self)