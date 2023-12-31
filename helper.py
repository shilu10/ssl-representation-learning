import tensorflow as tf 
from tensorflow import keras 
from argparse import ArgumentParser
import numpy as np 
from backbone import ResNet50, simple_cnn
import logging
import os, sys, shutil 
from glob import glob 
import imutils 
from imutils import paths 


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--num_epochs', type=int, 
                        default=30, help="Number of epochs to train our model") 

    #parser.add_argument('--steps_per_epoch', type=int, default=200, help=)
    parser.add_argument('--width', type=int, default=128, 
                        help="output shape of neural network, when backbone is simple cnn (not resnet50)")

    parser.add_argument('--backbone', type=str,
                        default='resnet50',help="Type of backbone network to use as encoder, options(resnet model, simplecnn)")

    parser.add_argument('--tensorboard', type=str,
                        default='logs/', help="Whether to use tensorboard summaries")

    parser.add_argument('--history', type=str,
                        default='history', help="Whether to use history callbacks")

    parser.add_argument('--result_path', type=str,
                        default="./results", help="Directory to store all the results")

    parser.add_argument('--resume', type=bool,
                        default=False, help="Whether to resume or not")

    parser.add_argument('--checkpoint', type=str,
                        default='ckpt', help="whether to use tensorboard checkpoint")

    parser.add_argument('--model_type', type=str,
                        default='simclr', help="type of ssl model to train, options(simclr, mocov1, v2)")

    parser.add_argument('--task', type=str, 
                        default='pretraining', help="Type of task, options(pretraining, lincls)")

    parser.add_argument('--unlabeled_datapath', type=str, 
                        default='cifar_dataset/train/', help="Directory path for the unlabeled data")

    parser.add_argument('--train_datapath', type=str, 
                        default='cifar_dataset/train/', help="Directory path for the train data")

    parser.add_argument('--batch_size', type=int, 
                        default=32, help="Batch Size, to be used in the dataloader")

    parser.add_argument('--shuffle', type=bool, 
                        default=True, help="Boolean value tells whether or not to shuffle data in the dataloader")

    parser.add_argument('--contrast', type=int, 
                        default=0.4, help="contrast value to use in data augmentation")

    parser.add_argument('--saturation', type=int, 
                        default=0.4, help="saturation value to use in data augmentation")

    parser.add_argument('--hue', type=int, 
                        default=0.4, help="hue value to use in data augmentation")

    parser.add_argument('--brightness', type=int, 
                        default=0.4, help="brightness value to use in data augmentation")

    parser.add_argument('--img_size', type=int,
                        default=96, help="Image shape(same as input shape for backbone)")

    parser.add_argument('--n_classes', type=int, 
                        default=10, help="Number of classes in the task (dataset)")

    parser.add_argument('--lr_mode', type=str, default="exponential", 
                        help="Type of mode in decay learning rate", 
                        choices=["exponential", "constant", "step", "inverse", "cosine"])

    parser.add_argument('--initial_lr', type=float, 
                        default=0.4, help="Initial Learning Rate value")

    parser.add_argument('--temperature', type=float, 
                        default=0.4, help="Initial Learning Rate value")

    parser.add_argument('--weight_decay', type=float, 
                        default=0.4, help="Decay value to use in decay learning rate")

    parser.add_argument("--gpus", type=str, default='-1')

    parser.add_argument('--pirl_pretext_task', type=str, default="jigsaw", choices=['jigsaw', 'rotation'])

    return parser.parse_args()



def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger.addHandler(screen_handler)
    return logger


def get_session(args):
    assert int(tf.__version__.split('.')[0]) >= 2.0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.gpus != '-1':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)


def create_stamp():
    weekday = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    temp = datetime.now()
    return "{:02d}{:02d}{:02d}_{}_{:02d}_{:02d}_{:02d}".format(
        temp.year % 100,
        temp.month,
        temp.day,
        weekday[temp.weekday()],
        temp.hour,
        temp.minute,
        temp.second,
    )


def get_encoder(enc_type, img_size, width):
    if enc_type=="resnet":
            encoder = ResNet50(
                data_format="channels_last",
                trainable = True,
                include_top = False,
                pooling='avg'
            )

    else:
        encoder = simple_cnn(img_size, width)

    return encoder


class OptionalLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, args, steps_per_epoch, initial_epoch):
        super(OptionalLearningRateSchedule, self).__init__()
        self.args = args
        self.steps_per_epoch = steps_per_epoch
        self.initial_epoch = initial_epoch

        if self.args.lr_mode == 'exponential':
            decay_epochs = [int(e) for e in self.args.lr_interval.split(',')]
            lr_values = [self.args.lr * (self.args.lr_value ** k)for k in range(len(decay_epochs) + 1)]
            self.lr_scheduler = \
                tf.keras.optimizers.schedules.PiecewiseConstantDecay(decay_epochs, lr_values)

        elif self.args.lr_mode == 'cosine':
            self.lr_scheduler = \
                tf.keras.experimental.CosineDecay(self.args.lr, self.args.epochs)

        elif self.args.lr_mode == 'constant':
            self.lr_scheduler = lambda x: self.args.lr
            
    def get_config(self):
        return {
            'steps_per_epoch': self.steps_per_epoch,
            'init_lr': self.args.lr,
            'lr_mode': self.args.lr_mode,
            'lr_value': self.args.lr_value,
            'lr_interval': self.args.lr_interval,}

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        step += self.initial_epoch * self.steps_per_epoch
        lr_epoch = (step / self.steps_per_epoch)
        if self.args.lr_mode == 'constant':
            return self.args.lr
        else:
            return self.lr_scheduler(lr_epoch)


class CNN(tf.keras.Model):
    def __init__(self, input_shape, output_dim):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same',
                                            input_shape=input_shape)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.activation = tf.keras.layers.ReLU()
        self.pool = tf.keras.layers.MaxPooling2D(strides=2, pool_size=2)

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()

        self.conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn4 = tf.keras.layers.BatchNormalization()

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')

        self.f = tf.keras.layers.Dense(units=output_dim, activation=None, name="head_f")
        self.g = tf.keras.layers.Dense(units=output_dim, activation=None, name="head_g")

    # @timeit
    def call(self, x, head, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.bn4(x, training=training)
        x = self.activation(x)
        x = self.pool(x)

        x = self.global_pool(x)

        out = tf.cond(tf.equal(head, 'f'), lambda: self.f(x), lambda: self.g(x))
        # if head == 'f':
        #     out = self.f(x)
        # elif head == 'g':
        #     out = self.g(x)

        return x, out