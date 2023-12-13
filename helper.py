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
                        help="output shape of neural network, when backbone is simple cnn (not resnet50)", 
                        choices=['simplecnn', 'resnet50'])

    parser.add_argument('--backbone', type=str,
                        default='resnet50',help="Type of backbone network to use as encoder, options(resnet model, simplecnn)")

    parser.add_argument('--tensorboard', type=str,
                        default='./logs',help="directory for tensorboard summaries")

    parser.add_argument('--checkpoint', type=str,
                        default='./tmp/ckpt/model.h5', help="directory  for tensorboard checkpoint")

    parser.add_argument('--model_type', type=str,
                        default='simclr', help="type of ssl model to train, options(simclr, mocov1, v2)")

    parser.add_argument('--task', type=str, 
                        default='pretraining', help="Type of task, options(pretraining, lincls)")

    parser.add_argument('--unlabeled_data_path', type=str, 
                        default='cifar_dataset/train/', help="Directory path for the unlabeled data")

    parser.add_argument('--train_data_path', type=str, 
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

    return parser.parse_args()


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


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


def get_encoder(enc_type, img_size):
    if enc_type=="resnet":
            encoder = ResNet50(
                data_format="channels_last",
                trainable = True,
                include_top = False,
                pooling='avg'
            )

    else:
        encoder = simple_cnn(img_size)

    return encoder

