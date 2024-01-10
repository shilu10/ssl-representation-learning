from argparse import ArgumentParser
import numpy as np 
import logging
import os, sys, shutil 
from glob import glob 
import imutils 
from imutils import paths 


def  main_parse_args():
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


def main_pretext_parse_args():

    parser = ArgumentParser()
    # model, save, load args
    parser.add_argument('--checkpoint', 
                        type=str, 
                        default="./checkpoint/", 
                        help='checkpoint directory path')

    parser.add_argument('--tensorboard', 
                        type=str, 
                        default='./logs/', 
                        help="tensorboard directory path")

    parser.add_argument('--chkpt_step', 
                        type=int, 
                        default=5, 
                        help='how frequently to save checkpoint')

    parser.add_argument('--gpu',
                        default=0, 
                        type=int, 
                        help='gpu id')

    #parser.add_argument('--num_classes', type=int, default=1000)
    # model type and dataloader args
    parser.add_argument('--pretext_task_type', 
                        type=str, 
                        default='jigsaw', 
                        choices=['jigsaw', 'rotation_prediction', 'context_prediction', 'context_encoder'],
                        help='type of pretext task')

    parser.add_argument('--unlabeled_datapath', 
                        type=str, 
                        default='./stl10/unlabeled_images/',
                        help='directory path to unlabeled data')
    
    # model training args
    parser.add_argument('--num_epochs',
                       type=int, 
                       default=100, 
                       help='number of epoch to train a model')

    parser.add_argument('--batch_size', 
                        type=int, 
                        default=32, 
                        help="number of batch")
    #parser.add_argument('--lr', type=float, default=0.001)

    parser.add_argument('--use_validation', 
                        default=False,
                        type=bool, 
                        help='to use validation or not, if True it splits the unlabeled data into train and val')

    parser.add_argument('--val_split_size', 
                        type=float, default=0.2, 
                        help="amount of data need for validation")

    parser.add_argument('--shuffle',
                       type=bool, 
                       default=True, 
                       help='whether or not to shuffle the dataset')

    #parser.add_argument('--permutation_arr_path', type=str, default='permutation_max_1000.npy')

    #parser.add_argument('--shuffle', type=bool, default=True)
    #parser.add_argument('--grid_size', type=Union[tuple, int], default=(3, 3))

    

    #parser.add_argument('--use_all_rotations', type=bool, default=False)

    #parser.add_argument('--patch_dim', type=int, default=15)
    #parser.add_argument('--gap', type=int, default=2)

    # config
    parser.add_argument('--config_path', 
                        type=str, 
                        default='config/', 
                        help='config path for specific pretext task')

    return parser.parse_args()