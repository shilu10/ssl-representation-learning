import tensorflow as tf 
from tensorflow import keras 
import os, sys, shutil
import numpy as np 
from argparse import ArgumentParser

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs 
from tensorflow.keras.layers.experimental import preprocessing

from dataloader import prepare_dataset
from augment import RandomResizedCrop, RandomColorJitter, RandomColorDisortion, GaussianBlur
from models import SimCLR, MoCo, PIRL
from losses import NTXent, InfoNCE
from helper import get_args, get_encoder, get_logger, CNN
from dataloader import DataLoader
from utils import search_same, get_session
from memory_bank import MemoryBank
from pirl_task_models import JigsawTask, GenericTask, CNN


tf.get_logger().setLevel("WARN")  # suppress info-level logs


def main(args):

    #args, initial_epoch = search_same(args)
    initial_epoch = 0

    if initial_epoch == -1:
        # training was already finished!
        return

    #elif initial_epoch == 0:
        # first training or training with snapshot
     #   args.stamp = create_stamp()

   # get_session(args)
    
    logger = get_logger(args.model_type)

    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    #######################
    # DATALOADER
    #######################

    loader = DataLoader(
        args = args,
        batch_size = args.batch_size,
        shuffle = args.shuffle,
        num_workers = 1 ,
    )
    pretraining_data_generator = loader()
    print(pretraining_data_generator)
    n_image_files = loader.num_image_files
    steps_per_epoch = n_image_files / args.batch_size

    logger.info("Loaded pretraining dataloader")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of steps_per_epoch: {steps_per_epoch}")

    ########################
    # ENCODER AND PROJ HEAD
    ########################

    encoder = get_encoder(
                enc_type=args.backbone, 
                img_size=args.img_size,
                width=args.width
            )

    projection_head = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(args.width,)),
                tf.keras.layers.Dense(args.width, activation="relu"),
                tf.keras.layers.Dense(args.width),
            ],
            name="projection_head",
        )

    #####################
    # OPTIMIZER
    #####################

    optimizer = tf.keras.optimizers.Adam()

    ###################
    # CALLBACKS
    ###################

    checkpoint_filepath = args.checkpoint
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='c_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=True,
    )

    tb_callback = tf.keras.callbacks.TensorBoard(args.tensorboard + '/' + args.model_type, 
                                                update_freq=1)

    ###################
    # model
    ###################

    if args.model_type == 'simclr':
        loss = NTXent(batch_size=args.batch_size)
        model = SimCLR(
            encoder = encoder,
            projection_head = projection_head,
        )

    elif args.model_type == "mocov1":
        loss = InfoNCE(temp=0.07)
        model = MoCo(
            encoder = encoder,
            projection_head = projection_head,
            version=args.model_type
        )

    elif args.model_type == "pirl":
        loss = InfoNCE(temp=0.07)

        encoder = CNN(input_shape=(96, 96, 3), output_dim=128)
        f = GenericTask(encoding_size=128)

        if args.pirl_pretext_task == "jigsaw":
            g = JigsawTask(128, (3, 3))

        else:
            g = GenericTask(encoding_size=128)

        memory_bank = MemoryBank(
            shape = (n_image_files, 128)
        )

        memory_bank.initialize(encoder, f, pretraining_data_generator, steps_per_epoch)

        model = PIRL(
            encoder = encoder,
            g = g, 
            f = f,
            memory_bank=memory_bank,
            pretext_task=args.pirl_pretext_task,
        )

    model.compile(
        optimizer=optimizer, 
        loss = loss,
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
    )

    logger.info(f"STARTING TRAINING OF MODEL WITH {args.num_epochs} EPOCHS.")

    history = model.fit(pretraining_data_generator, 
                        epochs=args.num_epochs, 
                        #validation_data=test_dataset, 
                        initial_epoch=initial_epoch,
                        callbacks=[tb_callback, model_checkpoint_callback], 
                        steps_per_epoch=steps_per_epoch)
    


if __name__ == '__main__':
    args = get_args()
    main(args)