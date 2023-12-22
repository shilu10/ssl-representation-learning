import tensorflow as tf 
from tensorflow.keras.utils import Progbar
from tensorflow import keras 
import os, sys, shutil
from imutils import paths 
from tqdm import tqdm
import numpy as np 
import argparse 
from augment import JigSaw
from typing import Union
from utils import read_image, preprocess_image
from backbone import AlexNet
from datetime import datetime 


AUTO = tf.data.experimental.AUTOTUNE


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--num_classes', type=int, default=1000)
	parser.add_argument('--model_type', type=str, default='jigsaw')
	parser.add_argument('--unlabeled_datapath', type=str, default='./stl10/unlabeled_images/')
	
	parser.add_argument('--num_epochs', type=int, default=100)
	parser.add_argument('--batch_size', type=int, default=32)
	parser.add_argument('--lr', type=float, default=0.001)

	parser.add_argument('--checkpoint', type=str, default="./checkpoint/")
	parser.add_argument('--tensorboard', type=str, default='./logs/')

	parser.add_argument('--gpu', default=0, type=int, help='gpu id')

	parser.add_argument('--use_validation', default=False, type=bool)
	parser.add_argument('--validation_datapath', type=str, default='./stl10/val_images/')

	parser.add_argument('--permutation_arr_path', type=str, default='permutation_max_1000.npy')

	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--grid_size', type=Union(tuple, int), default=(3, 3))

	return parser.parse_args()


def main(args):
	
	# assertion errors
	assert args.use_validation and os.path.exists(args.validation_datapath), "validation_datapath does'nt exists"
	assert os.path.exists(os.permutation_arr_path), "no file or folder exists, use hamming_set.py to initialize the permutation_arr"

	# setting specific gpu in multi-gpu workstation for cuda job.
	if args.gpu is not None:
		os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
		os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	else: 
		print('CPU mode')

	# transformation (jigsaw or rotation)
	jigsaw_transformation = JigSaw(args, permutation_arr)

	# trainloader and val dataloader 
	image_files = imutils.paths.list_images(args.unlabeled_datapath)
	dummy_labels = [_ for _ in range(len(image_files))]

	# dataloader
	dataset = tf.data.Dataset.from_tensor_slices((image_files, dummy_labels))
	dataset = dataset.repeat()
	
	if args.shuffle:
		dataset = dataset.shuffle(len(image_files))

	# for parallel extraction
	dataset = dataset.interleave(read_image, num_parallel_calls=AUTO)

	# for parallel preprocessing
	dataset = dataset.map(lambda x, y: preprocess_image(x, y, jigsaw_transformation))

	dataset = dataset.batch(self.batch_size, drop_remainder=True)
	dataset = dataset.prefetch(AUTO)

	iter_per_epoch = int(len(image_files) / args.batch_size)

	# network 
	network = AlexNet(args.num_classes)

	# optimizer
	optimizer = tf.keras.optimizer.SGD(lr=args.lr,momentum=0.9,weight_decay = 5e-4)

	# criterion
	criterion = tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
	
	# checkpoint 
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
							  optimizer=optimizer, 
							  net=network, 
							  iterator=iter(dataset))

	manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=3)

	ckpt.restore(manager.latest_checkpoint)

	if manager.latest_checkpoint:
		print("Restored from {}".format(manager.latest_checkpoint))

	else:
		print("Initializing network from scratch.")

	# metric trackers
	loss_tracker = tf.keras.metrics.Mean()
	top1_acc = tf.keras.metrics.tf.keras.metrics.TopKSparseCategoricalAccuracy(k=1, 
									name='top_k_categorical_accuracy', dtype=None)
	top5_acc = tf.keras.metrics.tf.keras.metrics.TopKSparseCategoricalAccuracy(k=5, 
									name='top_k_categorical_accuracy', dtype=None)

	# summary writer
	train_log_dir = f'{args.tensorboard}/batch_level/' + datetime.now().strftime("%Y%m%d-%H%M%S") + {model_type} + '/train'
	train_writer = tf.summary.create_file_writer(train_log_dir)

	# metrics name for progressbar
	metrics_names = ['loss', 'top1_acc', 'top5_acc']

	for epoch in rangge(args.num_epochs):
		print("\nepoch {}/{}".format(epoch+1, args.num_epochs))

		# progress bar
		pb_i = Progbar(len(image_files), stateful_metrics=metrics_names)
		
		for step, batch in enumerate(dataset):
			# for infinite dataset
			if step == iter_per_epoch:
				break
			
			result = train(
					network = network,
					batch = batch,
					optimizer = optimizer,
					criterion = criterion,
					top1_acc = top1_acc,
					top5_acc = top5_acc,
					loss_tracker = loss_tracker,
				)

			# batch-level summary writer
			batch_loss = loss_tracker.result()
			batch_top1_acc = top1_acc.result()
			batch_top5_acc = top5_acc.result()
			with train_writer.as_default(step=step):
				tf.summary.scalar('batch_loss', batch_loss)
				tf.summary.scalar('batch_top1_acc', batch_top1_acc)
				tf.summary.scalar('batch_top5_acc', batch_top5_acc)

			# update progress bar
			values=[('loss', batch_loss.numpy()), ('batch_top1_acc', batch_top1_acc.numpy()), ('batch_top5_acc', batch_top5_acc.numpy())]
        
        	pb_i.add(args.batch_size, values=values)

			# validation code goes here

		epoch_loss = loss_tracker.result()
		epoch_top1_acc = top1_acc.result()
		epoch_top5_acc = top5_acc.result()

		# epoch-level summary writer
		with train_writer.as_default(step=epoch):
			tf.summary.scalar('epoch_loss', epoch_loss)
			tf.summary.scalar('epoch_top1_acc', epoch_top1_acc)
			tf.summary.scalar('epoch_top5_acc', epoch_top5_acc)

		# reset the netrics
		loss_tracker.reset_state()
		top5_acc.reset_state()
		top1_acc.reset_state()

		# for checkpoint update
		ckpt.step.assign_add(1)
	    if int(ckpt.step) % 5 == 0:
	      save_path = manager.save()
	      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
	      print("loss {:1.2f}".format(epoch_loss.numpy()))


def train(network, batch, optimizer, criterion, top1_acc, top5_acc, loss_tracker):
	
	inputs, labels = batch 

	with tf.GradientTape() as tape:
		logits = network(inputs, training=True)

		# compute custom loss
		loss = criterion(logits, labels)

	# Compute gradients
    trainable_vars = network.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    # Update weights
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Compute our own metrics
    loss_tracker.update_state(loss)
    top1_acc.update_state(logits, labels)
    top5_acc.update_state(logits, labels)
    
	return loss

def test():
	pass 


