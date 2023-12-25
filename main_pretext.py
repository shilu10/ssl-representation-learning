import tensorflow as tf 
from tensorflow.keras.utils import Progbar
from tensorflow import keras 
import os, sys, shutil, random
from imutils import paths 
from tqdm import tqdm
import numpy as np 
import argparse 
from augment import JigSaw
from typing import Union
from utils import read_image, preprocess_image, PretextTaskDataGenerator, DataLoader, PretextTaskDataGenerator1
from backbone import AlexNet, AlexnetV1, Network
from datetime import datetime 
import itertools


AUTO = tf.data.experimental.AUTOTUNE


def parse_args():

	parser = argparse.ArgumentParser()
	parser.add_argument('--chkpt_step', type=int, default=5)

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
	parser.add_argument('--val_split_size', type=float, default=0.5)

	parser.add_argument('--permutation_arr_path', type=str, default='permutation_max_1000.npy')

	parser.add_argument('--shuffle', type=bool, default=True)
	parser.add_argument('--grid_size', type=Union[tuple, int], default=(3, 3))

	parser.add_argument('--pretext_task_type', type=str, default='jigsaw', choices=['jigsaw', 'rotation'])

	return parser.parse_args()


def main(args):
	
	# assertion errors
	#if args.pretext_task_type == 'jigsaw':
	#	permutation_arr_path = args.permutation_arr_path
	#	permutation_arr_path_n_classes = permutation_arr_path.split('.')[0].split('_')[-1]
	#	assert permutation_arr_path_n_classes == args.num_classes, "permutation_arr_path mismatch with num_classes"

	assert os.path.exists(args.unlabeled_datapath), f"no file or folder exists at {args.unlabeled_datapath}"
	assert os.path.exists(args.permutation_arr_path), "no file or folder exists, use hamming_set.py to initialize the permutation_arr"

	# setting specific gpu in multi-gpu workstation for cuda job.
	#if args.gpu is not None:
	#	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	#	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	#else: 
	#	print('CPU mode')


	# trainloader and val dataloader 
	image_files = list(paths.list_images(args.unlabeled_datapath))
	train_image_files = image_files

	if args.use_validation:
		num_val_images = int(len(image_files) * args.val_split_size)

		validation_image_files = image_files[: num_val_images]
		train_image_files = image_files[num_val_images+1: ]
		
	train_dataset = PretextTaskDataGenerator(args, train_image_files, args.batch_size, args.shuffle, args.num_classes, args.permutation_arr_path)

	if args.use_validation:
		val_dataset = PretextTaskDataGenerator(args, validation_image_files, args.batch_size, args.shuffle, args.num_classes, args.permutation_arr_path)

	'''
	# Apply optimizations
	dataset = tf.data.Dataset.from_generator(
		generator=lambda: iter(data_loader),
		output_signature=(
			tf.TensorSpec(shape=(None, None, None, None, 3), dtype=tf.float32),
			tf.TensorSpec(shape=(None,), dtype=tf.int32)
		)
	)

	dataset = dataset.shuffle(buffer_size=10000)
	dataset = dataset.batch(args.batch_size)
	dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
	'''

	iter_per_epoch = int(len(image_files) / args.batch_size)

	# network 
	network = Network(args.num_classes)

	# optimizer
	optimizer = tf.keras.optimizers.Adam()

	# criterion
	criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	
	# checkpoint 
	ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
							  optimizer=optimizer, 
							  net=network)
							 # iterator=iter(dataset))

	manager = tf.train.CheckpointManager(ckpt, args.checkpoint, max_to_keep=3)

	ckpt.restore(manager.latest_checkpoint)

	if manager.latest_checkpoint:
		print("Restored from {}".format(manager.latest_checkpoint))

	else:
		print("Initializing network from scratch.")

	# metric trackers
	loss_tracker = tf.keras.metrics.Mean()
	top1_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, 
									name='top_k_categorical_accuracy', dtype=None)
	top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, 
									name='top_k_categorical_accuracy', dtype=None)

	# val trackers
	val_loss_tracker = tf.keras.metrics.Mean()
	val_top1_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, 
									name='top_k_categorical_accuracy', dtype=None)
	val_top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, 
									name='top_k_categorical_accuracy', dtype=None)

	# summary writer
	common_log_parent_path = f'{args.tensorboard}/batch_level/' + datetime.now().strftime("%Y%m%d-%H%M%S") + args.model_type
	train_log_dir = common_log_parent_path + '/train'
	train_writer = tf.summary.create_file_writer(train_log_dir)

	if args.use_validation:
		val_log_dir = common_log_parent_path + '/val'
		val_writer = tf.summary.create_file_writer(val_log_dir)

	# metrics name for progressbar
	stateful_metrics = []
	metrics_names = ['loss', 'top1_acc', 'top5_acc']
	stateful_metrics += metrics_names

	if args.use_validation:
		val_metrics_names = ['val_loss', 'val_top1_acc', 'val_top5_acc']
		stateful_metrics += val_metrics_names

	# Set up Keras progress bar
	progbar = tf.keras.utils.Progbar(iter_per_epoch, stateful_metrics=stateful_metrics)

	for epoch in range(args.num_epochs):
		print("\nepoch {}/{}".format(epoch+1, args.num_epochs))

		# train step
		for step, batch in enumerate(train_dataset):

			result = train(
					network = network,
					batch = batch,
					optimizer = optimizer,
					criterion = criterion,
					top1_acc = top1_acc,
					top5_acc = top5_acc,
					loss_tracker = loss_tracker,
				)

			# metrics
			batch_loss = loss_tracker.result()
			batch_top1_acc = top1_acc.result()
			batch_top5_acc = top5_acc.result()

			# Update Keras progress bar
			values = [('loss', batch_loss), ('top1_acc', batch_top1_acc), ('top5_acc', batch_top5_acc)]
			progbar.update(step + 1, values=values)	

			# train batch summaries
			with train_writer.as_default(step=step):
				tf.summary.scalar('batch_loss', batch_loss)
				tf.summary.scalar('batch_top1_acc', batch_top1_acc)
				tf.summary.scalar('batch_top5_acc', batch_top5_acc)

		# validation code goes here
		if args.use_validation:
			for val_step, val_batch in enumerate(val_dataset):
				val_result = test(
						network = network,
						batch = val_batch,
						criterion = criterion,
						top1_acc = val_top1_acc,
						top5_acc = val_top5_acc,
						loss_tracker = val_loss_tracker,
					)
					
				# metrics
				val_batch_loss = val_loss_tracker.result()
				val_batch_top1_acc = val_top1_acc.result()
				val_batch_top5_acc = val_top5_acc.result()
					
				# val summary
				with val_writer.as_default(step=val_step):
					tf.summary.scalar('val_batch_loss', val_batch_loss)
					tf.summary.scalar('val_batch_top1_acc', val_batch_top1_acc)
					tf.summary.scalar('val_batch_top5_acc', val_batch_top5_acc)

				# updating the keras prohbar
				val_values = [('val_loss', val_batch_loss), ('val_top1_acc', val_batch_top1_acc), ('val_top5_acc', val_batch_top5_acc)]
				progbar.add(0, values=val_values)

				if val_step == 100:
					break

		# for training
		epoch_loss = loss_tracker.result()
		epoch_top1_acc = top1_acc.result()
		epoch_top5_acc = top5_acc.result()

		# epoch-level summary writer
		with train_writer.as_default(step=epoch):
			tf.summary.scalar('epoch_loss', epoch_loss)
			tf.summary.scalar('epoch_top1_acc', epoch_top1_acc)
			tf.summary.scalar('epoch_top5_acc', epoch_top5_acc)

		# for val
		if args.use_validation:
			val_epoch_loss = loss_tracker.result()
			val_epoch_top1_acc = top1_acc.result()
			val_epoch_top5_acc = top5_acc.result()

			# epoch-level summary writer
			with val_writer.as_default(step=epoch):
				tf.summary.scalar('val_epoch_loss', val_epoch_loss)
				tf.summary.scalar('val_epoch_top1_acc', val_epoch_top1_acc)
				tf.summary.scalar('val_epoch_top5_acc', val_epoch_top5_acc)

		# reset the netrics
		loss_tracker.reset_state()
		top5_acc.reset_state()
		top1_acc.reset_state()

		val_loss_tracker.reset_state()
		val_top1_acc.reset_state()
		val_top5_acc.reset_state()

		# for checkpoint update
		ckpt.step.assign_add(1)
		if int(ckpt.step) % args.chkpt_step == 0:
			save_path = manager.save()
			print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
			print("loss {:1.2f}".format(epoch_loss.numpy()))

		# shuffle the dataset
		train_dataset.on_epoch_end()
		if args.use_validation:
			val_dataset.on_epoch_end()


def train(network, batch, optimizer, criterion, top1_acc, top5_acc, loss_tracker):
	inputs, labels = batch 
	
	with tf.GradientTape() as tape:
		logits = network(inputs, training=True)

		# compute custom loss
		loss = criterion(labels, logits)

	# Compute gradients
	trainable_vars = network.trainable_weights

	#print(type(trainable_vars), len(trainable_vars))
	gradients = tape.gradient(loss, trainable_vars)

	#capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]

	# Update weights
	optimizer.apply_gradients(zip(gradients, trainable_vars))
	
	# Compute our own metrics
	loss_tracker.update_state(loss)
	top1_acc.update_state(labels, logits)
	top5_acc.update_state(labels, logits)
	
	return loss


def test(network, batch, criterion, top1_acc, top5_acc, loss_tracker):
	inputs, labels = batch  

	logits = network(inputs)

	loss = criterion(labels, logits)

	loss_tracker.update_state(loss)
	top1_acc.update_state(labels, logits)
	top5_acc.update_state(labels, logits)

	return loss 


if __name__ == '__main__':
	args = parse_args()

	main(args)