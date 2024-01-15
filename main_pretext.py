import tensorflow as tf 
from tensorflow.keras.utils import Progbar
from tensorflow import keras 
import os, sys, shutil, random
from imutils import paths 
from tqdm import tqdm
import numpy as np 
import argparse 
from datetime import datetime 
import itertools 

import src.networks.pretext_task as networks
import src.data.pretext_task as dataloaders
from src.utils.common import get_optimizer, get_criterion, load_module_from_source
from src.utils.pretext_task import get_center_block_mask, get_random_region_mask, generate_random_pattern, get_random_block_mask
from arguments import main_pretext_parse_args


AUTO = tf.data.experimental.AUTOTUNE


def main(args):
	global_random_pattern = None

	#---------------------------
	# Load config
	#---------------------------

	module_name = "config"
	file_path = args.config_path

	config = load_module_from_source(module_name, file_path)
	
	# assertion errors
	if args.pretext_task_type == 'jigsaw':
		permutation_arr_path = config.model.get('permutation_path')
		assert os.path.exists(permutation_arr_path), "no file or folder exists, use hamming_set.py to initialize the permutation_arr"
		
		permutation_arr_path_n_classes = permutation_arr_path.split('.')[0].split('_')[-1]
		print(permutation_arr_path_n_classes)
		assert int(permutation_arr_path_n_classes) == config.model.get("num_classes"), "permutation_arr_path mismatch with num_classes"
	
	assert os.path.exists(args.unlabeled_datapath), f"no file or folder exists at {args.unlabeled_datapath}"
	
	# setting specific gpu in multi-gpu workstation for cuda job.
	#if args.gpu is not None:
	#	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	#	os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

	#else: 
	#	print('CPU mode')


	# ---------------------------
	# Train and Val image splits
	# --------------------------
	image_files_path = list(paths.list_images(args.unlabeled_datapath))
	
	if args.use_validation:
		num_val_images = int(len(image_files_path) * args.val_split_size)
		validation_image_files_path = image_files_path[: num_val_images]
		train_image_files_path = image_files_path[num_val_images+1: ]

		train_labels = [np.random.randint(0, 10) for _ in range(len(train_image_files_path))]
		val_labels = [np.random.randint(0, 10) for _ in range(len(validation_image_files_path))]

	else:
		train_image_files_path = image_files_path
		train_labels = [np.random.randint(0, 10) for _ in range(len(train_image_files_path))]


	#------------------------
	# Dataloaders
	#------------------------
	train_dataloader = getattr(dataloaders,
							   config.dataloader.get('type'))
	train_dataloader = train_dataloader(args=config, 
										image_files_path=train_image_files_path, 
										labels=train_labels, 
										batch_size=args.batch_size, 
										shuffle=args.shuffle).create_dataset()

	# val dataloader
	if args.use_validation:
		val_dataloader = getattr(dataloaders, config.dataloader.get('type'))
		val_dataloader = val_dataloader(args=config, 
										image_files_path=train_image_files_path, 
										labels=train_labels, 
										batch_size=args.batch_size, 
										shuffle=args.shuffle).create_dataset()


	iter_per_epoch = int(len(image_files_path) / args.batch_size)

	tf.print(f"Iteration Per Epochs: {iter_per_epoch}")

	#------------------------
	# Load Network
	#------------------------
	if args.pretext_task_type == 'context_encoder':
		img_size = config.model.get('img_size')
		mask_area = config.model.get('mask_area')
		mask_size = int(np.sqrt(mask_area) * img_size)

		if config.model.get('random_masking'):
			out_size = img_size
		else:
			out_size = mask_size

		context_generator = getattr(networks, 
									config.networks.get('generator_name'))
		context_generator = context_generator(bottleneck_dim=config.model.get('bottleneck_dim'), 
											img_size=img_size, 
											out_size=out_size, 
											channels=3)

		context_discriminator = getattr(networks, 
										config.networks.get('discriminator_name'))
		context_discriminator = context_discriminator(input_size=out_size,
													 in_channels=3)

	else:
		network = getattr(networks, config.networks.get('type'))
		network = network(config=config, 
						 n_classes=config.model.get('num_classes'),
						 name=args.pretext_task_type)
	
	#-----------------------
	# Load Optimizer
	# ----------------------
	if args.pretext_task_type == 'context_encoder':
		g_optimizer = get_optimizer(optim_type=config.optimizer.get('generator_type'), 
									learning_rate=config.optimizer.get('generator_lr'))
		d_optimizer = get_optimizer(optim_type=config.optimizer.get('discriminator_type'), 
									learning_rate=config.optimizer.get('discriminator_lr'))

	else:
		optimizer = get_optimizer(optim_type=config.optimizer.get('type'), 
								learning_rate=config.optimizer.get('lr'))

	#if config.optimizer.get('use_lr_scheduler'):

	#-------------------------------
	# Load Loss Function(criterion)
	# -------------------------------
	if args.pretext_task_type == 'context_encoder':
		adversial_criterion = get_criterion(config.criterion.get('adversial_type'))
		reconstruction_criterion = get_criterion(config.criterion.get('reconstruction_type'))

	else:
		criterion = get_criterion(config.criterion.get('type'),
								 from_logits=True)


	#--------------------------------
	# Load Checkpoint
	#--------------------------------
	if args.pretext_task_type == "context_encoder":
		# checkpoint 
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
								  context_generator=context_generator, 
								  context_discriminator=context_discriminator, 
								  g_optim=g_optimizer, 
								  d_optim=d_optimizer)
	
	else:
		ckpt = tf.train.Checkpoint(step=tf.Variable(1), 
								  optimizer=optimizer, 
								  net=network)

	manager = tf.train.CheckpointManager(checkpoint=ckpt,
										 directory=args.checkpoint,
										 max_to_keep=3)

	ckpt.restore(manager.latest_checkpoint)

	if manager.latest_checkpoint:
		print("Restored from {}".format(manager.latest_checkpoint))

	else:
		print("Initializing network from scratch.")


	#------------------------------
	# Load Trackers
	#------------------------------
	if args.pretext_task_type == 'context_encoder':
		gen_total_loss_tracker = tf.keras.metrics.Mean(name='total_gen_loss_tracker')
		gen_adv_loss_tracker = tf.keras.metrics.Mean(name='gen_adv_loss_tracker')
		gen_recon_loss_tracker = tf.keras.metrics.Mean(name='gen_recon_loss_tracker')
		dis_loss_tracker = tf.keras.metrics.Mean(name='dis_loss_tracker')

		# accuracy tracker
		dis_acc_tracker = tf.keras.metrics.Accuracy(name='discriminator_accuracy')

	else:

		loss_tracker = tf.keras.metrics.Mean()
		top1_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, 
										name='top_k_categorical_accuracy', dtype=None)
		top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, 
										name='top_k_categorical_accuracy', dtype=None)

	# val trackers
	if args.use_validation:
		if args.pretext_task_type != 'context_encoder':
			val_loss_tracker = tf.keras.metrics.Mean()
			val_top1_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1, 
											name='top_k_categorical_accuracy', dtype=None)
			val_top5_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, 
											name='top_k_categorical_accuracy', dtype=None)

		else:
			val_gen_recon_loss_tracker = tf.keras.metrics.Mean(name='gen_recon_loss_tracker')


	# ----------------------------------
	# Load Summary Writer
	# ----------------------------------
	common_log_parent_path = f'{args.tensorboard}/batch_level/' + \
							datetime.now().strftime("%Y%m%d-%H%M%S") + \
							args.pretext_task_type
	train_log_dir = common_log_parent_path + '/train'
	train_writer = tf.summary.create_file_writer(train_log_dir)

	if args.use_validation:
		val_log_dir = common_log_parent_path + '/val'
		val_writer = tf.summary.create_file_writer(val_log_dir)


	#--------------------------------------
	# Create Progress Bar (keras.progbar)
	#--------------------------------------
	stateful_metrics = []
	if args.pretext_task_type != 'context_encoder':
		metrics_names = ['loss', 'top1_acc', 'top5_acc']
	else:
		metrics_names = ['gen_total_loss', 'gen_adv_loss', \
							 'gen_recon_loss', 'disc_loss', 'disc_acc']

	stateful_metrics += metrics_names

	if args.use_validation:
		if args.pretext_task_type != 'context_encoder':
			val_metrics_names = ['val_loss', 'val_top1_acc', 'val_top5_acc']
		else:
			val_metrics_names = ['val_batch_gen_recon_loss']

		stateful_metrics += val_metrics_names

	progbar = tf.keras.utils.Progbar(iter_per_epoch, stateful_metrics=stateful_metrics)


	#----------------------
	# Start train and val
	#----------------------
	if config.model.get('random_masking'):
		global_random_pattern = generate_random_pattern(mask_area=config.model.get('mask_area'), 
														resolution=config.model.get('resolution'), 
														max_pattern_size=config.model.get('max_pattern_size'))
	


	for epoch in range(args.num_epochs):
		print("\nepoch {}/{}".format(epoch+1, args.num_epochs))

		#values = [('loss', 0.0), ('top1_acc', 0.0), ('top5_acc', 0.0)]
		#val_values = [('val_loss', 0.0), ('val_top1_acc', 0.0), ('val_top5_acc', 0.0)]
		#progbar.add(0, values=values)
		#progbar.add(0, values=val_values)

		# train step
		for step, batch in enumerate(train_dataloader):

			# initial updation of val progbar

			# for context encoder
			if args.pretext_task_type == 'context_encoder':
				samples, _ = batch

				if not config.model.get('random_masking'):
					true_masks, masked_samples, _ = get_center_block_mask(samples=samples.numpy(), 
	                													mask_size=mask_size,
	                													overlap=args.model.get('overlap'))

					masked_samples = tf.convert_to_tensor(masked_samples, dtype=tf.float32)
					true_masks = tf.convert_to_tensor(true_masks, dtype=tf.float32)
					masked_region = None

				else:
					masked_samples, masked_region = get_random_region_mask(samples=samples.numpy(), 
	                													img_size=args.model.get('img_size'),
	                													mask_area=args.model.get('mask_area'),
	                													global_random_pattern=global_random_pattern)

					masked_samples = tf.convert_to_tensor(masked_samples, dtype=tf.float32)
					true_masks = samples

				inputs = (masked_samples, true_masks, masked_region)

				results = train_context_encoder(args=args, 
	                       inputs=inputs, 
	                       generator=context_gen, 
	                       discriminator=context_dis, 
	                       g_optim=cg_optim, 
	                       d_optim=cd_optim, 
	                       adv_loss_func=adversarial_loss, 
	                       recon_loss_func=reconstruction_loss,
	                       gen_total_loss_tracker=gen_total_loss_tracker, 
	                       gen_adv_loss_tracker=gen_adv_loss_tracker, 
	                       gen_recon_loss_tracker=gen_recon_loss_tracker, 
	                       dis_loss_tracker=dis_loss_tracker,
	                       dis_acc_tracker=dis_acc_tracker
	                )

				# metrics
				batch_gen_total_loss = gen_total_loss_tracker.result()
				batch_gen_adv_loss = gen_adv_loss_tracker.result()
				batch_gen_recon_loss = gen_recon_loss_tracker.result()
				batch_dis_total_loss = dis_loss_tracker.result()
				batch_dis_acc = dis_acc_tracker.result()

				values = [('gen_total_loss', batch_gen_total_loss), \
	                      ('gen_adv_loss', batch_gen_adv_loss), \
	                      ('gen_recon_loss', batch_gen_recon_loss), \
	                      ('disc_loss', batch_dis_total_loss), \
	                      ('batch_dis_acc', batch_dis_acc)]

				progbar.update(step + 1, values=values)	

				# summary writer
				with train_writer.as_default(step=step):
					tf.summary.scalar('batch_gen_total_loss', batch_gen_total_loss)
					tf.summary.scalar('batch_gen_adv_loss', batch_gen_adv_loss)
					tf.summary.scalar('batch_gen_recon_loss', batch_gen_recon_loss)
					tf.summary.scalar('batch_dis_total_loss', batch_dis_total_loss)
					tf.summary.scalar('batch_dis_acc', batch_dis_acc)
					train_writer.flush()

			else:

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
			for val_step, val_batch in enumerate(val_dataloader):

				# for context_encoder
				if args.pretext_task_type == 'context_encoder':
					samples, _ = batch 
					if not config.model.get('random_masking'):
						true_masks, masked_samples, _ = get_center_block_mask(samples=samples.numpy(),
																			 mask_size=mask_size,
																			 overlap=args.model.get('overlap'))

						masked_samples = tf.convert_to_tensor(masked_samples, dtype=tf.float32)
						true_masks = tf.convert_to_tensor(true_masks, dtype=tf.float32)
						masked_region = None

					else:
						masked_samples, masked_region = get_random_region_mask(samples=samples.numpy(),
																			 img_size=config.model.get('img_size'),
																			 mask_area=config.model.get('mask_area'),
																			 global_random_pattern=global_random_pattern)

						masked_samples = tf.convert_to_tensor(masked_samples, dtype=tf.float32)
						true_masks = samples


					inputs = (masked_samples, true_masks, masked_region)
					results = test_context_encoder(args=args, 
			                                inputs=inputs, 
			                                generator=context_gen, 
			                                recon_loss_func=reconstruction_loss,
			                                val_gen_recon_loss_tracker=val_gen_recon_loss_tracker
			                            )

					# batch metrics
					val_batch_gen_recon_loss = val_gen_recon_loss_tracker.result()


					# updating the keras prohbar
					val_values = [('val_batch_gen_recon_loss', val_batch_gen_recon_loss)]
					progbar.add(0, values=val_values)


					# summary writer
					with val_writer.as_default(step=val_step):
						tf.summary.scalar('val_batch_gen_recon_loss', val_batch_gen_recon_loss)
						val_writer.flush()

				else:
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

					# updating the keras prohbar
					val_values = [('val_loss', val_batch_loss), ('val_top1_acc', val_batch_top1_acc), ('val_top5_acc', val_batch_top5_acc)]
					progbar.add(0, values=val_values)

					# val summary
					with val_writer.as_default(step=val_step):
						tf.summary.scalar('val_batch_loss', val_batch_loss)
						tf.summary.scalar('val_batch_top1_acc', val_batch_top1_acc)
						tf.summary.scalar('val_batch_top5_acc', val_batch_top5_acc)

					
				if val_step == 100:
					break
						

		if args.pretext_task_type == 'context_encoder':
			# epoch metrics
			epoch_gen_total_loss = gen_total_loss_tracker.result()
			epoch_gen_adv_loss = gen_adv_loss_tracker.result()
			epoch_gen_recon_loss = gen_recon_loss_tracker.result()
			epoch_dis_total_loss = dis_loss_tracker.result()
			epoch_dis_acc = dis_acc_tracker.result()

		else:
			# for training
			epoch_loss = loss_tracker.result()
			epoch_top1_acc = top1_acc.result()
			epoch_top5_acc = top5_acc.result()
			

		# epoch-level summary writer
		with train_writer.as_default(step=epoch):
			if args.pretext_task_type != 'context_encoder':
				tf.summary.scalar('epoch_loss', epoch_loss)
				tf.summary.scalar('epoch_top1_acc', epoch_top1_acc)
				tf.summary.scalar('epoch_top5_acc', epoch_top5_acc)

			else:
				tf.summary.scalar('epoch_gen_total_loss', epoch_gen_total_loss)
				tf.summary.scalar('epoch_gen_adv_loss', epoch_gen_adv_loss)
				tf.summary.scalar('epoch_gen_recon_loss', epoch_gen_recon_loss)
				tf.summary.scalar('epoch_dis_total_loss', epoch_dis_total_loss)
				tf.summary.scalar('epoch_dis_acc', epoch_dis_acc)

			train_writer.flush()
	            

		# for val
		if args.use_validation:
			if args.pretext_task_type != "context_encoder":
				val_epoch_loss = loss_tracker.result()
				val_epoch_top1_acc = top1_acc.result()
				val_epoch_top5_acc = top5_acc.result()

			else:
				val_epoch_gen_recon_loss = val_gen_recon_loss_tracker.result()

			# epoch-level summary writer
			with val_writer.as_default(step=epoch):
				if args.pretext_task_type != "context_encoder":
					tf.summary.scalar('val_epoch_loss', val_epoch_loss)
					tf.summary.scalar('val_epoch_top1_acc', val_epoch_top1_acc)
					tf.summary.scalar('val_epoch_top5_acc', val_epoch_top5_acc)

				else:
					tf.summary.scalar('val_epoch_gen_recon_loss', val_epoch_gen_recon_loss)

				val_writer.flush()

		if args.pretext_task_type != "context_encoder":
			# reset the netrics
			loss_tracker.reset_state()
			top5_acc.reset_state()
			top1_acc.reset_state()

		else:
			gen_total_loss_tracker.reset_state()
			gen_adv_loss_tracker.reset_state()
			gen_recon_loss_tracker.reset_state()
			dis_loss_tracker.reset_state()

		if args.use_validation:
			if args.pretext_task_type != "context_encoder":
				val_loss_tracker.reset_state()
				val_top1_acc.reset_state()
				val_top5_acc.reset_state()

			else:
				val_gen_recon_loss_tracker.reset_state()


		# for checkpoint update
		ckpt.step.assign_add(1)
		if int(ckpt.step) % args.chkpt_step == 0:
			save_path = manager.save()
			print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
			print("loss {:1.2f}".format(epoch_loss.numpy()))


	tf.summary.FileWriter.close()

		
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


def train_context_encoder(args,
		               inputs, 
		               generator, 
		               discriminator, 
		               g_optim, 
		               d_optim, 
		               adv_loss_func, 
		               recon_loss_func, 
		               gen_total_loss_tracker, 
		               gen_adv_loss_tracker, 
		               gen_recon_loss_tracker, 
		               dis_loss_tracker, dis_acc_tracker):
    
    masked_samples, true_masks, masked_region = inputs
    
    # training generators
    true_labels = tf.ones(args.batch_size)
    fake_labels = tf.ones(args.batch_size)
    
    # discrimnator gradient cal
    with tf.GradientTape() as tape:
        # train discrinator
        dis_true_output = discriminator(true_masks)
        
        # generate fake images
        gen_fake_output = generator(masked_samples, training=True)
        
        dis_fake_output = discriminator(gen_fake_output, training=True)
        
        true_output_loss = adv_loss_func(true_labels, dis_true_output)
        fake_output_loss = adv_loss_func(fake_labels, dis_fake_output)
        
        # total discriminator loss
        dis_loss = (true_output_loss + fake_output_loss) * 0.5
    
    trainable_vars = discriminator.trainable_weights
    gradients = tape.gradient(dis_loss, trainable_vars)
    
    d_optim.apply_gradients(zip(gradients, trainable_vars))
    
    # generator gradient cal
    with tf.GradientTape() as tape:
        gen_fake_output = generator(masked_samples, training=True)
        
        dis_fake_output = discriminator(gen_fake_output, training=True)
        
        # Compute adversarial loss for generator
        gen_adv_loss = adv_loss_func(true_labels, dis_fake_output)
        
        # compute generator recontruction loss
        # l2_weights = get_l2_weights(args, gen_fake_output.shape, masked_region)
        gen_recon_loss = recon_loss_func(true_masks, gen_fake_output)
        
        gen_total_loss = (1 - args.w_rec) * gen_adv_loss + args.w_rec * gen_recon_loss
        
    trainable_vars = generator.trainable_weights
    gradients = tape.gradient(gen_total_loss, trainable_vars)
    
    g_optim.apply_gradients(zip(gradients, trainable_vars))
    
    # generator loss tracker
    gen_total_loss_tracker.update_state(gen_total_loss)
    gen_adv_loss_tracker.update_state(gen_adv_loss)
    gen_recon_loss_tracker.update_state(gen_recon_loss)
    
    # discrimnator loss tracker
    dis_loss_tracker.update_state(dis_loss)
    
    dis_acc_tracker.update_state(true_labels, dis_true_output)
    dis_acc_tracker.update_state(fake_labels, dis_fake_output)
    
    return {
        'gen_loss': gen_total_loss,
        'dis_loss': dis_loss
    }


def test_context_encoder(args, inputs, generator, recon_loss_func, val_gen_recon_loss_tracker):
    
    # unpack inputs
    masked_samples, true_masks, masked_region = inputs
    
    gen_fake_output = generator(masked_samples)
    
    # compute generator recontruction loss
    #l2_weights = get_l2_weights(args, gen_fake_output.shape, masked_region)
    gen_recon_loss = recon_loss_func(true_masks, gen_fake_output)
    
    # update generator tracker
    val_gen_recon_loss_tracker.update_state(gen_recon_loss)
        
    return gen_recon_loss
    

if __name__ == '__main__':
	args = main_pretext_parse_args()

	main(args)


