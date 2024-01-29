import tensorflow as tf 
import imutils, os
import pickle 

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # suppress info-level logs 

from arguments import main_parse_args
from helper import get_logger

import src.data.contrastive_task as dataloaders 
import src.losses as losses 
import src.algorithms as algorithms
from src.utils.common import load_module_from_source
from src.callbacks import CustomModelCheckpoint, CustomTensorBoard


tf.get_logger().setLevel("WARN")  # suppress info-level logs



def main(args):

    module_name = "config"
    file_path = args.config_path

    config = load_module_from_source(module_name, file_path)

    #args, initial_epoch = search_same(args)
    initial_epoch = 0

    if initial_epoch == -1:
        # training was already finished!
        return
    
    logger = get_logger(args.contrastive_task_type)

    for k, v in vars(args).items():
        logger.info("{} : {}".format(k, v))

    image_files_path = list(imutils.paths.list_images(args.unlabeled_datapath))
    
    if args.use_validation:
        num_val_images = int(len(image_files_path) * args.val_split_size)
        validation_image_files_path = image_files_path[: num_val_images]
        train_image_files_path = image_files_path[num_val_images+1: ]

    else:
        train_image_files_path = image_files_path

    #######################
    # DATALOADER
    #######################

    dataloader = getattr(dataloaders, config.dataloader.get("type"))
    dataloader = dataloader(config=config, 
                            image_files_path=train_image_files_path, 
                            batch_size=args.batch_size, 
                            shuffle=args.shuffle).create_dataset()


    n_image_files = len(train_image_files_path)
    steps_per_epoch = n_image_files / args.batch_size

    logger.info("Loaded pretraining dataloader")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Number of steps_per_epoch: {steps_per_epoch}")

    ########################
    # Load Model
    ########################
    model = getattr(algorithms, config.model.get('algorithm_type'))
    model = model(config)

    #####################
    # OPTIMIZER
    #####################

    optimizer = tf.keras.optimizers.Adam()

    # Build the optimizer with all trainable variables
   # img_dim = config.model.get("img_size")
    #model.one_step(input_shape=(1, img_dim, img_dim, 3))
    #all_trainable_params = model.get_all_trainable_params
    #print(all_trainable_params)
    #optimizer.build(all_trainable_params)

    ###################
    # Criterion
    ###################
    loss = getattr(losses, config.criterion.get("type"))(config, args.batch_size)

    ###################
    # CALLBACKS
    ###################

    checkpoint_dir = args.checkpoint

    # Ensure the checkpoint directory exists
    tf.io.gfile.makedirs(checkpoint_dir)

    checkpoint_filepath = checkpoint_dir + "/" + args.contrastive_task_type + "_{epoch:02d}.h5"
    model_checkpoint = CustomModelCheckpoint(
                            checkpoint_dir=checkpoint_dir,
                            filepath=checkpoint_filepath, 
                            save_freq=3, 
                            max_to_keep=2)

    ###################
    # TensorBoard
    ###################

    tensorbaord_dir_path = args.tensorboard + '/' + args.contrastive_task_type
    #tb_callback = tf.keras.callbacks.TensorBoard(tensorbaord_dir_path,
                                               # update_freq=1)
    tb_callback = CustomTensorBoard(tensorbaord_dir_path, 1)

    model.compile(
        optimizer=optimizer, 
        loss = loss,
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(1, 'acc1', dtype=tf.float32),
                tf.keras.metrics.SparseTopKCategoricalAccuracy(5, 'acc5', dtype=tf.float32)],
    )

    logger.info(f"STARTING TRAINING OF MODEL WITH {args.num_epochs} EPOCHS.")

    history = model.fit(dataloader, 
                        epochs=args.num_epochs, 
                        #validation_data=test_dataset, 
                        callbacks=[tb_callback, model_checkpoint],#)
                        steps_per_epoch=10)
    

if __name__ == '__main__':
    args = main_parse_args()
    main(args)