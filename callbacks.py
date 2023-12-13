import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil


class CustomCSVLogger(CSVLogger):
    """Save averaged logs during training.
    """
    def on_epoch_begin(self, epoch, logs=None):
        self.batch_logs = {}

    def on_batch_end(self, batch, logs=None):
        for k, v in logs.items():
            if k not in self.batch_logs:
                self.batch_logs[k] = [v]
            else:
                self.batch_logs[k].append(v)

    def on_epoch_end(self, epoch, logs=None):
        final_logs = {k: np.mean(v) for k, v in self.batch_logs.items()}
        super(CustomCSVLogger, self).on_epoch_end(epoch, final_logs)


def get_callbacks(args):
    if not args.resume:
        if args.checkpoint or args.history or args.tensorboard:
            if os.path.isdir(f'{args.result_path}/{args.task}/{args.stamp}'):
                flag = input(f'\n{args.task}/{args.stamp} is already saved. '
                              'Do you want new stamp? (y/n) ')
                if flag == 'y':
                    args.stamp = create_stamp()
                    initial_epoch = 0
                    logger.info(f'New stamp {args.stamp} will be created.')

                elif flag == 'n':
                    return -1, initial_epoch

                else:
                    logger.info(f'You must select \'y\' or \'n\'.')
                    return -2, initial_epoch

            os.makedirs(f'{args.result_path}/{args.task}/{args.stamp}')
            yaml.dump(
                vars(args), 
                open(f'{args.result_path}/{args.task}/{args.stamp}/model_desc.yml', 'w'), 
                default_flow_style=False)
        else:
            logger.info(f'{args.stamp} is not created due to '
                        f'checkpoint - {args.checkpoint} | '
                        f'history - {args.history} | '
                        f'tensorboard - {args.tensorboard}')

    callbacks = []
        
    if args.checkpoint:
        if args.task=="pretraining":
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(
                    f'{args.result_path}/{args.model_type}/{args.stamp}/checkpoint',
                    '{epoch:04d}_{loss:.4f}_{acc1:.4f}_{acc5:.4f}'),
                monitor='acc1',
                mode='max',
                verbose=1,
                save_weights_only=True))
        else:
            callbacks.append(ModelCheckpoint(
                filepath=os.path.join(
                    f'{args.result_path}/{args.model_type}/{args.stamp}/checkpoint',
                    '{epoch:04d}_{val_loss:.4f}_{val_acc1:.4f}_{val_acc5:.4f}'),
                monitor='val_acc1',
                mode='max',
                verbose=1,
                save_weights_only=True))

    if args.history:
        os.makedirs(f'{args.result_path}/{args.model_type}/{args.stamp}/history', exist_ok=True)
        callbacks.append(CustomCSVLogger(
            filename=f'{args.result_path}/{args.model_type}/{args.stamp}/history/epoch.csv',
            separator=',', append=True))

    if args.tensorboard:
        callbacks.append(TensorBoard(
            log_dir=f'{args.result_path}/{args.model_type}/{args.stamp}/logs',
            histogram_freq=args.tb_histogram,
            write_graph=True, 
            write_images=True,
            update_freq=args.tb_interval,
            profile_batch=100,))

    return callbacks, initial_epoch