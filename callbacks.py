import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil



class CustomMemoryBankCallback(CSVLogger):
    """Save averaged logs during training.
    """
    def on_epoch_begin(self, epoch, logs=None):
        pass


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


class ExponentialDecayScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None):
        super(ExponentialDecayScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.staircase = staircase
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            global_step_recomp = tf.cast(step, dtype)
            p = global_step_recomp / decay_steps
            if self.staircase:
                p = tf.floor(p)
            return tf.multiply(
                initial_learning_rate, tf.pow(decay_rate, p), name=name
            )
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "decay_rate": self.decay_rate,
            "staircase": self.staircase,
            "name": self.name,
        })
        
        return config


class CosineDecayWithWarmupScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_steps, warmup_target, warmup_steps=0, alpha=0.0, name=None):
        super(CosineDecayWithWarmupScheduler, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.warmup_target = warmup_target
        self.warmup_steps = warmup_steps
        self.alpha = alpha  # Controls the flatness of the cosine annealing
        self.name = name
        
    def _decay_function(self, step, decay_steps, decay_from_lr, dtype):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / decay_steps
            tf_pi = tf.constant(math.pi, dtype=dtype)
            cosine_decayed = 0.5 * (1.0 + tf.cos(tf_pi * completed_fraction))
            decayed = (1 - self.alpha) * cosine_decayed + self.alpha
            return tf.multiply(decay_from_lr, decayed)
    
    def _warmup_function(
        self, step, warmup_steps, warmup_target, initial_learning_rate
    ):
        with tf.name_scope(self.name or "CosineDecay"):
            completed_fraction = step / warmup_steps
            total_step_delta = warmup_target - initial_learning_rate
            return total_step_delta * completed_fraction + initial_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "CosineDecay"):
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate"
            )
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            global_step_recomp = tf.cast(step, dtype)
            
            if self.warmup_target is None:
                global_step_recomp = tf.minimum(global_step_recomp, decay_steps)
                return self._decay_function(
                    global_step_recomp,
                    decay_steps,
                    initial_learning_rate,
                    dtype,
                )
            
            warmup_target = tf.cast(self.warmup_target, dtype)
            warmup_steps = tf.cast(self.warmup_steps, dtype)
            
            global_step_recomp = tf.minimum(
                global_step_recomp, decay_steps + warmup_steps
            )

            return tf.cond(
                global_step_recomp < warmup_steps,
                lambda: self._warmup_function(
                    global_step_recomp,
                    warmup_steps,
                    warmup_target,
                    initial_learning_rate,
                ),
                lambda: self._decay_function(
                    global_step_recomp - warmup_steps,
                    decay_steps,
                    warmup_target,
                    dtype,
                ),
            )
        
    def get_config(self):
        config = super().get_config()
        
        config.update({
            'initial_learning_rate': self.initial_learning_rate,
            'decay_steps': self.decay_steps,
            'warmup_target': self.warmup_target,
            'alpha': self.alpha,
            'warmup_steps': self.warmup_steps,
            'name': self.name
        })
        
        return config
            

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