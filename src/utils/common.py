import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil
import importlib


def get_optimizer(optim_type, learning_rate, *args, **kwargs):
    if optim_type == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


    return optimizer


def get_criterion(criterion_type, reduction_type=None, from_logits=False):
    if criterion_type == 'mse':
        loss_func = tf.keras.losses.MeanSquaredError(
                            reduction='auto',
                            name='mean_squared_error')

    elif criterion_type == 'bce':
        loss_func = tf.keras.losses.BinaryCrossentropy(from_logits)

    elif criterion_type == 'sparse_categorical_crossentropy':
        loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits)

    elif criterion_type == 'categorical_crossentropy':
        loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits)


    return loss_func


def load_module_from_source(module_name, file_path):
    # Create a spec for the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    
    # Create a module based on the spec
    module = importlib.util.module_from_spec(spec)
    
    # Load the source code into the module
    spec.loader.exec_module(module)
    
    return module


def search_same(args):
    search_ignore = ['checkpoint', 'history', 'tensorboard',
                     'tb_interval', 'snapshot', 'summary',
                     'src_path', 'data_path', 'result_path',
                     'resume', 'stamp', 'gpus', 'ignore_search']
    if len(args.ignore_search) > 0:
        search_ignore += args.ignore_search.split(',')

    initial_epoch = 0
    stamps = os.listdir(f'{args.result_path}/{args.task}')
    for stamp in stamps:
        try:
            desc = yaml.full_load(
                open(f'{args.result_path}/{args.task}/{stamp}/model_desc.yml', 'r'))
        except:
            continue

        flag = True
        for k, v in vars(args).items():
            if k in search_ignore:
                continue

            if v != desc[k]:
                # if stamp == '210120_Wed_05_19_52':
                #     print(stamp, k, desc[k], v)
                flag = False
                break

        if flag:
            args.stamp = stamp
            df = pd.read_csv(
                os.path.join(
                    args.result_path,
                    f'{args.task}/{args.stamp}/history/epoch.csv'))

            if len(df) > 0:
                if int(df['epoch'].values[-1]+1) == args.epochs:
                    print(f'{stamp} Training already finished!!!')
                    return args, -1

                elif np.isnan(df['loss'].values[-1]) or np.isinf(df['loss'].values[-1]):
                    print('{} | Epoch {:04d}: Invalid loss, terminating training'.format(stamp, int(df['epoch'].values[-1]+1)))
                    return args, -1

                else:
                    ckpt_list = sorted(
                        [d.split('.index')[0] for d in os.listdir(
                            f'{args.result_path}/{args.task}/{args.stamp}/checkpoint') if 'index' in d])

                    if len(ckpt_list) > 0:
                        args.snapshot = f'{args.result_path}/{args.task}/{args.stamp}/checkpoint/{ckpt_list[-1]}'
                        initial_epoch = int(ckpt_list[-1].split('_')[0])
                    else:
                        print('{} Training already finished!!!'.format(stamp))
                        return args, -1
            break
    return args, initial_epoch


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

