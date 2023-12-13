import tensorflow as tf
import numpy as np
from glob import glob                                                           
import cv2 


cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)


from glob import glob                                                           
import cv2 

def convert_x_to_y_image_format(dir_path, x_format='png', y_format='jpg'):
    files = glob('/kaggle/input/cifar10/cifar10/train/**/*.png', recursive=True)
    for file in files:
        # Load .png image
        image = cv2.imread(file)

        # Save .jpg image
        splitted = file.split('/')
        dir_name = splitted[-2]
        file_name = splitted[-1]

        #out = cv2.imwrite(j[:-3] + 'jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        out = cv2.imwrite(f"cifar10/{dir_name}/{file_name[: -3]}" + 'jpeg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


## negative mask used in simclr loss
def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)


def _cosine_simililarity_dim1(x, y):
    v = cosine_sim_1d(x, y)
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v


def _dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v


def _dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v


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


def set_seed(SEED=42):
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)


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