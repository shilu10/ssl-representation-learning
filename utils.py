import tensorflow as tf
import numpy as np
from glob import glob                                                           
import cv2, os, shutil, sys 
from imutils import paths 
import random

cosine_sim_1d = tf.keras.metrics.CosineSimilarity(axis=1)
cosine_sim_2d = tf.keras.metrics.CosineSimilarity(axis=2)


from glob import glob                                                           
import cv2 

AUTO = tf.data.experimental.AUTOTUNE


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
    v = tf.abs(tf.keras.losses.cosine_similarity(x, y, axis=1))
    return v


def _cosine_simililarity_dim2(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = tf.abs(tf.keras.losses.cosine_similarity(tf.expand_dims(x, 1), tf.expand_dims(y, 0), axis=2))
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


def read_image(image_path, label):
    raw = tf.io.read_file(image_path)

    return tf.data.Dataset.from_tensors((raw, label))


def gen_random():
    r=tf.random.uniform([], minval=0, maxval=99, dtype=tf.dtypes.int32)

    return r 


class JigSaw(object):

    def __init__(self, args, permutation_arr):
        self.args = args 
        self.permutation_arr = permutation_arr
        self.permutation_indices = [_ for _ in range(len(permutation_arr))]

        
    @tf.function
    def transform(self, img, label):

      #  label = tf.numpy_function(lambda label: int(label.numpy()), [label], tf.int32)

       # print(tf.compat.v1.contrib.util.make_ndarray(label))
       # print(label.numpy(), "label")

        mean, std = mean_std
        img = tf.cast(img, tf.float32)
        img /= 255.
        img -= mean
        img /= std

        img = tf.clip_by_value(img, 0, 1)

        copy_img = tf.image.resize(img, (225, 225), method='bilinear')

        imgclips = []
        for i in range(3):
            for j in range(3):
                clip = copy_img[i * 75: (i + 1) * 75, j * 75: (j + 1) * 75, :]
                randomx = tf.experimental.numpy.random.randint(0, 10)
                randomy = tf.experimental.numpy.random.randint(0, 10)
                clip = clip[randomx: randomx+64, randomy:randomy+64, :]

                imgclips.append(clip)

        #random_index = random.randint(0, len(self.permutation_arr)-1)
        np.random.shuffle(self.permutation_indices)

        #print(shuffle_permu_ind)

        #print(self.permutation_indices)

       # r_index = self.permutation_indices[0]

        r_index = tf.random.uniform([], minval=0, maxval=99, dtype=tf.dtypes.int32)
        print(r_index, 'r_index')
        #dummy = tf.zeros(100)
        #print(dummy[tf.cast(label, tf.int32)])
        selected_permutation = self.permutation_arr[r_index.numpy()]

        imgclips = tf.convert_to_tensor(imgclips)
        
        shuffled_tiles = tf.gather(imgclips, selected_permutation, axis=0)

        return shuffled_tiles, r_index, imgclips

    def transform1(self, image, label):

        mean, std = mean_std
        image = tf.cast(image, tf.float32)
        image /= 255.
        image -= mean
        image /= std

        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        image = tf.clip_by_value(image, 0, 1)

        #x = random.randint(0, width - 225)
        #y = random.randint(0, height - 225)
        #cropped_image = image[y:y + 225, x:x + 225]

        cropped_image = tf.image.random_crop(image, size=(225, 225, 3))

        # grid size or grid dim
        grid_size = self.args.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        height, width, channels = cropped_image.shape

        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                coordinates.append((i * height // grid_size[0], j * width // grid_size[1]))

        grids = []
        for coordinate in coordinates:
            grid = cropped_image[coordinate[0]:coordinate[0] + height // grid_size[0], coordinate[1]:coordinate[1] + width // grid_size[1], :]
            grids.append(grid)

        grids = tf.convert_to_tensor(grids)

        # extract 65*65 tile from the grid
        n_grid = grid_size[0] * grid_size[1]
        tiles = tf.image.random_crop(grids, (n_grid, 64, 64, 3))

        random_index = np.random.randint(0, len(self.permutation_arr)-1)
        selected_permutation = self.permutation_arr[random_index]

        shuffled_tiles = tf.gather(tiles, selected_permutation)

        return shuffled_tiles, random_index, tiles


def preprocess_image(image_path, label, tranform_obj=None):
    value = tf.io.read_file(image_path)
    shape = tf.image.extract_jpeg_shape(value)

    print(tranform_obj)

    img = tf.image.decode_jpeg(value)

#    perm = random.randint(0, 9)

    #r = gen_random()

    # transformations
    titles, perm_label, _ = transform(tf.zeros((225, 255, 3)), label)
    print(perm_label, "perm label")

    return tf.zeros((225, 255, 3), dtype=tf.float32), perm_label



class DataLoader:
    def __init__(self, args, permutation_path, num_classes):
        self.args = args 
        self.permutations = self.__retrive_permutations(num_classes, permutation_path)
        self.num_classes = num_classes

    def preprocess_image(self, image_path, label):
        value = tf.io.read_file(image_path)
        shape = tf.image.extract_jpeg_shape(value)

        image = tf.image.decode_jpeg(value)

        result = self.transform(image, label)

        if len(result) == 1:
            print("not 64")

        tiles, perm_label = result

        return tiles, perm_label 

    def transform1(self, img, label):

      #  label = tf.numpy_function(lambda label: int(label.numpy()), [label], tf.int32)

       # print(tf.compat.v1.contrib.util.make_ndarray(label))
       # print(label.numpy(), "label")

        img = tf.cast(img, tf.float32)
  

        img = tf.clip_by_value(img, 0, 1)

        copy_img = tf.image.resize(img, (225, 225), method='bilinear')

        imgclips = []
        for i in range(3):
            for j in range(3):
                clip = copy_img[i * 75: (i + 1) * 75, j * 75: (j + 1) * 75, :]
                randomx = tf.experimental.numpy.random.randint(0, 10)
                randomy = tf.experimental.numpy.random.randint(0, 10)
                clip = clip[randomx: randomx+64, randomy:randomy+64, :]

                if clip.shape[0] != 64 or clip.shape[1] != 64:
                    print("not same shape")
                    return "not a same shape"

                imgclips.append(clip)

        #random_index = random.randint(0, len(self.permutation_arr)-1)
       # np.random.shuffle(self.permutation_indices)

        #print(shuffle_permu_ind)

        #print(self.permutation_indices)

       # r_index = self.permutation_indices[0]

        r_index = tf.random.uniform([], minval=0, maxval=99, dtype=tf.dtypes.int32)
        selected_permutation = self.permutations[r_index.numpy()]

        imgclips = tf.convert_to_tensor(imgclips)
        
        shuffled_tiles = tf.gather(imgclips, selected_permutation, axis=0)

        return shuffled_tiles, r_index

    def transform(self, image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.
        #image -= mean
        #image /= std

        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        image = tf.clip_by_value(image, 0, 1)

        cropped_image = tf.image.random_crop(image, size=(225, 225, 3))

        # grid size or grid dim
        grid_size = self.args.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        height, width, channels = cropped_image.shape

        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                coordinates.append((i * height // grid_size[0], j * width // grid_size[1]))

        grids = []
        for coordinate in coordinates:
            grid = cropped_image[coordinate[0]:coordinate[0] + height // grid_size[0], coordinate[1]:coordinate[1] + width // grid_size[1], :]
           
            randomx = tf.experimental.numpy.random.randint(0, 10)
            randomy = tf.experimental.numpy.random.randint(0, 10)
            #clip = grid[randomx: randomx+64, randomy:randomy+64, :]
            grids.append(grid)

            
        grids = tf.convert_to_tensor(grids)

        # extract 65*65 tile from the grid
        n_grid = grid_size[0] * grid_size[1]

        #print(grids.shape)
        tiles = tf.image.random_crop(grids, (-1, 64, 64, 3))

        r_index = tf.random.uniform([], minval=0, maxval=self.num_classes, dtype=tf.dtypes.int32)
        selected_permutation = self.permutations[r_index.numpy()]

        shuffled_tiles = tf.gather(tiles, selected_permutation, axis=0)

        return shuffled_tiles, r_index


    def __retrive_permutations(self, num_classes, permutation_path):
        all_perm = np.load(permutation_path)

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def get_dataset(self):
        
        # trainloader and val dataloader 
        image_files = list(paths.list_images(self.args.unlabeled_datapath))
        #dummy_labels = [_ for _ in range(len(image_files))]

        # dataloader
        dataset = tf.data.Dataset.from_tensor_slices((image_files))
        indices = tf.data.Dataset.from_tensor_slices([random.randint(0, 99) for _ in range(len(image_files))])
        dataset = tf.data.Dataset.zip((dataset, indices))
        
        #dataset = dataset.repeat()
        
        if self.args.shuffle:
            dataset = dataset.shuffle(len(image_files))

        # for parallel extraction
        #dataset = dataset.interleave(read_image, num_parallel_calls=AUTO)

        # for parallel preprocessing
        dataset = dataset.map(lambda x, y:  tf.py_function(self.preprocess_image, [x, y], [tf.float32, tf.int32]))

        dataset = dataset.batch(self.args.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(AUTO)

        return dataset


'''
class PretextTaskGenerator(keras.utils.Sequence):

    def __init__(self, data_path, txt_list, classes=1000):
        # Initialization
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.x = x_in
        self.y = y_in
        self.datalen = len(y_in)
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # get batch indexes from shuffled indexes
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch
    
    def __len__(self):
        # Denotes the number of batches per epoch
        return self.datalen // self.batch_size

    def on_epoch_end(self):
        # Updates indexes after each epoch
        self.indexes = np.arange(self.datalen)
        if self.shuffle:
            np.random.shuffle(self.indexes)
'''


class PretextTaskDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, args, image_paths, batch_size, shuffle, num_classes, permutation_path):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.permutations = self.__retrive_permutations(num_classes, permutation_path)
        self.args = args

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size

        images = []
        labels = []
        for image_path in self.image_paths[start: end]:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image)


            random_index = random.randint(0, self.num_classes-1)
            sel_permutation = self.permutations[random_index]

            transformed_image, _, _ = self.transform(image, sel_permutation)

            images.append(transformed_image)
            labels.append(random_index)
        
        return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)

    def on_epoch_end(self):
        #self.indexes = np.arange(self.image_paths)

        if self.shuffle:
            np.random.shuffle(self.image_paths)
    
    def __retrive_permutations(self, num_classes, permutation_path):
        all_perm = np.load(permutation_path)

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def transform(self, image, permutation):
        image = tf.cast(image, tf.float32)
        image /= 255.
        #image -= mean
        #image /= std

        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        image = tf.clip_by_value(image, 0, 1)

        #x = random.randint(0, width - 225)
        #y = random.randint(0, height - 225)
        #cropped_image = image[y:y + 225, x:x + 225]

        cropped_image = tf.image.random_crop(image, size=(225, 225, 3))

        # grid size or grid dim
        grid_size = self.args.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        height, width, channels = cropped_image.shape

        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                coordinates.append((i * height // grid_size[0], j * width // grid_size[1]))

        grids = []
        for coordinate in coordinates:
            grid = cropped_image[coordinate[0]:coordinate[0] + height // grid_size[0], coordinate[1]:coordinate[1] + width // grid_size[1], :]
            grids.append(grid)

        grids = tf.convert_to_tensor(grids)

        # extract 65*65 tile from the grid
        n_grid = grid_size[0] * grid_size[1]
        tiles = tf.image.random_crop(grids, (n_grid, 64, 64, 3))

       # random_index = np.random.randint(0, len(self.permutation_arr)-1)
       # selected_permutation = self.permutation_arr[random_index]

        shuffled_tiles = tf.gather(tiles, permutation)

        return shuffled_tiles, permutation, tiles


class PretextTaskDataGenerator1(tf.keras.utils.Sequence):

    def __init__(self, args, image_paths, batch_size, shuffle, num_classes, permutation_path):
        self.image_paths = image_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_classes = num_classes
        self.permutations = self.__retrive_permutations(num_classes, permutation_path)
        self.args = args

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, index):
        start = index * self.batch_size
        end = start + self.batch_size

        batch_images = []
        batch_labels = []
        for image_path in self.image_paths[start: end]:
            image = tf.io.read_file(image_path)
            image = tf.image.decode_image(image)


            random_index = random.randint(0, self.num_classes-1)
            sel_permutation = self.permutations[random_index]

            transformed_image, _, _ = self.transform(image, sel_permutation)

            batch_images.append(transformed_image)
            batch_labels.append(random_index)
        
        return np.stack(batch_images, axis=0), np.array(batch_labels)

    def on_epoch_end(self):
        #self.indexes = np.arange(self.image_paths)

        if self.shuffle:
            np.random.shuffle(self.image_paths)
    
    def __retrive_permutations(self, num_classes, permutation_path):
        all_perm = np.load(permutation_path)

        # from range [1,9] to [0,8]
        if all_perm.min() == 1:
            all_perm = all_perm - 1

        return all_perm

    def transform(self, image, permutation):
        image = tf.cast(image, tf.float32)
        image /= 255.
        #image -= mean
        #image /= std

        image = tf.image.resize(image, 
                                size=(256, 256), 
                                method=tf.image.ResizeMethod.BILINEAR)

        image = tf.clip_by_value(image, 0, 1)

        #x = random.randint(0, width - 225)
        #y = random.randint(0, height - 225)
        #cropped_image = image[y:y + 225, x:x + 225]

        cropped_image = tf.image.random_crop(image, size=(225, 225, 3))

        # grid size or grid dim
        grid_size = self.args.grid_size
        if isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)

        height, width, channels = cropped_image.shape

        coordinates = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                coordinates.append((i * height // grid_size[0], j * width // grid_size[1]))

        grids = []
        for coordinate in coordinates:
            grid = cropped_image[coordinate[0]:coordinate[0] + height // grid_size[0], coordinate[1]:coordinate[1] + width // grid_size[1], :]
            grids.append(grid)

        grids = tf.convert_to_tensor(grids)

        # extract 65*65 tile from the grid
        n_grid = grid_size[0] * grid_size[1]
        tiles = tf.image.random_crop(grids, (n_grid, 64, 64, 3))

       # random_index = np.random.randint(0, len(self.permutation_arr)-1)
       # selected_permutation = self.permutation_arr[random_index]

        shuffled_tiles = tf.gather(tiles, permutation)

        return shuffled_tiles, permutation, tiles