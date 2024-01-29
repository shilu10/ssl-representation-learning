import tensorflow as tf 
from imutils import paths

from src.utils.common import load_module_from_source
import src.networks.contrastive_task as networks
from src.memory_bank import MemoryBank



def parse_image(indices, image_path):
    raw = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(raw)

    image = tf.cast(image, dtype=tf.float32)

    image /= 255.

    return indices, image 


def get_dataset(config):
    datapath = config.memory_bank.get("datapath")

    image_files = list(paths.list_images(datapath))
    dataset_size = len(image_files)
    indices = [_ for _ in range(dataset_size)]

    steps_per_epoch = int(dataset_size / 10)


    dataset = tf.data.Dataset.from_tensor_slices((indices, image_files))
    dataset = dataset.shuffle(dataset_size)

    # parallel extraction
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.batch(batch_size=10)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset, dataset_size, steps_per_epoch


def main(config):
    projection_dims = config.model.get("projection_dims")

    # get dataset
    dataset, dataset_size, steps_per_epoch = get_dataset(config)
    
    # initialize memory bank
    memory_bank = MemoryBank(
                    shape=(dataset_size, projection_dims),
                    weight=config.memory_bank.get("weight"))

    # initialize networks
    encoder_type = config.networks.get("encoder_type")
    img_size = config.model.get("img_size")
    encoder = getattr(networks, encoder_type)(
                include_top=False, 
                input_shape=(img_size, img_size, 3), 
                pooling=None)
    
    f = getattr(networks, "GenericTask")(encoding_size=projection_dims)

    memory_bank.initialize(encoder=encoder, 
                           f=f, 
                           train_loader=dataset, 
                           steps_per_epoch=steps_per_epoch, 
                           sep_init=True)

    print('Completed the memory bank initialization')

    return memory_bank

