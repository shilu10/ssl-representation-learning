import tensorflow as tf 
from tensorflow import keras 
import numpy as np 
import os, sys, shutil


### for context-encoder
def get_center_block_mask(samples, mask_size, overlap):
    """
    Crop the center square region, mask it, and return relevant information.

    Parameters
    ----------
    samples: np.ndarray
        Batch of samples, e.g., images.
    mask_size: int
        Size of the squared block mask in pixels.
    overlap: int
        Number of pixels of overlap.

    Returns
    -------
    cropped_samples: np.ndarray
        Center-cropped samples.
    masked_samples: np.ndarray
        Samples with the center masked.
    center_dimensions: Tuple
        Tuple containing x and y dimensions of the center region.
    """
    img_size = samples.shape[1]
    center_index = (img_size - mask_size) // 2

    # Crop the center region
    cropped_samples = samples[:, center_index:center_index + mask_size, center_index:center_index + mask_size, :]

    # Image is not masked out in the overlap region
    m1, m2 = center_index + overlap, center_index + mask_size - overlap

    mask_color = np.array([2 * 117.0 / 255.0 - 1.0, 2 * 104.0 / 255.0 - 1.0, 2 * 123.0 / 255.0 - 1.0])

    # Mask the center region
    masked_samples = np.copy(samples)
    masked_samples[:, m1:m2, m1:m2, :] = mask_color

    center_dimensions = (center_index, center_index)

    return cropped_samples, masked_samples, center_dimensions


def get_random_block_mask(samples, mask_size, overlap):
    """
    Mask out a randomly positioned block in the samples with overlap.

    Parameters
    ----------
    samples: np.ndarray
        Batch of samples, e.g., images.
    mask_size: int
        Size of the squared block mask in pixels.
    overlap: int
        Number of pixels of overlap.

    Returns
    -------
    cropped_samples: np.ndarray
        Center-cropped samples.
    masked_samples: np.ndarray
        Original samples with the specified block masked with black pixels and overlap.
    block_indices: Tuple
        Tuple containing x and y indices of the upper left corner of the masked block.
    """
    img_size = samples.shape[1]

    # Randomly choose the position for the upper left corner of the masked block
    block_x = np.random.randint(0, img_size - mask_size + 1)
    block_y = np.random.randint(0, img_size - mask_size + 1)

    # Crop the center region
    cropped_samples = samples[:, block_x:block_x + mask_size, block_y:block_y + mask_size, :]

    # Create a mask with zeros at the specified block position
    masked_samples = np.copy(samples)

    # Add overlap region with zeros
    m1, m2 = block_x - overlap, block_x + mask_size + overlap
    n1, n2 = block_y - overlap, block_y + mask_size + overlap

    masked_samples[:, m1:m2, n1:n2, :] = 0.0

    block_indices = (block_x, block_y)

    return cropped_samples, masked_samples, block_indices


def generate_random_pattern(mask_area, resolution, max_pattern_size):
    """
    Generates global random pattern based on which random region masks can be sampled.
    TODO: Add reference
    """
    pattern = tf.random.uniform((int(resolution * max_pattern_size), int(resolution * max_pattern_size)), 0, 255)
    resized_pattern = tf.image.resize(tf.expand_dims(tf.expand_dims(pattern, axis=0), axis=-1), (max_pattern_size, max_pattern_size), method='bicubic')
    resized_pattern = tf.squeeze(resized_pattern) / 255.0
    return tf.math.less(resized_pattern, mask_area)


def get_random_region_mask(
        samples: np.ndarray,
        img_size: int,
        mask_area: float,
        global_random_pattern: np.ndarray,
):
    """
    Generate randomly masked images, which should be reconstructed / inpainted by context encoder generator.

    Parameters
    ----------
    samples: np.ndarray
        Batch of samples, e.g. images, which are passed through the network and for which specified intermediate
        results are extracted
    img_size: int
        Size of input images (squared images)
    mask_area: float
        Area of the image, which should be approximately masked out. The mask area is specified in percent of the
        total image area.
    global_random_pattern: np.ndarray
        Binary tensor which contains global random pattern based on which random region masks are computed.
        Tensor elements are either 1 or 0. 0 is indicating that the element is masked out.
    Returns
    -------
    masked_samples: np.ndarray
        Array containing samples to which the random mask is applied.
    mask: np.ndarray
        Binary array representing the mask applied to samples provided as input. It contains 1 for pixels which
        have not been masked out and 0 for pixels which have been masked out.
    """
    while True:
        x, y = np.random.randint(0, global_random_pattern.shape[0] - img_size, 2)
        mask = global_random_pattern[x: x + img_size, y: y + img_size]
        pattern_mask_area = np.mean(mask.astype(float))
        # If mask area is within +/- 25% of desired mask area, break and continue
        if mask_area / 1.25 < pattern_mask_area < mask_area * 1.25:
            break
    
    masked_samples = np.copy(samples)
    masked_samples[:, mask, 0] = 2 * 117.0 / 255.0 - 1.0
    masked_samples[:, mask, 1] = 2 * 104.0 / 255.0 - 1.0
    masked_samples[:, mask, 2] = 2 * 123.0 / 255.0 - 1.0

    return masked_samples, mask


def get_l2_weights(args, prediction_size, masked_region=None):
    """
    Get tensor of weights for the l2-reconstruction loss. Loss weights are chosen depending on whether they belong
    to the overlap region or not. For random masking all unmasked regions are taken as overlap region, i.e.
    straightforward reconstruction of the region in the original input image.

    Parameters
    ----------
    args: argparse.Namespace
        Batch of samples, e.g. images, which are passed through the network and for which specified intermediate
        results are extracted
    prediction_size: tf.TensorShape
        Size of the predictions / generated image part based on which the generator l2-loss is calculated
    masked_region: tf.Tensor
        Binary tensor encoding the masked region of the input image (in case of random masking).
    """
    if args.overlap != 0:
        loss_weights = tf.constant(args.w_rec * args.overlap_weight_multiplier, shape=prediction_size)

        if not args.random_masking:
            loss_weights[:, args.overlap:-args.overlap, args.overlap:-args.overlap] = args.w_rec
        else:
            # Assuming masked_region is a boolean mask
            loss_weights = tf.where(masked_region, args.w_rec, loss_weights)
    else:
        if not args.random_masking:
            loss_weights = tf.ones(prediction_size)
        else:
            loss_weights = tf.zeros(prediction_size)
            # Assuming masked_region is a boolean mask
            loss_weights = tf.where(masked_region, args.w_rec, loss_weights)
    return loss_weights


def weighted_mse_loss(outputs, targets, weights):
    return tf.pow(tf.reduce-mean(weights * (outputs - targets)), 2)