import numpy as np
import itertools
from scipy.spatial.distance import cdist
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_crops', type=int, default=9)
    parser.add_argument('--num_permutations', type=int, default=1000)
    parser.add_argument('--selection', type=str, default="max", choices=["max", "mean"])
    parser.add_argument('--output_filepath', type=str, default='./')

    return parser.parse_args()


def hamming_set(args):
    """
    generate and save the hamming set
    :param num_crops: number of tiles from each image
    :param num_permutations: Number of permutations to select (i.e. number of classes for the pretext task)
    :param selection: Sample selected per iteration based on hamming distance: [max] highest; [mean] average
    :param output_file_name: name of the output HDF5 file
    """
    num_crops = args.num_crops
    num_permutations = args.num_permutations
    selection = args.selection
    output_filepath = args.output_filepath

    save_path = output_filepath + "permutation_" + str(selection) + "_" + str(num_permutations)

    P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
    n = P_hat.shape[0]

    for i in range(num_permutations):
        if i == 0:
            j = np.random.randint(n)
            P = np.array(P_hat[j]).reshape([1, -1])
        else:
            P = np.concatenate([P, P_hat[j].reshape([1, -1])], axis=0)

        P_hat = np.delete(P_hat, j, axis=0)
        D = cdist(P, P_hat, metric='hamming').mean(axis=0).flatten()

        if selection == 'max':
            j = D.argmax()
        elif selection == 'mean':
            m = int(D.shape[0] / 2)
            S = D.argsort()
            j = S[np.random.randint(m - 10, m + 10)]


    with open(save_path + ".npy", 'wb') as f:
        np.save(f, arr=P)
  
    print('file created --> ' + save_path + '.npy')


if __name__ == "__main__":
    args = parse_args()

    hamming_set(args)