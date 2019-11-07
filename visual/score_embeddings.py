""" Score an embedding."""
import argparse
import sys
import time
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input', type=str, help='Input seed text',
        default='')
    parser.add_argument(
        '--output', type=str, help='Output directory',
        default='')

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('{} {}'.format(arg, getattr(args, arg)))

    return args


def main(args):
    start_time = time.clock()

    print('__Python VERSION:', sys.version)

    # np.set_printoptions(precision=4)

    np_embedding = np.load(args.input)
    features = np_embedding['features']
    labels = np_embedding['labels']
    print('feature shape {}, labels shape {}'.format(
        features.shape, labels.shape))

    # print(labels[0:20])
    # print(features[0][0:20])

    # print('')
    # # np_norm(np.expand_dims(np.array(features[0]), axis=0))
    # np_embed_norm = np_norm(np.array(features[0:10]))
    # print('')
    # # sk_norm(np.expand_dims(np.array(features[0]), axis=0))
    # sk_embed_norm = sk_norm(np.array(features[0:10]))

    # print('')
    # np_cosine(np.array(np_embed_norm))
    # print('')
    # sk_cosine(np.array(np_embed_norm))
    # print('')

    print('start full norm')
    np_embed_norm = np_norm(features)

    np.set_printoptions(precision=6, linewidth=120)
    for img_idx, img in enumerate(labels):
        print('%s - %s, %s' %
              (labels[img_idx], features[img_idx][0:5], np_embed_norm[img_idx][0:5]))
        if img_idx > 20:
            break

    print('start full cosine')
    np_embed_cosine = np_cosine(np_embed_norm)
    print(np_embed_cosine[0:10])
    print(np_embed_cosine.shape)
    print(np_embed_cosine[0][0])
    print(np_embed_cosine[1000][1000])
    # print(np_embed_cosine[20000][20000])
    # print(np_embed_cosine[21123][21123])

    print(np_embed_cosine.shape)
    print('argsort')
    idx = (-np_embed_cosine).argsort()[:100]
    print(idx.shape)
    # print(idx[0:5])
    for ctr in range(30):
        print(labels[idx[ctr][0]], labels[idx[ctr][1]],
              labels[idx[ctr][2]], labels[idx[ctr][3]],
              labels[idx[ctr][4]], labels[idx[ctr][5]])

    print('...complete, time {}'.format((time.clock() - start_time)))


def np_norm(np_embed_norm):
    # print('np norm')
    # print(np_embed_norm[0][0:10])
    np_embed_norm_copy = np.copy(np_embed_norm)
    np_norm_val = np.linalg.norm(np_embed_norm_copy,
                                 axis=1, keepdims=True)
    # print(np_norm_val[0])
    # print('')
    np_embed_norm_copy /= np_norm_val

    # print(np_embed_norm[0][0:10])

    # confirm unit norm
    # sq_np_embed_norm = np_embed_norm ** 2
    # print(np.sum(sq_np_embed_norm, axis=1))

    return np_embed_norm_copy


def sk_norm(sk_embed_norm):
    # print('sk norm')
    # print(sk_embed_norm[0][0:10])

    # sklearn norm
    sk_embed_norm = normalize(sk_embed_norm, norm='l2')

    # print(sk_embed_norm[0][0:10])

    # confirm unit norm
    # sq_sk_embed_norm = sk_embed_norm ** 2
    # print(np.sum(sq_sk_embed_norm, axis=1))

    return sk_embed_norm


def np_cosine(A):
    # base similarity matrix (all dot products)
    # replace this with A.dot(A.T).toarray() for sparse representation
    similarity = np.dot(A, A.T)

    # squared magnitude of preference vectors (number of occurrences)
    square_mag = np.diag(similarity)

    # inverse squared magnitude
    inv_square_mag = 1 / square_mag

    # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
    inv_square_mag[np.isinf(inv_square_mag)] = 0

    # inverse of the magnitude
    inv_mag = np.sqrt(inv_square_mag)

    # cosine similarity (elementwise multiply by inverse magnitudes)
    cosine = similarity * inv_mag
    cosine = cosine.T * inv_mag
    # print(cosine)
    return cosine


def sk_cosine(A):
    cosine = cosine_similarity(A, A)
    # print(cosine)
    return cosine


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
