
import os
import argparse
import sys
import time
import numpy as np


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str,
                        default='', help='Input path')

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('%s %s' % (arg, getattr(args, arg)))

    return args


def main(args):
    start_time = time.process_time()

    print('__Python VERSION: %s' % (sys.version))

    np_embedding = np.load(args.input, allow_pickle=True)
    decode_metadata = np_embedding['metadata'].item()

    meta_image_id = decode_metadata['image_id']
    meta_ref_utf8_text = decode_metadata['utf8_ref_text']
    meta_ref_uxxxx_text = decode_metadata['uxxxx_ref_text']
    meta_image = decode_metadata['image']
    meta_embedding = decode_metadata['embedding']

    print('image_id:', meta_image_id)
    print('ref_utf8_text:', meta_ref_utf8_text)
    print('ref_uxxxx_text:', meta_ref_uxxxx_text)
    print('image.shape:', meta_image.shape)
    print('embedding.shape:', meta_embedding.shape)

    print('...complete, time %.2f' % (time.process_time() - start_time))


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
