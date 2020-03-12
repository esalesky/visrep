import os
import argparse
import sys
import time
import numpy as np
import math

from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from text_utils import utf8_to_uxxxx, uxxxx_to_utf8


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--font', type=str,
                        default='', help='font file')

    args = parser.parse_args(argv)
    for arg in vars(args):
        print('%s %s' % (arg, getattr(args, arg)))

    return args


def main(args):
    try:
        font = TTFont(args.font)
        cmap = font.getBestCmap()
        for char_idx, char in enumerate(sorted(cmap)):
            print(char_idx, chr(char), utf8_to_uxxxx(chr(char)))
    except Exception as e:
        print("Failed to read", args)
        print(e)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
