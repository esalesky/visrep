#!/usr/bin/env python3

"""
Scrambles interior letters in a word.
"""

import sys
import random

def main(args):
    for line in sys.stdin:
        words = line.rstrip().split()
        for i, word in enumerate(words):
            scramble = True if random.random() < args.probability else False
            # needs to be len 4+ to enable a permutation with first and last chars held constant
            if len(word) > 3 and scramble:
                letters = list(word[1:-1])
                random.shuffle(letters)
                words[i] = word[0] + "".join(letters) + word[-1]
        print(" ".join(words))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability", default=0.1, help="probability of scramble")
    args = parser.parse_args()

    main(args)
