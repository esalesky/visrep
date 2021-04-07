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
            if len(word) > 2:
                letters = list(word[1:-1])
                random.shuffle(letters)
                words[i] = word[0] + "".join(letters) + word[-1]
        print(" ".join(words))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)
