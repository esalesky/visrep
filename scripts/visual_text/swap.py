#!/usr/bin/env python3

"""
Swap pairs of adjacent chars (only one per token)
"""

import sys
import random

def main(args):
    for line in sys.stdin:
        words = line.rstrip().split()
        for i, word in enumerate(words):
            swap = True if random.random() < args.probability else False
            if len(word) > 1 and swap:
                #left idx for swap: subtract 1 from len to have a char to the right
                idx = random.choice(range(len(word)-1))
                letters = list(word)
                letters[idx]=word[idx+1]
                letters[idx+1]=word[idx]
                words[i] = ''.join(letters)
        print(" ".join(words))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability", default=0.1, help="probability of swap")
    args = parser.parse_args()

    main(args)
