#!/usr/bin/env python3

"""
Maps characters to l33t-speak
"""

import sys
import random

l33t_map = {
    'a' : 4,
    'b' : 8,
    'e' : 3,
    'f' : 'ƒ',
    'i' : 1,
    'o' : 0,
    's' : 5,
    't' : 7,
    'v' : '\/',
    'z' : 2,
    }

def l33tify(char):
    return str(l33t_map[char.lower()])

def main(args):
    for line in sys.stdin:
        words = line.rstrip().split()
        for i, word in enumerate(words):
            letters = list(word)
            for j, char in enumerate(letters):
                swap = True if random.random() < args.probability else False
                if char.lower() in l33t_map and swap:
                    letters[j] = l33tify(char)
            words[i] = ''.join(letters)
        print(" ".join(words))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability", default=0.1, type=float, help="probability of l33t mapping")
    args = parser.parse_args()

    main(args)
