#!/usr/bin/env python3

"""
Maps characters to l33t-speak
"""

import sys
import random

cyr2rom = {
    'а' : 'a',
    'В' : 'B',
    'ь' : 'b',
    'Н' : 'H',
    'н' : 'H',
    'е' : 'e',
    'о' : 'o',
    'О' : 'O',
    'с' : 'c',
    'С' : 'C',
    'Я' : 'R',
    'м' : 'M',
    'М' : 'M',
    }

def noisify(char):
    return str(cyr2rom[char])

def main(args):
    for line in sys.stdin:
        words = line.rstrip().split()
        for i, word in enumerate(words):
            letters = list(word)
            for j, char in enumerate(letters):
                swap = True if random.random() < args.probability else False
                if char in cyr2rom and swap:
                    letters[j] = noisify(char)
            words[i] = ''.join(letters)
        print(" ".join(words))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability", default=0.1, type=float, help="probability of l33t mapping")
    args = parser.parse_args()

    main(args)
