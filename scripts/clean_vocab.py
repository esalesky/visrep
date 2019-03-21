#!/usr/bin/env python
__author__ = 'arenduchintala'
import argparse

if __name__ == '__main__':
    opt = argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='dict_file', required=True)
    options = opt.parse_args()

    for l in open(options.dict_file, 'r', encoding='utf-8').readlines():
        i, c = l.split()
        if len(i) > 1:
            i = '_'
        elif ord(i) == 65533:
            i = '_'
        else:
            pass
        print(i)
