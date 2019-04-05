#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
PIPE = '|'

def find_ngrams(input_list, n):
    return [''.join(j) for j in zip(*[input_list[i:] for i in range(n)])]


if __name__ == '__main__':
    swap = sys.argv[1].lower().strip() == 'first-last-swap'
    n = int(sys.argv[2])  #ngram size for char
    maxw = int(sys.argv[3])  # max num chars in word
    for line in sys.stdin:
        o = []
        for w in line.strip().split():
            c = [i for i in w.strip() if i != PIPE]
            if n > 1:
                c = find_ngrams(c, n)
            if len(c) >= 3:
                if swap:
                    c[1], c[-1] = c[-1], c[1]  # swap
                    if len(c) > maxw:
                        c = c[:maxw]  # no swapping
                else:
                    pass
                pass
            else:
                pass
            f = PIPE.join(c)
            o.append(f)
        print(' '.join(o))
