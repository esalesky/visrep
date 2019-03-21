#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
PIPE = '|'
maxw = 30

if __name__ == '__main__':
    swap = sys.argv[1].lower().strip() == 'first-last-swap'
    for line in sys.stdin:
        o = []
        for w in line.strip().split():
            c = [i for i in w.strip() if i != PIPE]
            if len(c) > maxw:
                c = c[:maxw]  # no swapping
            elif len(c) > 2:
                if swap:
                    c[1], c[-1] = c[-1], c[1]  # swap
                else:
                    pass
                pass
            elif len(c) == 1:
                pass
            else:
                pass
            f = PIPE.join(c)
            o.append(f)
        print(' '.join(o))
