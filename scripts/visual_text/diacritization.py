#!/usr/bin/env python3

"""
Diacritization for Arabic
"""

import sys
import random
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator


# For each disambiguated word d in disambig, d.analyses is a list of analyses
# sorted from most likely to least likely. Therefore, d.analyses[0] would
# be the most likely analysis for a given word. 

def main(args):
    mle = MLEDisambiguator.pretrained()

    for line in sys.stdin:
        # The disambiguator expects pre-tokenized text
        original = line.rstrip()
        sentence = simple_word_tokenize(original)
        disambig = mle.disambiguate(sentence)

        words       = [d.word for d in disambig]
        diacritized = [d.analyses[0].analysis['diac'] if len(d.analyses)>0 else d.word for d in disambig]

        exchange = True if random.random() < args.probability else False
        for i, (w,d) in enumerate(zip(words,diacritized)):
            if exchange:
                words[i] = d
        
        print(" ".join(words))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--probability", default=0.1, type=float, help="probability of diacritization")
    args = parser.parse_args()

    main(args)
