import math
import sys
import numpy as np


def uxxxx_to_utf8(in_str):
    idx = 0
    result = ''
    if in_str.strip() == "":
        return ""

    for uxxxx in in_str.split():
        if uxxxx == '':
            continue

        if uxxxx == "<unk>" or uxxxx == "<s>" or uxxxx == "</s>":
            cur_utf8_char = uxxxx
        else:
            # First get the 'xxxx' part out of the current 'uxxxx' char

            cur_char = uxxxx[1:]

            # Now decode the hex code point into utf-8
            try:
                cur_utf8_char = chr(int(cur_char, 16))
            except:
                print("Exception converting cur_char = [%s]" % cur_char)
                sys.exit(1)

        # And add it to the result buffer
        result = result + cur_utf8_char

    return result


def utf8_to_uxxxx(in_str, output_array=False):

    char_array = []
    for char in in_str:
        raw_hex = hex(ord(char))[2:].zfill(4).lower()
        char_array.append("u%s" % raw_hex)

    if output_array:
        return char_array
    else:
        return ' '.join(char_array)


# Rudimintary dynamic programming method of computing edit distance bewteen two sequences
# Let dist[i,j] = edit distance between A[0..i] and B[0..j]
#  Then dist[i,j] is the smallest of:
#    (1) dist[i,j-1] + 1   i.e. between A[0..i] and B[0..j-1] plus 1 to cover the insertion of B[j]
#    (2) dist[i-1,j] + 1   i.e. between A[0..i-1] and B[0..j] plus 1 to cover the insertion of A[i]
#    (3) dist[i-1,j-1] + (1 if A[i]===B[j], else 0)  i.e. between A[0..i-1] and B[0..j-1] and  the edit distance between A[i],B[j]


def edit_distance(A, B):
    # If both strings are empty, edit distance is 0
    if len(A) == 0 and len(B) == 0:
        return 0
    # If one or the other is empty, then edit distance is length of the other one
    if len(A) == 0 or len(B) == 0:
        return len(A) + len(B)

    # Otherwise have to actually compute it the hard way :)
    dist_matrix = np.zeros((len(A)+1, len(B)+1))
    for i in range(len(A)+1):
        dist_matrix[i, 0] = i
    for j in range(len(B)+1):
        dist_matrix[0, j] = j

    for i in range(1, len(A)+1):
        for j in range(1, len(B)+1):
            if A[i-1] == B[j-1]:
                dist_matrix[i, j] = dist_matrix[i-1, j-1]
            else:
                dist_matrix[i, j] = 1 + min(dist_matrix[i, j - 1], dist_matrix[i - 1, j],
                                            dist_matrix[i - 1, j - 1])

    return dist_matrix[-1, -1]


def form_tokenized_words(chars, with_spaces=False):
    punctuations = {"u002e", "u002c", "u003b", "u0027", "u0022", "u002f", "u0021", "u0028", "u0029", "u005b", "u005d", "u003c", "u003e",
                    "u002d", "u005f", "u007b", "u007d", "u0024", "u0025", "u0023", "u0026", "u060c", "u201d", "u060d", "u060f", "u061f",
                    "u066d", "ufd3e", "ufd3f", "u061e", "u066a", "u066b", "u066c", "u002a", "u002b", "u003a", "u003d", "u005e", "u0060", "u007c", "u007e"}
    digits = {
        "u0660", "u0661", "u0662", "u0663", "u0664", "u0665", "u0666", "u0667", "u0668", "u0669", "u0030", "u0031",
        "u0032", "u0033", "u0034", "u0035", "u0036", "u0037", "u0038", "u0039"
    }

    words = []
    start_idx = 0
    for i in range(len(chars)):
        if chars[i] == 'u0020':  # Space denotes new word
            if start_idx != i:
                words.append('_'.join(chars[start_idx:i]))

                if with_spaces:
                    words.append("u0020")
            start_idx = i + 1
            continue
        if chars[i] in punctuations or chars[i] in digits:
            if start_idx != i:
                words.append('_'.join(chars[start_idx:i]))
            words.append(chars[i])
            start_idx = i + 1
            continue
        if i == len(chars) - 1:
            # At end of line, so just toss remaining line into word array
            if start_idx == i:
                words.append(chars[start_idx])
            else:
                words.append('_'.join(chars[start_idx:]))

    return words


def compute_cer_wer(hyp_transcription, ref_transcription):
    # Assume input in uxxxx format, i.e. looks like this:  "u0062 u0020 u0064 ..."

    # To compute CER we need to split on uxxxx chars, which are seperated by space
    hyp_chars = hyp_transcription.split(' ')
    ref_chars = ref_transcription.split(' ')

    char_dist = edit_distance(hyp_chars, ref_chars)

    #logger.info('hyp %s' % hyp_chars)
    #logger.info('ref %s' % ref_chars)
    #logger.info('dist %s' % char_dist)

    # To compute WER we need to split by words, and tokenize on punctuation
    # We rely on Alphabet objects to provide the chars to tokenize on
    hyp_words = form_tokenized_words(hyp_chars)
    ref_words = form_tokenized_words(ref_chars)
    # Remove whitespace at beg/end
    while len(hyp_words) > 0 and hyp_words[0] == 'u0020':
        hyp_words = hyp_words[1:]
    while len(hyp_words) > 0 and hyp_words[-1] == 'u0020':
        hyp_words = hyp_words[:-1]
    while len(ref_words) > 0 and ref_words[0] == 'u0020':
        ref_words = ref_words[1:]
    while len(ref_words) > 0 and ref_words[-1] == 'u0020':
        ref_words = ref_words[:-1]

    word_dist = edit_distance(hyp_words, ref_words)

    if len(ref_chars) == 0 or len(ref_words) == 0:
        return None, None
    else:
        return float(char_dist) / len(ref_chars), float(word_dist) / len(ref_words)
