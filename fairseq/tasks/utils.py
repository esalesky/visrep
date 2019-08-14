
import sys

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
