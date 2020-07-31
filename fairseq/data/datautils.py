import torch
from torch.utils.data.sampler import Sampler


def uxxxx_to_utf8(in_str):
    idx = 0
    result = ""
    if in_str.strip() == "":
        return ""

    for uxxxx in in_str.split():
        if uxxxx == "":
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
            #    sys.exit(1)

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
        return " ".join(char_array)


class OcrGroupedSampler(Sampler):
    """Dataset is divided into sub-groups, G_1, G_2, ..., G_k
       Samples Randomly in G_1, then moves on to sample randomly into G_2, etc all the way to G_k

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, rand=True, max_items=-1, fixed_rand=False):
        self.size_group_keys = data_source.src.size_group_keys
        self.size_groups = data_source.src.size_groups
        self.num_samples = len(data_source.src)
        self.rand = rand
        self.fixed_rand = fixed_rand
        self.max_items = max_items
        self.rand_perm = dict()

    def __iter__(self):
        n_items = 0
        for g in self.size_group_keys:
            if len(self.size_groups[g]) == 0:
                continue

            if self.fixed_rand:
                if g not in self.rand_perm:
                    self.rand_perm[g] = torch.randperm(len(self.size_groups[g])).long()
                g_idx_iter = iter(self.rand_perm[g])
            else:
                if self.rand:
                    g_idx_iter = iter(torch.randperm(len(self.size_groups[g])).long())
                else:
                    g_idx_iter = iter(range(len(self.size_groups[g])))

            while True:
                try:
                    g_idx = next(g_idx_iter)
                except StopIteration:
                    break

                n_items += 1
                if self.max_items > 0 and n_items > self.max_items:
                    raise StopIteration
                yield self.size_groups[g][g_idx]

        raise StopIteration

    def __len__(self):
        return self.num_samples

