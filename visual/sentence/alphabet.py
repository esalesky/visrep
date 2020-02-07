import torch


class Alphabet(object):

    def __init__(self, char_array, left_to_right=False):

        self.left_to_right = left_to_right
        self.char_to_idx = dict(zip(char_array, range(len(char_array))))
        self.idx_to_char = dict(zip(range(len(char_array)), char_array))
        self.char_array = char_array
        self.max_id = range(len(char_array))[-1]
        self.blank_index = self.char_to_idx['<ctc-blank>']

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.blank_index

    def len(self):
        return len(self.idx_to_char)

    def __len__(self):
        return len(self.idx_to_char)

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.idx_to_char[t] for t in tensor)

        sent = ' '.join(self.idx_to_char[i] for i in tensor)
        return sent
