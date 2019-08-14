
import torch
from fairseq.data import Dictionary
import json
import os
import sys

    
class JSONDictionary(Dictionary):
    """
    Dictionary for OCR tasks. This extends Dictionary by
    adding the blank symbol.
    """
    def __init__(self, blank='<blank>', pad='<pad>', eos='</s>', unk='<unk>', bos='<s>'):
        self.blank_word, self.unk_word, self.pad_word, self.eos_word = blank, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        self.blank_index = self.add_symbol(blank)
        self.pad_index = self.add_symbol(pad)
        self.eos_index = self.add_symbol(eos)
        self.unk_index = self.add_symbol(unk)        
        self.nspecial = len(self.symbols)

    def blank(self):
        """Helper to get index of blank symbol"""
        return self.blank_index


    @classmethod
    def load(cls, f, ignore_utf_errors=False):
        """Loads the dictionary from a text file with the format:"""
        #with open(os.path.join(f, 'desc.json'), 'r') as fh:
        with open(f, 'r') as fh:
            input_json = json.load(fh)
            
        unique_chars = {}

        for split in ['train', 'validation', 'test']:
            for entry in input_json[split]:
                for char in entry['trans'].split():  
                    if char not in unique_chars:               
                        unique_chars[char] = 0
                    unique_chars[char] += 1
        print('u6bcf', unique_chars['u6bcf'])  
        d = cls()
        for char in unique_chars.keys():
            word = char
            count = int(unique_chars[char])
            d.indices[word] = len(d.symbols)
            d.symbols.append(word)
            d.count.append(count)
        return d       
        
                                        
    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return '\n'.join(self.string(t) for t in tensor)

        sent = ' '.join(self[i] for i in tensor)
        return sent
