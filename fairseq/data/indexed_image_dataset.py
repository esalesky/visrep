
import os
import struct

import numpy as np
import torch

from fairseq.tokenizer import Tokenizer, tokenize_line
from fairseq.data.image_generator import ImageGenerator
import pdb


class IndexedImageWordDataset(torch.utils.data.Dataset):

    def __init__(self, path, dictionary, image_verbose=False,
                 image_font_path=None, image_font_size=16,
                 image_width=150, image_height=30,
                 word_encoder=True, append_eos=True, reverse_order=False):
        print('...loading IndexedImageWordDataset')
        self.tokens_list = []
        self.word_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)
        self.word_encoder = word_encoder
        self.image_verbose = image_verbose
        self.image_generator = ImageGenerator(image_font_path, image_font_size,
                                              image_width, image_height)

    def read_data(self, path, dictionary):
        print('IndexedImageWordDataset loading data %s' % (path))
        drop_cnt = 0
        with open(path, 'r', encoding='utf-8') as f:
            for line_ctr, line in enumerate(f):
                self.lines.append(line.strip('\n'))

                words = tokenize_line(line)
                if line_ctr % 10000 == 0:
                    print(line_ctr, len(words), line.strip('\n'))

                # if len(words) > 30:
                #    drop_cnt += 1
                #    continue

                self.word_list.append(words)

                tokens = Tokenizer.tokenize(
                    line, dictionary, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                # print(words)

                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))

                # if line_ctr > 5000:
                #    print('....Only loading 5000 lines in src')
                #    break

        self.sizes = np.array(self.sizes)
        print('...data load complete, total lines %d, dropped %d' % (len(self.lines), drop_cnt))

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, i):
        self.check_index(i)

        img_data_list = []
        # img_data_width_list = []

        for word in self.word_list[i]:
            img_data, img_data_width = self.image_generator.get_default_image(word)

            img_data_list.append(img_data)

            # img_data_width_list.append(img_data_width)

        return self.tokens_list[i], self.word_list[i], self.lines[i], img_data_list  # , img_data_width_list

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)


class IndexedImageLineDataset(torch.utils.data.Dataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(self, path, dictionary, font_file_path, font_size=16,
                 word_encoder=True, append_eos=True, reverse_order=False):
        self.tokens_list = []
        self.word_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)
        self.word_encoder = word_encoder
        self.image_generator = ImageGenerator(font_file_path, font_size)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                # print(line.strip('\n'))

                words = tokenize_line(line)
                self.word_list.append(words)

                tokens = Tokenizer.tokenize(
                    line, dictionary, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                # print(words)

                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    def __getitem__(self, i):
        self.check_index(i)

        img_data, img_data_width = self.image_generator.get_default_image(
            self.lines[i])

        return self.tokens_list[i], self.lines[i], img_data, img_data_width

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    @staticmethod
    def exists(path):
        return os.path.exists(path)

