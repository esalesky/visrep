from functools import lru_cache
from . import FairseqDataset
import cv2
import pygame.freetype
import os
import sys
import random

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from . import data_utils, FairseqDataset
import torchvision.transforms as transforms


def image_collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True, image_type=None
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            print("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    src_images_tensor = None
    if samples[0]['source_image']:
        num_channels, word_height, word_width = samples[0]['source_image'][0].shape
#        print(f"source image: {samples[0]['source_image'][0].shape} from word ID {samples[0]['source'][0]}")
        
        src_images_tensor = None
        if image_type == "word":
            # In this situation, words are padded to the same width and height
            src_images_tensor = torch.zeros(len(samples), len(
                src_tokens[0]), num_channels, word_height, word_width)  # (batch, sentlen, channel, height, width)
            max_word = len(src_tokens[0])

            for sample_idx, sentence_sample in enumerate(samples):
                len_sentence = len(sentence_sample['source_image'])
                for word_idx, word_sample in enumerate(sentence_sample['source_image']):
                    width = word_sample.shape[2]
                    src_images_tensor[sample_idx, word_idx:, :, :, :width] = word_sample

        elif image_type == "line":
            # Words are padded to same height, but widths differ
            widths = [sum([word.shape[2] for word in sample['source_image']]) for sample in samples]
            max_sentence_width = max(widths)
            print(f"BATCH max width is {max_sentence_width}", file=sys.stderr)

            # pad sentences to maximum width observed
            src_images_tensor = torch.zeros(len(samples), 1, num_channels, word_height, max_sentence_width)
            
            for i, sample in enumerate(samples):
                # concatenate all the images along the width
                sentence = torch.cat(sample['source_image'], 2)
#                print(f"SAMPLE[{i}] has shape {sentence.shape}")
                # assign to the array
                src_images_tensor[i, 0, :, :, 0:widths[i]] = sentence

        src_images_tensor = src_images_tensor.index_select(0, sort_order)
        print(f"SRC_IMAGES_TENSOR: {src_images_tensor.shape}")

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor(
            [s['target'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_images': src_images_tensor,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    if samples[0].get('alignment', None) is not None:
        bsz, tgt_sz = batch['target'].shape
        src_sz = batch['net_input']['src_tokens'].shape[1]

        offsets = torch.zeros((len(sort_order), 2), dtype=torch.long)
        offsets[:, 1] += (torch.arange(len(sort_order),
                                       dtype=torch.long) * tgt_sz)
        if left_pad_source:
            offsets[:, 0] += (src_sz - src_lengths)
        if left_pad_target:
            offsets[:, 1] += (tgt_sz - tgt_lengths)

        alignments = [
            alignment + offset
            for align_idx, offset, src_len, tgt_len in zip(sort_order, offsets, src_lengths, tgt_lengths)
            for alignment in [samples[align_idx]['alignment'].view(-1, 2)]
            if check_alignment(alignment, src_len, tgt_len)
        ]

        if len(alignments) > 0:
            alignments = torch.cat(alignments, dim=0)
            align_weights = compute_alignment_weights(alignments)

            batch['alignments'] = alignments
            batch['align_weights'] = align_weights

    return batch


class ImagePairDataset(FairseqDataset):
    """
    Pairs an ImageDataset for the source side with a standard torch.utils.data.Dataset for the target side.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        image_type: type of image processing (word or line)
    """

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None, image_type=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.image_type = image_type

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        # src_item: list of token IDS
        # src_img_list: list of three-channel image vectors of shape (3, width, height)
        src_item, src_img_list = self.src[index]

        #src_images = []
        # for src_img in src_img_list:
        #    src_images.append(src_img)

        # image pair torch.Size([3, 32, 128])
        # index 149
        # torch.Size([149])
        # torch.Size([147])

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat(
                    [self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
            'source_image': src_img_list,
        }

        # print(src_item.shape)  # torch.Size([23])
        # print(tgt_item.shape)  # torch.Size([21])

        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]
        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return image_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding, image_type=self.image_type
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(
                self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)


class ImageGenerator():

    def __init__(self,
                 font_file_path,
                 surf_width=1500, surf_height=250,
                 start_x=50, start_y=50, dpi=120,
                 image_height=128, image_width=32,
                 font_size=8,
                 pad_right=None):

        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.dpi = dpi

        self.font_rotation = [-6, -4, -2, 0, 2, 4, 6]
        self.pad_top = [0]
        self.pad_bottom = [0]
        self.pad_left = [0]
        self.pad_right = [0]
#        self.pad_top = [0, 2, 4, 6, 8]
#        self.pad_bottom = [0, 2, 4, 6, 8]
#        self.pad_left = [0, 2, 4, 6, 8]
#        self.pad_right = pad_right if pad_right else random.choice([0, 2, 4, 6, 8])
        self.font_sizes = [font_size] if font_size else [10, 14, 18, 24, 32]
        self.font_color = ['black']
        self.bkg_color = ['white']

        self.image_height = image_height
        self.image_width = image_width

        if font_file_path is not None:
            self.font_list = self.get_font_list(font_file_path)

    def get_font_list(self, font_file_path):
        fontlist = []
        fontcnt = 0
        print('...loading fonts from %s' % font_file_path)
        with open(font_file_path, 'r') as file:  # , encoding='utf8') as file:
            for ctr, line in enumerate(file.readlines()):
                fontname = line.strip()
                fontcnt += 1
                fontlist.append(fontname)
        print('Found %d fonts' % (len(fontlist)))
        return fontlist

    def image_resize(self, image, width=None, height=None,
                     inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and grab the
        # image size
        dim = None
        (h, w) = image.shape[:2]
        # print(h,w,height,width)

        # if both the width and height are None, then return the original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            dim = (int(w * r), height)
            # print('resize height to ', height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))
            # print('resize width to ', width)

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]
        return resized

    def resize_or_pad(self, img_data, height, width=None):
        """
        For line-based decoding, we don't want to change the width.
        """
        img_height, img_width = img_data.shape[:2]
        # print('input h, w', img_height, img_width)
        if img_height > height:
            img_data = self.image_resize(img_data, height=height)
            img_height, img_width = img_data.shape[:2]
        # print('height resize h, w', img_height, img_width)

        # Only adjust width if a requested width was passed (i.e., for word-based embeddings)
        if width:
            if img_width > width:
                img_data = self.image_resize(img_data, width=width)
                img_height, img_width = img_data.shape[:2]
                # print('width resize h, w', img_height, img_width)

            img_height, img_width = img_data.shape[:2]
            pad_height = height - img_height
            pad_width = width - img_width

            border_color = [255, 255, 255]
            # border_color = [0, 0, 0]

            # print('img h w', img_height, img_width)
            # print('pad h w',pad_height, pad_width)
            img_data = cv2.copyMakeBorder(
                img_data, pad_height, 0, 0, pad_width, cv2.BORDER_CONSTANT,
                value=border_color)

        return img_data

    def get_image(self, line_text,
                  font_name=None, font_size=None, font_style=None,
                  font_color=None, bkg_color=None, font_rotate=None,
                  pad_top=None, pad_bottom=None, pad_left=None, pad_right=None):
        ''' Create pygame surface '''

        # Replace Unicode Character 'LOWER ONE EIGHTH BLOCK' (U+2581)
        # many of the fonts can not render this code
        # TODO
        line_text = line_text.replace('▁', '_')

        surf = pygame.Surface((self.surface_width, self.surface_height))

        if font_name:
            font_name = font_name
        else:
            font_name = random.choice(self.font_list)

        if font_size:
            font = pygame.freetype.Font(font_name, font_size)
        else:
            font = pygame.freetype.Font(
                font_name, random.choice(self.font_sizes))

        if font_style:
            font_style = font_style
        else:
            font_style = random.randint(1, 6)

        if font_style == 1:
            font.style = pygame.freetype.STYLE_NORMAL
        elif font_style == 2:
            font.style = pygame.freetype.STYLE_OBLIQUE
        elif font_style == 3:
            font.style = pygame.freetype.STYLE_STRONG
        else:
            font.style = pygame.freetype.STYLE_DEFAULT

        if font_color:
            font.fgcolor = pygame.color.THECOLORS[font_color]
        else:
            font.fgcolor = pygame.color.THECOLORS[random.choice(
                self.font_color)]

        if font_rotate:
            font.rotation = font_rotate
        else:
            if font_rotate != 0:
                font_rotate_val = random.choice(self.font_rotation)
                if font_rotate != 0:
                    font.rotation = font_rotate_val

        if bkg_color:
            surf.fill(pygame.color.THECOLORS[bkg_color])
        else:
            surf.fill(pygame.color.THECOLORS[random.choice(self.font_color)])

        text_rect = font.render_to(
            surf, (self.start_x, self.start_y), line_text)

        if pad_top:
            pad_top = pad_top
        else:
            pad_top = random.choice(self.pad_top)

        if pad_bottom:
            pad_bottom = pad_bottom
        else:
            pad_bottom = random.choice(self.pad_bottom)

        if pad_left:
            pad_left = pad_left
        else:
            pad_left = random.choice(self.pad_left)

        if not pad_right:
            pad_right = self.pad_right

        crop = (self.start_x - pad_left, self.start_y - pad_top,
                text_rect.width + (pad_left + pad_right),
                max(self.image_height, text_rect.height + (pad_top + pad_bottom)))

        sub_surf = surf.subsurface(crop)

        img_data = pygame.surfarray.array3d(sub_surf)
        # print(img_data.shape)
        img_data = img_data.swapaxes(0, 1)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        # print(img_data.shape)

        return img_data


class ImageDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory.

    :param path:
    :param dictionary:
    :param append_eos:
    :param reverse_order:
    :param transform:
    :param image_type: word or sentence
    :param image_font_path:
    :param image_height:
    :param image_width:
    :param image_cache:
    """

    def __init__(self, path, dictionary, append_eos=True, reverse_order=False,
                 transform=None,
                 font_size=8,
                 image_type="",
                 image_font_path=None,
                 image_height=30,
                 image_width=120,
                 image_cache=None
                 ):

        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = append_eos
        self.reverse_order = reverse_order
        self.dictionary = dictionary
        self.read_data(path, dictionary)
        self.size = len(self.tokens_list)
        self.image_type = image_type.lower()
        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform
        self.font_size = font_size
        self.image_cache = image_cache
        self.image_use_cache = True if self.image_cache else False

        if self.image_type is not None and not self.image_use_cache:
            self.image_generator = ImageGenerator(font_file_path=image_font_path,
                                                  image_width=image_width,
                                                  image_height=image_height)

    def read_data(self, path, dictionary):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))
        self.sizes = np.array(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self.size:
            raise IndexError('index out of range')

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        self.check_index(i)

        img_data_list = None
        if self.image_type == "word":
            img_data_list = []
            for id in self.tokens_list[i]:
                if self.image_use_cache:
                    word = self.dictionary[id]
                    if word in self.image_cache:
                        img_data = self.image_cache[word]
                    else:
                        print('...MISSING word in cache', word)
                        img_data_list.append(torch.zeros(
                            3, self.image_height, self.image_width))
                else:
                    img_data = self.image_generator.get_image(
                        word, font_color='black', font_size=self.font_size,
                        height=self.image_height, bkg_color='white')
                    img_data = self.image_generator.resize_or_pad(
                        img_data, height=self.image_height, width=self.image_width)

                # print(cv_resize_image.shape) # (32, 128, 3) H, W, C
                # print('image dataset resize', img_data.shape)

                img_tensor = self.transform(img_data)

                # Resnet expects shape (C x H x W)
                # cv2 gen image (34, 43, 3) (H X W X C)
                # image resize (32, 128, 3)  (H X W X C)
                # image tensor torch.Size([3, 32, 128]) (C x H x W)

                img_data_list.append(img_tensor)

        elif self.image_type == "line":
            assert self.image_use_cache
            img_data = [self.image_cache[self.dictionary[word]] for word in self.tokens_list[i]]
            img_data_list = [self.transform(image) for image in img_data]

        return self.tokens_list[i], img_data_list

    def get_original_text(self, i):
        self.check_index(i)
        return self.lines[i]

    def __del__(self):
        pass

    def __len__(self):
        return self.size

    def num_tokens(self, index):
        return self.sizes[index]

    def size(self, index):
        return self.sizes[index]

    @staticmethod
    def exists(path):
        return os.path.exists(path)
