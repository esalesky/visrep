from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import pygame.freetype
import random

from torch.utils.data.sampler import Sampler
from statistics import mean, median
import gc
import sys
import inspect
import os
from typing import Optional, Iterable, Any
from fontTools.ttLib import TTFont

import logging
LOG = logging.getLogger(__name__)


def traverse_bfs(*objs, marked: Optional[set] = None) -> Iterable[Any]:
    if marked is None:
        marked = set()

    while objs:
        # Get the object's ids
        objs = ((id(o), o) for o in objs)

        # Filter:
        #  - Object that are already marked (using the marked set).
        #  - Type objects such as a class or a module as they are common among all objects.
        #  - Repeated objects (using dict notation).
        objs = {o_id: o for o_id,
                o in objs if o_id not in marked and not isinstance(o, type)}

        # Update the marked set with the ids so we will not traverse them again.
        marked.update(objs.keys())

        # Yield traversed objects
        yield from objs.values()

        # Lookup all the object referred to by the object from the current round.
        # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
        objs = gc.get_referents(*objs.values())


def get_deep_size(*objs) -> int:
    return sum(map(sys.getsizeof, traverse_bfs(*objs)))


def convert_bytes(num):
    """
    this function will convert bytes to MB.... GB... etc
    """
    step_unit = 1000.0  # 1024 bad the size

    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < step_unit:
            return "%3.1f %s" % (num, x)
        num /= step_unit


class ImageGroupSampler(Sampler):
    """Dataset is divided into sub-groups, G_1, G_2, ..., G_k
       Samples Randomly in G_1, then moves on to sample randomly into G_2, etc all the way to G_k

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, rand=True, max_items=-1, fixed_rand=False):
        self.size_group_keys = data_source.size_group_keys
        self.size_groups = data_source.size_groups
        self.num_samples = len(data_source)
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
                    self.rand_perm[g] = torch.randperm(
                        len(self.size_groups[g])).long()
                g_idx_iter = iter(self.rand_perm[g])
            else:
                if self.rand:
                    g_idx_iter = iter(torch.randperm(
                        len(self.size_groups[g])).long())
                else:
                    g_idx_iter = iter(range(len(self.size_groups[g])))

            while True:
                try:
                    g_idx = next(g_idx_iter)
                except StopIteration:
                    break

                n_items += 1
                if self.max_items > 0 and n_items > self.max_items:
                    break
                    # raise StopIteration

                yield self.size_groups[g][g_idx]

    def __len__(self):
        return self.num_samples


def image_collater(batch):
    batch.sort(key=lambda d: d[-1]['width'], reverse=True)

    target_transcription_widths = torch.IntTensor(len(batch))

    trans_ids = []
    meta_groups = []
    meta_seed_text = []
    meta_image_id = []
    for idx, (tensor, transcript, metadata) in enumerate(batch):
        target_transcription_widths[idx] = len(transcript)
        meta_groups.append(metadata['group'])
        meta_seed_text.append(metadata['seed_text'])
        if 'transcription' in metadata:
            trans_ids.append(metadata['transcription'])
        if 'image_id' in metadata:
            meta_image_id.append(metadata['image_id'])

    # batch, sentence, word image
    num_channels, word_height, word_width = batch[0][0][0].shape

    target_transcription = torch.ones(
        len(batch), target_transcription_widths[0], dtype=torch.long)
    for idx, (tensor, transcript, metadata) in enumerate(batch):
        for trans_idx, trans_sample in enumerate(transcript):
            target_transcription[idx, trans_idx] = trans_sample

    target_transcription = target_transcription.flatten()

    input_tensor = torch.ones(len(batch),
                              target_transcription_widths[0], num_channels, word_height, word_width)  # (batch, sentlen, channel, height, width)
    for sample_idx, sentence_sample in enumerate(batch):
        for word_idx, word_sample in enumerate(sentence_sample[0]):
            word_sample = sentence_sample[0][word_idx]
            width = word_sample.shape[2]  # 32, 32, 3
            input_tensor[sample_idx, word_idx, :, :, :width] = word_sample

    batch = {
        'net_input': {
            'src_tokens': input_tensor,
        },
        'target': target_transcription,
        'target_length': target_transcription_widths,
        'seed_text': meta_seed_text,
        'group_id': meta_groups[-1],
        'batch_shape': input_tensor.shape,
        'image_id': meta_image_id,
    }

    LOG.debug('COLLATE: group_id %s', batch['group_id'])
    LOG.debug('COLLATE: src_tokens %s',
              batch['net_input']['src_tokens'].shape)
    LOG.debug('COLLATE: target_length %s', batch['target_length'])
    LOG.debug('COLLATE: target %s', batch['target'])
    LOG.debug('COLLATE: seed_text %s', batch['seed_text'])
    LOG.debug('COLLATE: batch_shape %s', batch['batch_shape'])
    LOG.debug('COLLATE: image_id %s', batch['image_id'])

    return batch


class ImageSynthDataset(Dataset):

    def __init__(self,
                 text_file_path,
                 font_file,
                 font_size=16,
                 surf_width=5000, surf_height=200,
                 start_x=25, start_y=25, dpi=120,
                 pad_size=2,
                 image_height=None,
                 image_width=None,
                 transform=None,
                 alphabet=None,
                 max_text_width=3000,
                 min_text_width=1,
                 image_cache=None,
                 cache_output=None
                 ):

        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.image_height = image_height
        self.image_width = image_width
        self.max_text_width = max_text_width
        self.min_text_width = min_text_width
        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.pad_size = pad_size
        self.dpi = dpi
        self.transform = transform
        self.font_file = font_file
        self.font_size = font_size
        self.alphabet = alphabet
        self.cache_output = cache_output

        self.tokens_list = []
        self.lines = []
        self.sizes = []
        self.append_eos = True
        self.reverse_order = False
        self.read_data(text_file_path, self.alphabet)
        self.size = len(self.tokens_list)

        self.size_group_limits = [2, 5, 10, 15, 20, 30, 40,
                                  50, 70, 100, 200, 300, 500, 750, 1000, np.inf]

        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()
        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        # Now figure out which size-group it belongs in
        drop_large_cnt = 0
        drop_small_cnt = 0
        for idx, text_line in enumerate(self.tokens_list):
            text_width = len(text_line)
            if text_width < self.max_text_width and text_width >= self.min_text_width:
                for cur_limit in self.size_group_limits:
                    if text_width < cur_limit:
                        self.size_groups[cur_limit].append(idx)
                        self.size_groups_dict[cur_limit][idx] = 1
                        break
            elif text_width >= self.max_text_width:
                drop_large_cnt += 1
            elif text_width < self.min_text_width:
                drop_small_cnt += 1
            else:
                LOG.info('...ERROR with data load')

        LOG.info('dropped %d images > %d', drop_large_cnt, self.max_text_width)
        LOG.info('dropped %d images < %d', drop_small_cnt, self.min_text_width)

        # Now get final size (might have dropped large entries!)
        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])
            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max

        LOG.info('final nbr of lines %s', self.nentries)

        if image_cache:
            self.image_cache = image_cache
            LOG.info('...using previous cache')
        else:
            self.image_cache = {}
            self.load_image_cache()

    def read_data(self, path, dictionary):
        LOG.info('loading data %s', path)
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                self.lines.append(line.strip('\n'))
                tokens = dictionary.encode_line(
                    line, add_if_not_exist=False,
                    append_eos=self.append_eos, reverse_order=self.reverse_order,
                ).long()
                self.tokens_list.append(tokens)
                self.sizes.append(len(tokens))

                # if len(self.lines) > 25000:
                #    break
        self.sizes = np.array(self.sizes)
        LOG.info('Total from {}, lines read {}, text length: max {}, min {}, mean {}, median {}'.format(
            path, len(self.lines), max(self.sizes), min(
                self.sizes), int(mean(self.sizes)), int(median(self.sizes))
        ))

    def get_default_image(self, line_text):
        line_text = line_text.replace('▁', '_')

        pad_top = self.pad_size
        pad_bottom = self.pad_size
        pad_left = self.pad_size
        pad_right = self.pad_size

        font = pygame.freetype.Font(self.font_file, self.font_size)

        # get surface
        text_rect_size = font.get_rect(line_text)
        curr_surface_width = self.surface_width
        curr_surface_height = self.surface_height
        if (text_rect_size.width + (self.start_x * 2) + pad_left + pad_right) > self.surface_width:
            curr_surface_width = text_rect_size.width + \
                (self.start_x * 2) + pad_left + pad_right + 20
            LOG.info('...get_default_image, expand surface width %s %s %s',
                     self.surface_width, curr_surface_width, text_rect_size)
        if (text_rect_size.height + (self.start_y * 2) + pad_top + pad_bottom) > self.surface_height:
            curr_surface_height = text_rect_size.height + \
                (self.start_y * 2) + pad_top + pad_bottom + 20
            LOG.info('...get_default_image, expand surface height %s %s %s',
                     self.surface_height, curr_surface_height, text_rect_size)
        surf = pygame.Surface((curr_surface_width, curr_surface_height))

        font.style = pygame.freetype.STYLE_NORMAL
        font.fgcolor = pygame.color.THECOLORS['black']
        surf.fill(pygame.color.THECOLORS['white'])

        text_rect = font.render_to(
            surf, (self.start_x, self.start_y), line_text)

        crop = (self.start_x - pad_left, self.start_y - pad_top,
                text_rect.width + (pad_left + pad_right),
                text_rect.height + (pad_top + pad_bottom))

        sub_surf = surf.subsurface(crop)

        img_data = pygame.surfarray.array3d(sub_surf)
        img_data = img_data.swapaxes(0, 1)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        return img_data

    def load_image_cache(self):
        LOG.info('...building image cache')
        LOG.info('cache_output %s', self.cache_output)

        widths = []
        write_cnt = 0

        for iter_idx, seed_char in enumerate(self.alphabet.symbols):
            seed_idx = self.alphabet.indices[seed_char]
            cv_image = self.get_default_image(seed_char)

            (h, w) = cv_image.shape[:2]
            if h > self.image_height:
                cv_image = self.image_resize(
                    cv_image, height=self.image_height)
            if w > self.image_width:
                cv_image = self.image_resize(
                    cv_image, width=self.image_width)

            cv_image = self.image_pad(
                cv_image, input_pad_height=self.image_height, input_pad_width=self.image_width)

            # sometimes pad is off by a pixel in height or width
            (h, w) = cv_image.shape[:2]
            if h != 32 or w != 32:
                cv_image = cv2.resize(
                    cv_image, (32, 32), interpolation=cv2.INTER_AREA)

            (h, w) = cv_image.shape[:2]
            widths.append(w)

            if iter_idx < 10:
                LOG.info('--->cache %s %s %s %s', iter_idx, seed_idx,
                         seed_char, cv_image.shape)

            self.image_cache[seed_idx] = cv_image

            if self.cache_output:
                outpath = self.cache_output + '/' + \
                    str(seed_idx) + '_' + seed_char + '.png'
                cv2.imwrite(outpath, cv_image)
                write_cnt += 1

            if seed_idx > 0 and seed_idx % 1000 == 0:
                LOG.info('.. loaded images: {}, current cache size {}'.format(
                    len(widths), convert_bytes(
                        get_deep_size(self.image_cache))
                ))

        LOG.info('Cache images: {}, widths: max {}, min {}, mean {}, median {}, cache size {}'.format(
            len(widths), max(widths), min(widths), int(
                mean(widths)), median(widths), convert_bytes(get_deep_size(self.image_cache))
        ))

    def image_resize(self, image, height=None, width=None,
                     inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and grab the
        # image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the dimensions
            r = height / float(h)
            w_r = int(w * r)
            if w_r < 1:
                w_r = 1
            dim = (w_r, height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            h_r = int(h * r)
            if h_r < 1:
                h_r = 1
            dim = (width, h_r)

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]
        return resized

    def image_pad(self, img_data, input_pad_height=None, input_pad_width=None):
        img_height, img_width = img_data.shape[:2]

        if input_pad_height is None:
            pad_height = 0
        else:
            pad_height = input_pad_height - img_height

        if input_pad_width is None:
            pad_width = 0
        else:
            pad_width = input_pad_width - img_width

        top, bottom = pad_height//2, pad_height-(pad_height//2)

        border_color = [255, 255, 255]
        try:
            img_data_pad = cv2.copyMakeBorder(img_data,
                                              top, bottom, 0, pad_width, cv2.BORDER_CONSTANT,
                                              value=border_color)
        except Exception as e:
            LOG.info('EXCEPT image_pad %s %s', img_height, img_width)
            img_data_pad = img_data
        return img_data_pad

    def __getitem__(self, idx):
        transcription = self.tokens_list[idx].tolist()
        sent_list = []
        for word_idx in transcription:
            sent_list.append(self.alphabet[word_idx])

        seed_text = ''.join(sent_list)

        max_width = 0
        for cur_limit in self.size_group_limits:
            if idx in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break
        group_id = max_width

        line_image = []
        for word_idx in transcription:
            cv_image = self.image_cache[word_idx]
            cv_image = self.transform(cv_image)
            line_image.append(cv_image)

        metadata = {
            'idx': idx,
            'transcription': transcription,
            'seed_text': seed_text,
            'width': len(transcription),
            'group': group_id,
            'image_id': idx,
        }

        LOG.debug('GETITEM: group %s', metadata['group'])
        LOG.debug('GETITEM: image_id %s', idx)
        LOG.debug('GETITEM: width (transcription) %s', len(transcription))
        LOG.debug('GETITEM: image shape %s', len(line_image))
        LOG.debug('GETITEM: seed text %s', seed_text)
        LOG.debug('GETITEM: src_ids (transcription) %s', transcription)
        LOG.debug('GETITEM: src_length %s', len(sent_list))
        LOG.debug('GETITEM: src_tokens %s', sent_list)

        return line_image, transcription, metadata

    def __len__(self):
        return self.nentries
