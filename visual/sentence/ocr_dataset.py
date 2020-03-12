from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import pygame.freetype
import random
import logging
from alphabet import Alphabet
from torch.utils.data.sampler import Sampler
from statistics import mean, median
import json
import lmdb
import gc
import sys
import inspect
import os
from typing import Optional, Iterable, Any
from text_utils import utf8_to_uxxxx, uxxxx_to_utf8
from fontTools.ttLib import TTFont

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


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

        # print('...GroupedSampler stopiteration', n_items, g)
        # raise StopIteration

    def __len__(self):
        return self.num_samples


def image_collater(batch, image_verbose):
    # _use_shared_memory = True

    # image_verbose = True

    # Sort by tensor width
    batch.sort(key=lambda d: d[-1]['width'], reverse=True)

    # Deal with use_shared_memory???
    # Does this to extra copies???
    biggest_size = batch[0][0].size()

    input_tensor = torch.zeros(
        len(batch), biggest_size[0], biggest_size[1], biggest_size[2])
    input_tensor_widths = torch.IntTensor(len(batch))
    target_transcription_widths = torch.IntTensor(len(batch))

    trans_ids = []
    meta_heights = []
    meta_widths = []
    meta_groups = []
    meta_seed_text = []
    meta_image_id = []
    for idx, (tensor, transcript, metadata) in enumerate(batch):
        width = tensor.size(2)
        input_tensor[idx, :, :, :width] = tensor
        # image may come to us already padded
        input_tensor_widths[idx] = metadata['width']
        target_transcription_widths[idx] = len(transcript)

        meta_heights.append(metadata['height'])
        meta_widths.append(metadata['width'])
        meta_groups.append(metadata['group'])
        meta_seed_text.append(metadata['seed_text'])

        if 'transcription' in metadata:
            trans_ids.append(metadata['transcription'])

        if 'image_id' in metadata:
            meta_image_id.append(metadata['image_id'])

    target_transcription = torch.IntTensor(
        target_transcription_widths.sum().item())
    cur_offset = 0
    for idx, (tensor, transcript, metadata) in enumerate(batch):
        for j, char in enumerate(transcript):
            target_transcription[cur_offset] = char
            cur_offset += 1

    batch = {
        'heights': meta_heights,
        'widths': meta_widths,
        'batch_img_width': biggest_size[2],
        'net_input': {
            'src_tokens': input_tensor,
            'src_widths': input_tensor_widths,
        },
        'target': target_transcription,
        'target_length': target_transcription_widths,
        'seed_text': meta_seed_text,
        'group_id': meta_groups[-1],
        'batch_shape': input_tensor.shape,
        'image_id': meta_image_id,
    }

    if image_verbose:
        print('\nCOLLATE: group_id', batch['group_id'])
        print('COLLATE: heights', batch['heights'])
        print('COLLATE: widths', batch['widths'])
        print('COLLATE: src_widths',
              batch['net_input']['src_widths'])
        print('COLLATE: target_length', batch['target_length'])
        print('COLLATE: target', batch['target'])
        print('COLLATE: seed_text', batch['seed_text'])
        print('COLLATE: batch_shape', batch['batch_shape'])

    return batch


class LineSynthDataset(Dataset):

    def __init__(self,
                 text_file_path, font_file_path, bkg_file_path,
                 train_cache_output=None,
                 surf_width=5000, surf_height=200,
                 start_x=25, start_y=25, dpi=120,
                 image_height=32,
                 max_cache_write=5000,
                 max_seed=2500000,
                 transform=None,
                 alphabet=None,
                 save_images=False,
                 image_verbose=False,
                 use_image_cache=False,
                 use_background_image=True,
                 use_default_image=False,
                 split_text=False,
                 augment=False,
                 use_font_chars=False,
                 max_image_cache=250000,
                 min_text_width=1,
                 max_text_width=3000,
                 max_image_width=1000,
                 min_image_width=32,
                 max_len_text_rotate=60,
                 max_len_text_font=100,
                 background_mod=3,
                 font_rotation=[0, 1, 2, -2, -1],
                 pad_top=[0, 1, 2, 3, 4, 5, 6],
                 pad_bottom=[0, 1, 2, 3, 4, 5, 6],
                 pad_left=[0, 1, 2, 3, 4, 5, 6],
                 pad_right=[0, 1, 2, 3, 4, 5, 6],
                 font_size=[12],  # , 12, 16],
                 font_color_list=['black', 'white', 'red',
                                  'green', 'blue', 'yellow'],
                 bkg_color_list=['white', 'black', 'red',
                                 'green', 'blue', 'yellow']
                 ):

        print('.....init linesynth')
        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.use_image_cache = use_image_cache
        self.use_background_image = use_background_image
        self.split_text = split_text
        self.max_seed = max_seed
        self.max_cache_write = max_cache_write
        self.save_images = save_images
        self.max_image_cache = max_image_cache
        self.train_cache_output = train_cache_output
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.min_image_width = min_image_width
        self.image_verbose = image_verbose
        self.max_text_width = max_text_width
        self.min_text_width = min_text_width
        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.dpi = dpi
        self.max_len_text_rotate = max_len_text_rotate
        self.max_len_text_font = max_len_text_font
        self.use_default_image = use_default_image
        self.augment = augment
        self.use_font_chars = use_font_chars

        self.font_rotation = font_rotation
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.font_size = font_size
        self.font_color_list = font_color_list
        self.bkg_color_list = bkg_color_list
        self.background_mod = background_mod

        self.transform = transform

        self.font_list = self.get_font_list(font_file_path)
        self.text_list = self.get_text_list(text_file_path)
        if self.use_font_chars:
            self.get_font_chars(self.font_list[0], self.text_list)

        self.text_list.sort()
        self.bkg_list = self.get_background_list(bkg_file_path)

        # Init alphabet
        alphabet_expand_cnt = 0
        if alphabet is not None:
            print(
                "Explicitly providing alphabet via initializatin parameter, as opposed to inferring from data")
            self.alphabet = alphabet

            for idx, text_line in enumerate(self.text_list):
                uxxxx_text_list = utf8_to_uxxxx(text_line, output_array=True)
                for char in uxxxx_text_list:
                    if char not in self.alphabet.char_to_idx:
                        self.alphabet.max_id += 1
                        self.alphabet.char_to_idx[char] = self.alphabet.max_id
                        self.alphabet.idx_to_char[self.alphabet.max_id] = char
                        # print('...expanding Alphabet, adding id %d, char %s' % (
                        #    self.alphabet.max_id, char))
                        alphabet_expand_cnt += 1

        else:
            self.init_alphabet()

        if alphabet_expand_cnt > 0:
            print('...expanded alphabet, added ', alphabet_expand_cnt)
        print('Alphabet size ', len(self.alphabet))

        # this is text size
        self.size_group_limits=[2, 5, 10, 15, 20, 30, 40,
                                  50, 70, 100, 200, 300, 500, 750, 1000, np.inf]

        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()
        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        # Now figure out which size-group it belongs in
        for idx, text_line in enumerate(self.text_list):
            text_width = len(text_line)
            for cur_limit in self.size_group_limits:
                if text_width < cur_limit and text_width < self.max_text_width and text_width >= self.min_text_width:
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break

        # Now get final size (might have dropped large entries!)
        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])
            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max

        # self.size_group_limits.sort(reverse = True)

        if self.use_image_cache:
            self.image_cache = {}
            self.load_image_cache()

    def get_text_list(self, input_text):
        """ Load input text """
        text_list = []
        line_cnt = 0

        with open(input_text, 'r') as file:
            for orig_line in file.readlines():
                line_cnt += 1
                line = orig_line.strip().replace('\n', ' ').replace('\r', '')
                # line = orig_line.split()
                # line = line[0].strip().replace('\n', ' ').replace('\r', '')
                if self.split_text:
                    for txt in list(self.chunkstring(line, self.max_text_width)):
                        text_list.append(txt)
                else:
                    text_list.append(line)
                if line_cnt >= self.max_seed:
                    print('...max seed text input reached', self.max_seed)
                    break

        len_list = [len(x) for x in text_list]

        LOG.info('Total from {}, read {}, included {}, max {}, min {}, mean {}, median {}'.format(
            input_text, line_cnt, len(text_list), max(len_list), min(
                len_list), int(mean(len_list)), median(len_list)
        ))
        return text_list

    def chunkstring(self, string, length):
        return (string[0+i: length+i] for i in range(0, len(string), length))

    def get_font_chars(self, font_file_path, text_list):
        try:
            font = TTFont(font_file_path)
            cmap = font.getBestCmap()
            for char_idx, char in enumerate(sorted(cmap)):
                # print(char_idx, chr(char), utf8_to_uxxxx(chr(char)))
                text_list.append(chr(char))
        except Exception as e:
            print("Failed to read", font_file_path)
            print(e)

    def get_font_list(self, font_file_path):
        fontlist = []
        fontcnt = 0
        LOG.info('...loading fonts from %s', font_file_path)
        with open(font_file_path, 'r') as file:  # , encoding='utf8') as file:
            for ctr, line in enumerate(file.readlines()):
                fontname = line.strip()
                fontcnt += 1
                fontlist.append(fontname)
                # break
        LOG.info('Found %d fonts', len(fontlist))
        return fontlist

    def get_background_list(self, input_bkg):
        linecnt = 0
        background_list = []
        LOG.info('...loading backgrounds from %s', input_bkg)
        with open(input_bkg, 'r') as file:  # , encoding='utf8') as file:
            for line in file.readlines():
                linecnt += 1
                line = line.strip()
                if len(line) > 0:
                    background_list.append(line)
                    # if len(background_list) > 200:
                    #    break
        LOG.info('Backgrounds read %d, added %d',
                 linecnt, len(background_list))
        return background_list

    def init_alphabet(self):
        # Read entire train/val/test data to deterimine set of unique characters we should have in alphabet
        unique_chars = set()

        for idx, text_line in enumerate(self.text_list):
            uxxxx_text_list = utf8_to_uxxxx(text_line, output_array=True)
            for char in uxxxx_text_list:
                unique_chars.add(char)

        # Now add CTC blank as first letter in alphabet. Also sort alphabet lexigraphically for convinience
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

    def get_default_image(self, line_text):
        line_text = line_text.replace('▁', '_')

        pad_top = 2
        pad_bottom = 2
        pad_left = 2
        pad_right = 2

        font = pygame.freetype.Font(self.font_list[0], 8)

        # get surface
        text_rect_size = font.get_rect(line_text)
        curr_surface_width = self.surface_width
        curr_surface_height = self.surface_height
        if (text_rect_size.width + (self.start_x * 2) + pad_left + pad_right) > self.surface_width: 
            curr_surface_width = text_rect_size.width + (self.start_x * 2) + pad_left + pad_right + 20
            print('...get_default_image, expand surface width', self.surface_width, curr_surface_width, text_rect_size)
        if (text_rect_size.height + (self.start_y * 2) + pad_top + pad_bottom) > self.surface_height: 
            curr_surface_height = text_rect_size.height + (self.start_y * 2) + pad_top + pad_bottom + 20
            print('...get_default_image, expand surface height', self.surface_height, curr_surface_height, text_rect_size)
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

    def get_image(self, line_text,
                  font_name=None, font_size=None, font_style=None,
                  font_color=None, bkg_color=None, font_rotate=None,
                  bkg_image=None,
                  pad_top=None, pad_bottom=None, pad_left=None, pad_right=None):

        #surf = pygame.Surface((self.surface_width, self.surface_height))

        # Replace Unicode Character 'LOWER ONE EIGHTH BLOCK' (U+2581)
        # many of the fonts can not render this code
        line_text = line_text.replace('▁', '_')

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

        if pad_right:
            pad_right = pad_right
        else:
            pad_right = random.choice(self.pad_right)

        if font_name:
            font_name = font_name
        else:
            font_name = random.choice(self.font_list)

        if len(line_text) < self.max_len_text_font:
            if font_size:
                font = pygame.freetype.Font(font_name, font_size)
            else:
                font = pygame.freetype.Font(
                    font_name, random.choice(self.font_size))
        else:
            font = pygame.freetype.Font(font_name, 8)

        # get surface
        text_rect_size = font.get_rect(line_text)
        curr_surface_width = self.surface_width
        curr_surface_height = self.surface_height
        if (text_rect_size.width + (self.start_x * 2) + pad_left + pad_right) > self.surface_width: 
            curr_surface_width = text_rect_size.width + (self.start_x * 2) + pad_left + pad_right + 20
            print('...get_default_image, expand surface width', self.surface_width, curr_surface_width, text_rect_size)
        if (text_rect_size.height + (self.start_y * 2) + pad_top + pad_bottom) > self.surface_height: 
            curr_surface_height = text_rect_size.height + (self.start_y * 2) + pad_top + pad_bottom + 20
            print('...get_default_image, expand surface height', self.surface_height, curr_surface_height, text_rect_size)
        surf = pygame.Surface((curr_surface_width, curr_surface_height))

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
            font_color = random.choice(self.font_color_list)
            font.fgcolor = pygame.color.THECOLORS[font_color]

        if bkg_color:
            surf.fill(pygame.color.THECOLORS[bkg_color])
        else:
            all_bkg_color = self.bkg_color_list.copy()
            all_bkg_color.remove(font_color)
            bkg_color = random.choice(all_bkg_color)
            # print(font_color, all_bkg_color, all_bkg_color_remove)
            surf.fill(pygame.color.THECOLORS[bkg_color])

        if len(line_text) < self.max_len_text_rotate:
            if font_rotate:
                font.rotation = font_rotate
            else:
                if font_rotate != 0:
                    font_rotate_val = random.choice(self.font_rotation)
                    if font_rotate != 0:
                        font.rotation = font_rotate_val

        if self.use_background_image:
            background_random = range(self.background_mod)
            rand_back_choice = random.choice(background_random)
            # print(background_random, rand_back_choice)
            if rand_back_choice == 0:
                if bkg_image:
                    background = pygame.image.load(bkg_image)
                    background = pygame.transform.scale(
                        background, (self.surface_width, self.surface_height))
                    surf.blit(background, (0, 0))
                else:
                    rand_bkg_image = random.choice(self.bkg_list)
                    background = pygame.image.load(rand_bkg_image)
                    background = pygame.transform.scale(
                        background, (self.surface_width, self.surface_height))
                    surf.blit(background, (0, 0))

        text_rect = font.render_to(
            surf, (self.start_x, self.start_y), line_text)

        crop = (self.start_x - pad_left, self.start_y - pad_top,
                text_rect.width + (pad_left + pad_right),
                text_rect.height + (pad_top + pad_bottom))

        #try:
        sub_surf = surf.subsurface(crop)

        img_data = pygame.surfarray.array3d(sub_surf)
        img_data = img_data.swapaxes(0, 1)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
        #except Exception as e:
        #    print('-> ERROR %s text %d, font size %d, font name %s, rectangle %s, crop %s' %
        #          (e, len(line_text), font.size, font.name, text_rect, crop))
        #
        #    img_data = self.get_default_image(line_text)
        
        return img_data

    def load_image_cache(self):
        print('...building image cache, max_image_cache',
              self.max_image_cache, self.max_cache_write, self.train_cache_output, self.save_images)

        if self.train_cache_output and self.save_images:
            print('...save images to ', self.train_cache_output)
        else:
            print('...do not save images',
                  self.train_cache_output, self.save_images)

        widths = []
        write_cnt = 0
        for idx, seed_text in enumerate(self.text_list):
            cv_image = self.get_image(seed_text,
                                      font_name=self.font_list[0], font_size=8, font_style=1,
                                      font_color='black', bkg_color='white', font_rotate=0,
                                      pad_top=2, pad_bottom=2, pad_left=2, pad_right=2)

            # cv_resize_image = image_resize(
            #    cv_image, height=self.image_height)
            cv_resize_image = self.resize_or_pad(cv_image)

            (h, w) = cv_resize_image.shape[:2]
            widths.append(w)
            self.image_cache[idx] = cv_resize_image

            if self.train_cache_output and self.save_images:
                # print('write image', self.train_cache_output, write_cnt)
                if self.max_cache_write < write_cnt:
                    outpath = self.train_cache_output + '/' + \
                        str(idx) + '_h' + str(h) + '_w' + \
                        str(w) + '_' + seed_text + '.png'
                    # print('....save image', outpath)
                    cv2.imwrite(outpath, cv_resize_image)
                write_cnt += 1

            if idx % 10000 == 0:
                LOG.info('.. loaded images: {}, current cache size {}'.format(
                    len(widths), convert_bytes(get_deep_size(self.image_cache))
                ))

            if idx > self.max_image_cache:
                print('reached max_image_cache')
                break

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
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]
        return resized

    def resize_or_pad(self, img_data):
        img_height, img_width = img_data.shape[:2]

        if img_height > self.image_height:
            img_data = self.image_resize(
                img_data, height=self.image_height)

        img_height, img_width = img_data.shape[:2]

        if img_width > self.max_image_width:
            img_data = self.image_resize(
                img_data, width=self.max_image_width)

        img_height, img_width = img_data.shape[:2]

        if img_height < self.image_height:
            img_data = self.image_pad(
                img_data, input_pad_height=self.image_height)

        img_height, img_width = img_data.shape[:2]

        if img_width < self.min_image_width:
            img_data = self.image_pad(
                img_data, input_pad_width=self.min_image_width)

        return img_data

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
            print('EXCEPT image_pad', img_height, img_width)
            img_data_pad = img_data
            # img_height, img_width = img_data_pad.shape[:2]
        # print('->image_pad', img_height, img_width)
        return img_data_pad

    def __getitem__(self, idx):
        # rint('.......getitem,,,,')
        seed_text = self.text_list[idx]
        if self.image_verbose:
            print('\nGETITEM:', idx, seed_text)

        max_width = 0
        for cur_limit in self.size_group_limits:
            if idx in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break
        group_id = max_width

        if self.use_image_cache:
            if idx in self.image_cache:
                cv_resize_image = self.image_cache[idx]
            else:
                if self.use_default_image:
                    cv_image = self.get_default_image(seed_text)
                else:
                    cv_image = self.get_image(seed_text,
                                              font_name=self.font_list[0], font_size=8, font_style=1,
                                              font_color='black', bkg_color='white', font_rotate=0,
                                              pad_top=2, pad_bottom=2, pad_left=2, pad_right=2)

                cv_resize_image = self.resize_or_pad(cv_image)

        else:
            if self.use_default_image:
                cv_image = self.get_default_image(seed_text)
            else:
                cv_image = self.get_image(seed_text, font_style=1)
                            # pad_top=0, pad_bottom=0, pad_left=0, pad_right=0)



            if self.image_verbose:
                print('GETITEM: image size generate', cv_image.shape)

            cv_resize_image = self.resize_or_pad(cv_image)


            if self.image_verbose:
                print('GETITEM: image size after pad', cv_resize_image.shape)

        # if self.augment:
        resize_rand = random.randint(0, 4)
        (height, width) = cv_resize_image.shape[:2]
        if resize_rand == 0:
            # print('...wide', seed_text, height, width)
            if width < 700:
                cv_resize_image = cv2.resize(cv_resize_image,
                                            (int(width * 1.2), self.image_height),
                                            interpolation=cv2.INTER_AREA)
                                            
                # (height, width) = cv_resize_image.shape[:2]
                # print('...resize to', seed_text, height, width)
            # if resize_rand == 1:
            #     #print('...shrink', seed_text, height, width)
            #     cv_resize_image = cv2.resize(cv_resize_image,
            #                                 (int(width * 0.25), self.image_height),
            #                                 interpolation=cv2.INTER_AREA)
            #     # pad incase image is less than min
            #     cv_resize_image = self.resize_or_pad(cv_resize_image)
            #     (height, width) = cv_resize_image.shape[:2]
            #     #print('...resize to', seed_text, height, width)

        line_image = self.transform(cv_resize_image)

        original_width = line_image.size(2)
        original_height = line_image.size(1)

        transcription = []
        uxxxx_text_list = utf8_to_uxxxx(seed_text, output_array=True)
        for char in uxxxx_text_list:
            transcription.append(self.alphabet.char_to_idx[char])

        metadata = {
            'idx': idx,
            'transcription': transcription,
            'seed_text': seed_text,
            'width': original_width,
            'height': original_height,
            'group': group_id,
            'image_id': idx,
        }

        if self.image_verbose:
            print('GETITEM: group', metadata['group'])
            print('GETITEM: image shape', line_image.shape)
            print('GETITEM: target_ids', transcription)
            print('GETITEM: seed text', seed_text)
            print('GETITEM: target_length', len(transcription))

        return line_image, transcription, metadata

    def __len__(self):
        return self.nentries


class LmdbDataset(Dataset):
    def __init__(self, data_dir, split, transforms, alphabet=None, image_verbose=False,
                 image_height=30, min_image_width=1, max_image_width=1000):

        self.image_verbose = image_verbose
        self.data_dir = data_dir
        self.split = split
        self.image_height = image_height
        self.max_image_width = max_image_width
        self.min_image_width = min_image_width
        self.preprocess = transforms

        # Read Dataset Description
        with open(os.path.join(data_dir, 'desc.json'), 'r') as fh:
            self.data_desc = json.load(fh)

        # Init alphabet
        alphabet_expand_cnt = 0
        if alphabet is not None:
            LOG.info(
                "Explicitly providing alphabet via initializatin parameter, as opposed to inferring from data")
            self.alphabet = alphabet

            for split in ['train', 'validation', 'test']:
                for entry in self.data_desc[split]:
                    for char in entry['trans'].split():
                        if char not in self.alphabet.char_to_idx:
                            self.alphabet.max_id += 1
                            self.alphabet.char_to_idx[char] = self.alphabet.max_id
                            self.alphabet.idx_to_char[self.alphabet.max_id] = char
                            #print('...expanding Alphabet, adding id %d, char %s' % (
                            #    self.alphabet.max_id, char))
                            alphabet_expand_cnt += 1

        else:
            self.init_alphabet()

        if alphabet_expand_cnt > 0:
            print('...expanded alphabet, added ', alphabet_expand_cnt)
        print('Alphabet size ', len(self.alphabet))

        # for alphabet_idx in range(0, len(self.alphabet)):
        #     if alphabet_idx < 150:
        #         print(alphabet_idx, self.alphabet.idx_to_char[alphabet_idx])
        #     elif alphabet_idx > len(self.alphabet) - 25:
        #         print(alphabet_idx, self.alphabet.idx_to_char[alphabet_idx])

        # Read LMDB image database
        self.lmdb_env = lmdb.Environment(os.path.join(
            data_dir, 'line-images.lmdb'), map_size=1e6, readonly=True, lock=False)
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        self.size_group_limits = [150, 200, 300,
                                  350, 450, 600, 900, 1200, np.inf]

        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        self.writer_id_map = dict()

        for idx, entry in enumerate(self.data_desc[self.split]):
            # First handle writer id
            if 'writer' in entry:
                if not entry['writer'] in self.writer_id_map:
                    self.writer_id_map[entry['writer']] = len(
                        self.writer_id_map)

            image_id = entry['id']
            image_uxxxx_trans = entry['trans']
            image_width_orig, image_height_orig = entry['width'], entry['height']
            # if image_width_orig > 2:
            if image_height_orig > 10 and image_width_orig > 20 and len(image_uxxxx_trans.split()) > 2:
                # Now figure out which size-group it belongs in
                for cur_limit in self.size_group_limits:
                    if ('height' in entry) and ('width' in entry):
                        width_orig, height_orig = entry['width'], entry['height']
                        normalized_height = 30
                        normalized_width = width_orig * \
                            (normalized_height / height_orig)
                    elif 'width' in entry:
                        normalized_width = entry['width']
                    else:
                        raise Exception(
                            "Json entry must list width & height of image.")

                    if normalized_width < cur_limit and normalized_width < self.max_image_width:
                        self.size_groups[cur_limit].append(idx)
                        self.size_groups_dict[cur_limit][idx] = 1
                        break

        # Now get final size (might have dropped large entries!)
        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])

            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max

        print("...finished loading {}, size {}".format(split, self.nentries))

    def init_alphabet(self):
        # Read entire train/val/test data to deterimine set of unique characters we should have in alphabet
        unique_chars = set()

        for split in ['train', 'validation', 'test']:
            for entry in self.data_desc[split]:
                for char in entry['trans'].split():
                    unique_chars.add(char)

        # Now add CTC blank as first letter in alphabet. Also sort alphabet lexigraphically for convinience
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

    def determine_width_cutoffs(self):

        # The purpsoe of this function is to break the data into groups of similar widths
        # for example:  all images of width between 0 and 100; all images of widths between 101 and 200; etc
        #
        # We do this for performance reasons as mentioned above; we set the group cutoffs based on the dataset

        # First let's cycle through data and get count of all widths:
        max_normalized_width = 800
        normalized_height = 30
        normalized_width_counts = np.zeros((max_normalized_width))

        remove_count = 0
        for idx, entry in enumerate(self.data_desc[self.split]):
            w, h = entry['width'], entry['height']
            normalized_width = (normalized_height/h) * w

            if normalized_width > max_normalized_width:
                remove_count += 1
            else:
                normalized_width_counts[normalized_width-1] += 1

        LOG.info("Removed %d images due to max width cutoff" % remove_count)

        # Now use data to determine cutoff points
        normalized_width_cumsum = np.cumsum(normalized_width_counts)

        return None

    def __getitem__(self, idx):
        entry = self.data_desc[self.split][idx]

        max_width = 0
        for cur_limit in self.size_group_limits:
            if idx in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break
        group_id = max_width

        img_bytes = np.asarray(self.lmdb_txn.get(
            entry['id'].encode('ascii')), dtype=np.uint8)
        line_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)  # -1)

        # Do a check for RGBA images; if found get rid of alpha channel
        if len(line_image.shape) == 3 and line_image.shape[2] == 4:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGRA2BGR)

        if line_image.shape[0] == 0 or line_image.shape[1] == 0:
            print("ERROR, line image is 0 area; id = %s; idx = %d" %
                  (entry['id'], idx))

        line_image = self.resize_or_pad(line_image)

        line_image = self.preprocess(line_image)

        original_width = line_image.size(2)
        original_height = line_image.size(1)

        transcription = []
        for char in entry['trans'].split():
            transcription.append(self.alphabet.char_to_idx[char])

        seed_text = uxxxx_to_utf8(entry['trans'])

        metadata = {
            'idx': idx,
            'transcription': transcription,
            'seed_text': seed_text,
            'width': original_width,
            'height': original_height,
            'group': group_id,
            'image_id':  entry['id'],
        }

        if self.image_verbose:
            print('\nGETITEM: group', metadata['group'])
            print('GETITEM: image shape', line_image.shape)
            print('GETITEM: target_ids', transcription)
            print('GETITEM: seed text', seed_text)
            print('GETITEM: target_length', len(transcription))

        return line_image, transcription, metadata

    def __len__(self):
        return self.nentries

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
            dim = (int(w * r), height)
        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)

        (h, w) = resized.shape[:2]
        return resized

    def resize_or_pad(self, img_data):
        img_height, img_width = img_data.shape[:2]

        if img_height > self.image_height:
            img_data = self.image_resize(
                img_data, height=self.image_height)

        img_height, img_width = img_data.shape[:2]

        if img_width > self.max_image_width:
            img_data = self.image_resize(
                img_data, width=self.max_image_width)

        img_height, img_width = img_data.shape[:2]

        if img_height < self.image_height:
            img_data = self.image_pad(
                img_data, input_pad_height=self.image_height)

        img_height, img_width = img_data.shape[:2]

        if img_width < self.min_image_width:
            img_data = self.image_pad(
                img_data, input_pad_width=self.min_image_width)

        return img_data

    def image_pad(self, img_data, input_pad_height=None, input_pad_width=None):
        img_height, img_width = img_data.shape[:2]

        if input_pad_height is None:
            pad_height = img_height
        else:
            pad_height = input_pad_height - img_height

        if input_pad_width is None:
            pad_width = img_width
        else:
            pad_width = input_pad_width - img_width

        top, bottom = pad_height//2, pad_height-(pad_height//2)
        border_color = [255, 255, 255]
        try:
            img_data_pad = cv2.copyMakeBorder(img_data,
                                              top, bottom, 0, 0, cv2.BORDER_CONSTANT,
                                              value=border_color)
        except Exception as e:
            print('EXCEPT image_pad', img_height, img_width)
            img_data_pad = img_data
            # img_height, img_width = img_data_pad.shape[:2]
        # print('->image_pad', img_height, img_width)
        return img_data_pad


class LmdbDatasetUnion(Dataset):
    def __init__(self, data_dir_list, split, transforms, alphabet=None, image_height=30, max_image_width=1500):

        self.datasets = []
        self.nentries = 0
        for data_dir in data_dir_list:
            print('...load %s' % (data_dir))
            dataset = OcrDataset(data_dir, split, transforms,
                                 alphabet, image_height=image_height, max_image_width=max_image_width)
            self.datasets.append(dataset)
            self.nentries += len(dataset)

        # Because different datasets might have different alphabets, we need to merge them and unify
        self.merge_alphabets()

        # Merge size group stuff (ugh)
        self.size_group_keys = self.datasets[0].size_group_keys
        self.size_groups = dict()
        for cur_limit in self.size_group_keys:
            self.size_groups[cur_limit] = []

        accumulatd_max_idx = 0
        for ds in self.datasets:
            # For now we only merge if szme set of size groups  (need to change this requirement!)
            assert ds.size_group_keys == self.size_group_keys
            for cur_limit in self.size_group_keys:
                self.size_groups[cur_limit].extend(
                    [accumulatd_max_idx + idx for idx in ds.size_groups[cur_limit]])
            accumulatd_max_idx += ds.max_index

    def merge_alphabets(self):
        alphabet_list = [ds.alphabet for ds in self.datasets]
        unique_chars = set()
        for alphabet in alphabet_list:
            # First entry is always <ctc-blank>, so let's just grab all the entries after that
            unique_chars.update(alphabet.char_array[1:])

        # Now create a new alphabet w/ merged characters
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

        # And propogate it to each of our datasets
        for ds in self.datasets:
            ds.alphabet = self.alphabet

    def __getitem__(self, index):
        accumulatd_max_idx = 0
        for dataset in self.datasets:
            if index <= accumulatd_max_idx + dataset.max_index:
                return dataset[index - accumulatd_max_idx]
            accumulatd_max_idx += dataset.max_index

        print("index = %d" % index)
        print("total num entries = %d" % self.nentries)
        print("Size of each dataset = ")
        for ds in self.datasets:
            print("Size = %d" % len(ds))
        assert False, "Should never get here"

    def __len__(self):
        return self.nentries
