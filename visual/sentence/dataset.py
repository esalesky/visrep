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

import gc
import sys
import inspect
from typing import Optional, Iterable, Any
from text_utils import utf8_to_uxxxx, uxxxx_to_utf8

LOG = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)


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


class GroupedSampler(Sampler):
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
                    raise StopIteration
                yield self.size_groups[g][g_idx]

        raise StopIteration

    def __len__(self):
        return self.num_samples


def SortByWidthCollater(batch, image_verbose):
    _use_shared_memory = True

    #image_verbose = True

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

    target_transcription = torch.IntTensor(
        target_transcription_widths.sum().item())
    cur_offset = 0
    for idx, (tensor, transcript, metadata) in enumerate(batch):
        for j, char in enumerate(transcript):
            target_transcription[cur_offset] = char
            cur_offset += 1

    metadata = {}
    metadata['transcription'] = trans_ids

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


class ImageDataset(Dataset):

    def __init__(self,
                 text_file_path, font_file_path,
                 train_cache_output=None,
                 surf_width=3000, surf_height=250,
                 start_x=25, start_y=25, dpi=120,
                 image_height=30,
                 max_cache_write=5000,
                 max_seed=500000,
                 transform=None,
                 alphabet=None,
                 save_images=False,
                 image_verbose=False,
                 use_image_cache=False,
                 max_image_cache=500000,
                 max_allowed_width=3000):

        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.use_image_cache = use_image_cache
        self.max_seed = max_seed
        self.max_cache_write = max_cache_write
        self.save_images = save_images
        self.max_image_cache = max_image_cache
        self.train_cache_output = train_cache_output
        self.image_height = image_height
        self.image_verbose = image_verbose
        self.max_allowed_width = max_allowed_width
        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.dpi = dpi

        self.font_rotation = [0]  # [-6, -4, -2, 0, 2, 4, 6]
        self.pad_top = [0, 1, 2]  # [0, 2, 4, 6, 8]
        self.pad_bottom = [0, 1, 2]  # [0, 2, 4, 6, 8]
        self.pad_left = [0, 1, 2]  # [0, 2, 4, 6, 8]
        self.pad_right = [0, 1, 2]  # [0, 2, 4, 6, 8]
        self.font_size = [10]  # [10, 14, 18, 24, 32]
        self.font_color = ['black']
        self.bkg_color = ['white']

        self.transform = transform

        self.font_list = self.get_font_list(font_file_path)
        self.text_list = self.get_text_list(text_file_path)
        self.text_list.sort()

        # Init alphabet
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
                        print('...expanding Alphabet, adding id %d, char %s' % (
                            self.alphabet.max_id, char))

        else:
            self.init_alphabet()

        # this is text size
        self.size_group_limits = [1, 5, 10, 15,
                                  20, 25, 30, 35, 40, 45, 50, np.inf]

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
                if text_width < cur_limit and text_width < self.max_allowed_width:
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
                orig_line = orig_line.strip()
                line = orig_line.split()
                text_list.append(line[0])
                if line_cnt > self.max_seed:
                    print('...max seed text input reached', self.max_seed)
                    break

        len_list = [len(x) for x in text_list]

        LOG.info('Total from {}, read {}, included {}, max {}, min {}, mean {}, median {}'.format(
            input_text, line_cnt, len(text_list), max(len_list), min(
                len_list), int(mean(len_list)), median(len_list)
        ))
        return text_list

    def get_font_list(self, font_file_path):
        fontlist = []
        fontcnt = 0
        LOG.info('...loading fonts from %s', font_file_path)
        with open(font_file_path, 'r') as file:  # , encoding='utf8') as file:
            for ctr, line in enumerate(file.readlines()):
                fontname = line.strip()
                fontcnt += 1
                fontlist.append(fontname)
        LOG.info('Found %d fonts', len(fontlist))
        return fontlist

    def init_alphabet(self):
        # Read entire train/val/test data to deterimine set of unique characters we should have in alphabet
        unique_chars = set()

        for idx, text_line in enumerate(self.text_list):
            uxxxx_text_list = utf8_to_uxxxx(text_line, output_array=True)
            for char in uxxxx_text_list:
                unique_chars.add(char)

        # Now add CTC blank as first letter in alphabet. Also sort alphabet lexigraphically for convinience
        self.alphabet = Alphabet(['<ctc-blank>', *sorted(unique_chars)])

    def image_resize(self, image, width=None, height=None,
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

    def get_image(self, line_text,
                  font_name=None, font_size=None, font_style=None,
                  font_color=None, bkg_color=None, font_rotate=None,
                  pad_top=None, pad_bottom=None, pad_left=None, pad_right=None):
        ''' Create pygame surface '''

        surf = pygame.Surface((self.surface_width, self.surface_height))

        # Replace Unicode Character 'LOWER ONE EIGHTH BLOCK' (U+2581)
        # many of the fonts can not render this code
        line_text = line_text.replace('▁', '_')

        if font_name:
            font_name = font_name
        else:
            font_name = random.choice(self.font_list)

        if font_size:
            font = pygame.freetype.Font(font_name, font_size)
        else:
            font = pygame.freetype.Font(
                font_name, random.choice(self.font_size))

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

        if pad_right:
            pad_right = pad_right
        else:
            pad_right = random.choice(self.pad_right)

        crop = (self.start_x - pad_left, self.start_y - pad_top,
                text_rect.width + (pad_left + pad_right),
                text_rect.height + (pad_top + pad_bottom))

        sub_surf = surf.subsurface(crop)

        img_data = pygame.surfarray.array3d(sub_surf)
        img_data = img_data.swapaxes(0, 1)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

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
            cv_resize_image = self.image_resize(
                cv_image, height=self.image_height)
            (h, w) = cv_resize_image.shape[:2]
            widths.append(w)
            self.image_cache[idx] = cv_resize_image

            if self.train_cache_output and self.save_images:
                #print('write image', self.train_cache_output, write_cnt)
                if self.max_cache_write < write_cnt:
                    outpath = self.train_cache_output + '/' + \
                        str(idx) + '_h' + str(h) + '_w' + \
                        str(w) + '_' + seed_text + '.png'
                    #print('....save image', outpath)
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

    def __getitem__(self, idx):
        seed_text = self.text_list[idx]

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
                #print('...cache miss, create image ', idx, seed_text)
                cv_image = self.get_image(seed_text,
                                          font_name=self.font_list[0], font_size=8, font_style=1,
                                          font_color='black', bkg_color='white', font_rotate=0,
                                          pad_top=2, pad_bottom=2, pad_left=2, pad_right=2)

                cv_resize_image = self.image_resize(
                    cv_image, height=self.image_height)
        else:
            # cv_image = self.get_image(seed_text,
            #                          font_name=self.font_list[0], font_size=8, font_style=1,
            #                          font_color='black', bkg_color='white', font_rotate=0,
            #                          pad_top=2, pad_bottom=2, pad_left=2, pad_right=2)

            cv_image = self.get_image(seed_text,
                                      font_size=8,
                                      font_color='black', bkg_color='white')

            cv_resize_image = self.image_resize(
                cv_image, height=self.image_height)

        line_image = self.transform(cv_resize_image)

        original_width = line_image.size(2)
        original_height = line_image.size(1)

        if max_width < self.size_group_limits[-1]:
            torch.backends.cudnn.benchmark = True
            line_image_padded = torch.zeros(
                line_image.size(0), line_image.size(1), max_width)
            line_image_padded = line_image[:, :, :line_image.size(2)]
        else:
            torch.backends.cudnn.benchmark = False
            line_image_padded = line_image

        transcription = []
        uxxxx_text_list = utf8_to_uxxxx(seed_text, output_array=True)
        for char in uxxxx_text_list:
            transcription.append(self.alphabet.char_to_idx[char])

        metadata = {
            'transcription': transcription,
            'seed_text': seed_text,
            'width': original_width,
            'height': original_height,
            'group': group_id,
        }

        if self.image_verbose:
            print('\nGETITEM: group', metadata['group'])
            print('GETITEM: image shape', line_image.shape)
            print('GETITEM: padded shape', line_image_padded.shape)
            print('GETITEM: target_ids', transcription)
            print('GETITEM: seed text', seed_text)
            print('GETITEM: target_length', len(transcription))

        return line_image_padded, transcription, metadata

    def __len__(self):
        # return len(self.text_list)
        return self.nentries
