from torch.utils.data import Dataset
import cv2
import pygame.freetype
import random
import logging


LOG = logging.getLogger(__name__)


class ImageDataset(Dataset):

    def __init__(self,
                 text_file_path, font_file_path,
                 surf_width=1500, surf_height=250,
                 start_x=50, start_y=50, dpi=120,
                 image_height=128, image_width=32,
                 use_cache=False, mod_cache=2, default_image=False,
                 transform=None, label_dict=None, rev_label_dict=None):

        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.dpi = dpi

        self.font_rotation = [-6, -4, -2, 0, 2, 4, 6]
        self.pad_top = [0, 2, 4, 6, 8]
        self.pad_bottom = [0, 2, 4, 6, 8]
        self.pad_left = [0, 2, 4, 6, 8]
        self.pad_right = [0, 2, 4, 6, 8]
        self.font_size = [10, 14, 18, 24, 32]
        self.font_color = ['black']
        self.bkg_color = ['white']

        self.image_height = image_height
        self.image_width = image_width
        self.transform = transform

        self.font_list = self.get_font_list(font_file_path)
        self.text_list = self.get_text_list(text_file_path)
        self.label_dict = label_dict
        self.rev_label_dict = rev_label_dict
        if not self.label_dict:
            self.label_dict, self.rev_label_dict = self.build_dictionary()

        self.counter = 0
        self.default_image = default_image
        # self.image_cache = {}
        # self.use_cache = use_cache
        # self.mod_cache = mod_cache
        # if self.use_cache:
        #     self.load_image_cache()

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

        LOG.info('Total from {}, read {}, included {}'.format(
            input_text, line_cnt, len(text_list)))
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

    def build_dictionary(self):
        label_dict = {}
        rev_label_dict = {}
        dict_id = 0
        for text_item in self.text_list:
            if text_item not in label_dict:
                label_dict[text_item] = dict_id
                rev_label_dict[dict_id] = text_item
                dict_id += 1
        LOG.info('max label %d', dict_id)
        return label_dict, rev_label_dict

    def __len__(self):
        return len(self.text_list)

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

    def resize_or_pad(self, img_data, img_width, img_height):
        img_height, img_width = img_data.shape[:2]
        if img_height > self.image_height:
            img_data = self.image_resize(img_data, height=self.image_height)
            img_height, img_width = img_data.shape[:2]

        if img_width > self.image_width:
            img_data = self.image_resize(img_data, width=self.image_width)
            img_height, img_width = img_data.shape[:2]

        img_height, img_width = img_data.shape[:2]
        pad_height = self.image_height - img_height
        pad_width = self.image_width - img_width

        border_color = [255, 255, 255]

        img_data_pad = cv2.copyMakeBorder(
            img_data, pad_height, 0, 0, pad_width, cv2.BORDER_CONSTANT,
            value=border_color)

        return img_data_pad

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
        for idx, seed_text in enumerate(self.text_list):
            cv_image = self.get_image(seed_text,
                                      font_name=self.font_list[0], font_size=16, font_style=1,
                                      font_color='black', bkg_color='white', font_rotate=None,
                                      pad_top=5, pad_bottom=5, pad_left=5, pad_right=5)
            cv_resize_image = self.resize_or_pad(
                cv_image, self.image_width, self.image_height)
            self.image_cache[idx] = cv_resize_image

    def __getitem__(self, idx):
        self.counter += 1

        seed_text = self.text_list[idx]
        seed_id = self.label_dict[seed_text]

        # if self.use_cache:
        #    if self.counter % self.mod_cache:
        #        cv_resize_image = self.image_cache[idx]
        # else:
        if self.default_image:
            cv_image = self.get_image(seed_text,
                                      font_name=self.font_list[0], font_size=16, font_style=1,
                                      font_color='black', bkg_color='white', font_rotate=0,
                                      pad_top=5, pad_bottom=5, pad_left=5, pad_right=5)
        else:
            cv_image = self.get_image(seed_text,
                                      font_color='black', bkg_color='white')

        # LOG.info(cv_image.shape)  # (32, 128, 3) H, W, C
        cv_resize_image = self.resize_or_pad(
            cv_image, self.image_width, self.image_height)
        # LOG.info(cv_resize_image.shape) # (32, 128, 3) H, W, C

        img_tensor = self.transform(cv_resize_image)
        # Resnet expects shape (3 x H x W)
        # LOG.info(img_tensor.shape) # torch.Size([3, 32, 128])
        return img_tensor, seed_id, seed_text
