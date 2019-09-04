import numpy as np
# from PIL import Image, ImageDraw, ImageFont
import pygame.freetype
from collections import Counter
import os
import random
import logging
import cv2

logger = logging.getLogger('root')

'''

Render an image from a word or line of text

'''


class ImageGenerator():

    def __init__(self, font_file_path, font_size=16,
                 font_color='white', bkg_color='black',
                 image_width=150, image_height=30,
                 surf_width=1500, surf_height=100,
                 start_x=10, start_y=10,
                 pad_top=2, pad_bottom=2, pad_left=4, pad_right=2,
                 dpi=120, image_rand_font=False, image_rand_style=False):

        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_y
        self.start_y = start_y
        self.pad_top = pad_top
        self.pad_bottom = pad_bottom
        self.pad_left = pad_left
        self.pad_right = pad_right
        self.font_size = font_size
        self.font_list = self.get_font_list(font_file_path)
        self.dpi = dpi
        self.image_font_color = font_color
        self.image_bkg_color = bkg_color
        self.image_height = image_height
        self.image_width = image_width
        self.image_rand_font = image_rand_font
        self.image_rand_style = image_rand_style

        # self.max_height = 0
        # self.max_width = 0

        pygame.freetype.init()
        pygame.freetype.set_default_resolution(self.dpi)

    def get_font_list(self, font_file_path):
        fontlist = []
        fontcnt = 0
        logger.info('...loading fonts from %s' % font_file_path)
        with open(font_file_path, 'r') as file:  # , encoding='utf8') as file:
            for ctr, line in enumerate(file.readlines()):
                fontname = line.strip()
                fontcnt += 1
                fontlist.append(fontname)
        logger.info('Found %d fonts' % (len(fontlist)))
        return fontlist

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
        # print('...resized ', h, w)
        # return the resized image
        return resized

    def get_default_image(self, line_text):
        ''' Create pygame surface '''

        surf = pygame.Surface((self.surface_width, self.surface_height))

        ''' Get font '''
        if self.image_rand_font:
            font_name = self.font_list[0]
        else:
            font_name = random.choice(self.font_list)

        font = pygame.freetype.Font(font_name, self.font_size)

        if self.image_rand_style:
            font_style = random.randint(1, 6)
        else:
            font_style = 3

        if font_style == 1:
            font.style = pygame.freetype.STYLE_NORMAL
        elif font_style == 2:
            font.style = pygame.freetype.STYLE_OBLIQUE
        elif font_style == 3:
            font.style = pygame.freetype.STYLE_STRONG
        else:
            font.style = pygame.freetype.STYLE_DEFAULT

        ''' Get style '''
        # font.style = pygame.freetype.STYLE_NORMAL
        font.style = pygame.freetype.STYLE_STRONG

        ''' Set colors '''
        font.fgcolor = pygame.color.THECOLORS[self.image_font_color]
        surf.fill(pygame.color.THECOLORS[self.image_bkg_color])

        ''' Render to surface '''
        finaltxtRect = font.render_to(
            surf, (self.start_x, self.start_y), line_text)

        ''' Crop text from pygram surface '''
        crop = (self.start_x - self.pad_left, self.start_y - self.pad_top,
                finaltxtRect.width + (self.pad_left + self.pad_right),
                finaltxtRect.height + (self.pad_top + self.pad_bottom))
        sub_surf = surf.subsurface(crop)
        # Creates a 3D array (RGB pixel values) that is copied from any type of
        # surface
        img_data = pygame.surfarray.array3d(sub_surf)
        # convert from (width, height, channel) to (height, width, channel)
        img_data = img_data.swapaxes(0, 1)
        img_height, img_width = img_data.shape[:2]

        # print(img_data.shape[:2])

        ''' Resize or pad image '''
        if img_width > self.image_width:
            img_data = self.image_resize(img_data, width=self.image_width)
        img_height, img_width = img_data.shape[:2]

        if img_height > self.image_height:
            img_data = self.image_resize(img_data, height=self.image_height)
        img_height, img_width = img_data.shape[:2]

        assert (img_height <= self.image_height), \
            ("Current height %d, must be less than %d" % (
            img_height, self.image_height))
        assert (img_width <= self.image_width), \
            ("Current width %d, must be less than %d" % (
            img_width, self.image_width))

        img_height, img_width = img_data.shape[:2]
        pad_height = self.image_height - img_height
        pad_width = self.image_width - img_width

        if self.image_bkg_color == 'white':
            border_color = [255, 255, 255]
        else:
            border_color = [0, 0, 0]

        img_data_pad = cv2.copyMakeBorder(
            img_data, pad_height, 0, 0, pad_width, cv2.BORDER_CONSTANT,
            value=border_color)

        img_pad_height, img_pad_width = img_data_pad.shape[:2]

        return img_data_pad, img_pad_width
