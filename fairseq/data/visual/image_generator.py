#!/usr/bin/env python3

import cv2
import logging
import pygame.freetype
import os
import sys
import random


import numpy as np
import torch

import torchvision.transforms as transforms

logger = logging.getLogger(__name__)

class TextImageGenerator():
    def __init__(self,
                 font_file=None,
                 surf_width=5000, surf_height=200,
                 start_x=25, start_y=25, dpi=120,
                 bkg_color="white",
                 font_color="black",
                 font_style=1,
                 font_size=8,
                 font_rotation=0,
                 pad_size=5,
                 window=30,
                 stride=20,
             ):
        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.font_file = font_file
        logger.info(f"Font path: {self.font_file}")

        self.surface_width = surf_width
        self.surface_height = surf_height
        self.start_x = start_x
        self.start_y = start_y
        self.dpi = dpi

        self.font_rotation = [font_rotation] if font_rotation is not None else [-6, -4, -2, 0, 2, 4, 6]
        self.pad_top = pad_size
        self.pad_bottom = pad_size
        self.pad_left = pad_size
        self.pad_right = pad_size

        self.font_size = font_size
        self.font_style = font_style
        self.font_color = "black"
        self.bkg_color = "white"

        self.font = pygame.freetype.Font(self.font_file, self.font_size)
        self.font.style = pygame.freetype.STYLE_NORMAL
        self.font.fgcolor = pygame.color.THECOLORS[font_color]

        # Get the maximum image height
        self.image_height = self.font.get_rect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.").height + self.pad_top + self.pad_bottom
        logger.info(f"Image height for font size {self.font_size} is {self.image_height}")

        self.window = window
        self.overlap = window - stride

    def get_surface(self, line_text):
        """Creates a single image from an entire line and returns the surface."""

        # Replace Unicode Character 'LOWER ONE EIGHTH BLOCK' (U+2581)
        # many of the fonts can not render this code
        # TODO
        line_text = line_text.replace('▁', '_')

        curr_surface_width = self.surface_width
        curr_surface_height = self.surface_height
        text_rect_size = self.font.get_rect(line_text)

        surf = pygame.Surface((curr_surface_width, curr_surface_height))
        surf.fill(pygame.color.THECOLORS['white'])

        text_rect = self.font.render_to(
            surf, (self.start_x, self.start_y), line_text)

        # Make sure crop width is a multiple of the stride
        crop_width = text_rect.width + (self.pad_left + self.pad_right)
        if crop_width % self.window != 0:
            crop_width = ((crop_width // self.window) + 1) * self.window

        crop = (self.start_x - self.pad_left,
                self.start_y - self.pad_top - self.pad_bottom,
                crop_width,
                self.image_height)

        surf = surf.subsurface(crop)

        return surf

    def get_image_from_surface(self, surf):
        """Transforms a surface containing a rendered image into a numpy image."""
        image = pygame.surfarray.pixels3d(surf)
        image = image.swapaxes(0, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def get_image(self, text):
        """
        Returns a single image from a line of text.
        """
        return self.get_image_from_surface(self.get_surface(text))

    def get_images(self, line_text):
        """
        Returns images from all pieces in a line of text.
        """

        surface = self.get_surface(line_text)
        (width, height) = surface.get_size()

        whole_image = self.get_image_from_surface(surface)

        # Move a window over the image. The image width is guaranteed to be at
        # least as wide as the window.
        image_pieces = []
        for x in range(0, width - self.window + 1, self.window - self.overlap):
            crop_width = self.window
            # if x + crop_width > width:
            #     crop_width -= (x + crop_width - width)
            crop_region = (x, 0, crop_width, height)

            image = self.get_image_from_surface(surface.subsurface(crop_region))
            image_pieces.append(image)
            # tensors.append(self.transform(image))

        return whole_image, image_pieces

    def get_tensor(self, text):
        """Returns a single representing images for all pieces in a sentence.
        Dimension (num_pieces x channels=1 x height x width)
        """
        whole_image, image_pieces = self.get_images(text)
        tensors = []
        for image in image_pieces:
            image_tensor = self.transform(image)
            tensors.append(image_tensor)

        assert len(tensors) != 0, text
        return torch.stack(tensors)

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


def main(args):
    gen = TextImageGenerator(window=args.image_window,
                             stride=args.image_stride,
                             font_size=args.font_size,
    )
    whole_image = gen.get_image(args.text)
    imagepath = f"test_image.png"
    print(f"Writing to {imagepath}", file=sys.stderr)
    cv2.imwrite(imagepath, whole_image)

    whole_image, image_pieces = gen.get_images(args.text)
    for i, image in enumerate(image_pieces, 1):
        imagepath = f"test_image.{i}.png"
        cv2.imwrite(imagepath, image)
        print(f"Writing to {imagepath}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-file", type=str, default="/home/hltcoe/mpost/code/fairseq-ocr/fairseq/data/visual/fonts/NotoMono-Regular.ttf")
    parser.add_argument("--font-size", type=int, default=8)
    parser.add_argument("--image-window", type=int, default=30)
    parser.add_argument("--image-stride", type=int, default=10)
    parser.add_argument("--text", type=str, default="This is a test.")
    args = parser.parse_args()

    main(args)
