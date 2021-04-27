#!/usr/bin/env python3

import cv2
import logging
import os
import sys
import random

import numpy as np
import torch

import torchvision.transforms as transforms

# This gets rid of the rude message printed to STDOUT
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame.freetype

from fairseq.options import DEFAULT_FONT_SIZE, DEFAULT_PAD_SIZE, DEFAULT_WINDOW, DEFAULT_STRIDE, MAX_SURFACE_WIDTH


logger = logging.getLogger(__name__)

class TextImageGenerator():
    def __init__(self,
                 font_file=None,
                 surface_width=MAX_SURFACE_WIDTH,
                 dpi=120,
                 bkg_color="white",
                 font_color="black",
                 font_size=DEFAULT_FONT_SIZE,
                 pad_size=DEFAULT_PAD_SIZE,
                 window=DEFAULT_WINDOW,
                 stride=DEFAULT_STRIDE,
             ):
        pygame.freetype.init()
        pygame.freetype.set_default_resolution(dpi)

        # This normalizes the 0--255 grayscale values
        # and creates the tensors.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.fonts = self.load_fonts(font_file, font_size)

        if surface_width > MAX_SURFACE_WIDTH:
            logger.warning(f"Reducing surface width from {surface_width} to {MAX_SURFACE_WIDTH} or {16384//stride} slices")
            surface_width = MAX_SURFACE_WIDTH

        self.surface_width = surface_width
        self.dpi = dpi

        self.pad_top = pad_size
        self.pad_bottom = pad_size
        self.pad_left = pad_size
        self.pad_right = pad_size

        self.font_size = font_size
        self.font_color = "black"
        self.bkg_color = "white"

        # Get the maximum image height
        self.image_height = 0
        for font in self.fonts.values():
            self.image_height = max(
                self.image_height,
                font.get_rect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.").height + self.pad_top + self.pad_bottom
            )

        # Get the estimated upper bound on character width
        self.estimated_max_char_width = 0
        for font in self.fonts.values():
            self.estimated_max_char_width = max(
                self.estimated_max_char_width,
                font.get_rect("W.").width + self.pad_left + self.pad_right
            )

        font_name = os.path.basename(font_file)
        logger.info(f"Created {self.font_size}pt {font_name} with image height {self.image_height} and est. char width {self.estimated_max_char_width}")

        self.window = window
        self.stride = stride
        self.overlap = window - stride

        logger.info(f"Image window size {self.window} stride {self.stride}")

    @property
    def height(self):
        return self.image_height

    @property
    def width(self):
        return self.window

    def get_surface(self, line_text, lang="*", remove_subword=True):
        """Creates a single image from an entire line and returns the surface."""

        if remove_subword:
            line_text = line_text.replace(" ", "").replace("▁", " ").strip()

        # We can save a lot of time by creating a smaller surface, so here,
        # we estimate the width based on how much 
        surface_width = self.estimated_max_char_width * len(line_text)
        # print(f"SURFACE: {surface_width} x {self.image_height * 2}")
        surf = pygame.Surface((surface_width, self.image_height * 2))
        surf.fill(pygame.color.THECOLORS['white'])

        text_rect = self.fonts[lang].render_to(
            surf, (self.pad_left, self.pad_top), line_text)

        # Make sure the stride + window fit within the surface
        crop_width = max(self.window, text_rect.width + (self.pad_left + self.pad_right))
        if (crop_width - self.window) % self.window != 0:
            # The width minus the stride has to factorize by the window size.
            # If not, find the size of the last piece, and increase it to be
            # exactly one window size.
            crop_width += self.window - ((crop_width - self.stride) % self.window)

        if crop_width > self.surface_width:
            old_width = crop_width
            # smallest number <= self.surface_width that self.stride factorizes into
            while crop_width > self.surface_width:
                crop_width -= self.stride
            logger.warning(f"{len(line_text.split())} raw text tokens ({old_width // self.window} slices) too wide, truncating {old_width} -> {crop_width}")

        crop = (0, 0, crop_width, self.image_height)
        surf = surf.subsurface(crop)

        return surf

    def get_image_from_surface(self, surf, lang="*"):
        """Transforms a surface containing a rendered image into a numpy image."""
        image = pygame.surfarray.pixels3d(surf)
        image = image.swapaxes(0, 1)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return image

    def get_image(self, text, lang="*"):
        """
        Returns a single image from a line of text.
        Dimensions: height x width
        """
        return self.get_image_from_surface(self.get_surface(text))

    def get_images(self, line_text, lang="*"):
        """
        Returns images from all pieces in a line of text.
        Shape: slices x height x width
        """

        surface = self.get_surface(line_text)
        (width, height) = surface.get_size()

        whole_image = self.get_image_from_surface(surface)

        # Move a window over the image. The image width is guaranteed to be at
        # least as wide as the window.
        image_pieces = []
        for x in range(0, width - self.window + 1, self.stride):
            crop_width = self.window
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

    @classmethod
    def load_fonts(cls, font_file_path, font_size, font_color="black"):
        """The font file path is either (a) the path to a font or (b) the path
        to a file containing (language pair, font path) pairs, e.g.,

            * NotoSans-Regular.ttf
            ar NotoNaskhArabic-Regular.ttf

        The '*' keyword functions as a default, when the language is
        not specified or provided.
        """
        fonts = {}
        logger.info(f"Loading fonts from {font_file_path}")
        if font_file_path.endswith(".map"):
            with open(font_file_path, 'r') as infile:
                for line in infile:
                    langpair, path = line.rstrip().split()
                    fonts[langpair] = pygame.freetype.Font(path, font_size)
                    logger.info(f"-> Found {langpair} font {path}")
        else:
            fonts["*"] = pygame.freetype.Font(font_file_path, font_size)

        for langpair, font in fonts.items():
            font.style = pygame.freetype.STYLE_NORMAL
            font.fgcolor = pygame.color.THECOLORS[font_color]

        if "*" not in fonts:
            logger.error(f"Found no default font!")
            sys.exit(1)

        return fonts


def main(args):
    gen = TextImageGenerator(window=args.window,
                             stride=args.stride,
                             font_size=args.font_size,
                             font_file=args.font_file,
    )
    whole_image = gen.get_image(args.text)
    imagepath = f"{args.prefix}.png"
    print(f"Writing to {imagepath}", file=sys.stderr)
    cv2.imwrite(imagepath, whole_image)

    whole_image, image_pieces = gen.get_images(args.text)
    for i, image in enumerate(image_pieces, 1):
        imagepath = f"{args.prefix}.{i:02d}.png"
        cv2.imwrite(imagepath, image)
        print(f"Writing to {imagepath}", file=sys.stderr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-file", type=str, default="/home/hltcoe/mpost/code/fairseq-ocr/fairseq/data/visual/fonts/NotoMono-Regular.ttf")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--prefix", type=str, default="test_image")
    parser.add_argument("--text", type=str, default="The quick brown fox jumped over the lazy dog.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
