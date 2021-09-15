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
from arabic_reshaper import ArabicReshaper
from bidi.algorithm import get_display

DEFAULT_FONT_SIZE = 8
DEFAULT_PAD_SIZE = 3
DEFAULT_WINDOW = 30
DEFAULT_STRIDE = 20
MAX_SURFACE_WIDTH = 16383

logger = logging.getLogger(__name__)

class TextImageGenerator():
    def __init__(self,
                 font_file=None,
                 dpi=120,
                 bkg_color="white",
                 font_color="black",
                 font_size=DEFAULT_FONT_SIZE,
                 pad_size=DEFAULT_PAD_SIZE,
                 window=DEFAULT_WINDOW,
                 stride=DEFAULT_STRIDE,
             ):
        pygame.freetype.init()
#        pygame.display.init()
        pygame.freetype.set_default_resolution(dpi)

        # This normalizes the 0--255 grayscale values
        # and creates the tensors.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        self.fonts = self.load_fonts(font_file, font_size)

        self.dpi = dpi

        self.pad_top = pad_size
        self.pad_bottom = pad_size
        self.pad_left = pad_size
        self.pad_right = pad_size

        self.font_size = font_size
        self.font_color = "black"
        self.bkg_color = "white"

        #Arabic-specific init
        reshaper_configuration = {
            'delete_harakat': False, #whether to drop diacritics
            'delete_tatweel': False,
            'shift_harakat_position': False, #for some renderers, diacritics will appear over the next char when reversed. ours doesnt so False
            'support_ligatures': True, #whether to support ligatures (ie contextual rendering, not render as isolated chars)
            'RIAL SIGN': True,  #replace ر ي ا ل with ﷼
            }
        self.reshaper = ArabicReshaper(configuration=reshaper_configuration)

        # Get the maximum image height
        self.image_height = 0
        for font in self.fonts.values():
            #arabic font sizes can differ from latin (by font), check max height of diacritized arabic separately 
            if 'Arabic' in font.path:
                self.image_height = max(
                    self.image_height,
                    font.get_rect("أَنا كَنَدِيَّةٍ ، وَأَنا أَصْغَرِ إِخْوانِي السَبْعَةِ").height + self.pad_top + self.pad_bottom
                )
            else:
                self.image_height = max(
                    self.image_height,
                    font.get_rect("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.").height + self.pad_top + self.pad_bottom
                )

        # Get the estimated upper bound on character width
        self.estimated_max_char_width = 0
        for font in self.fonts.values():
            self.estimated_max_char_width = max(
                self.estimated_max_char_width,
                font.get_rect("W凯").width + self.pad_left + self.pad_right
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

    def arabic_proc(self, text_to_be_reshaped):
        reshaped_text = self.reshaper.reshape(text_to_be_reshaped)
        bidi_text     = get_display(reshaped_text)        
        return bidi_text
    
    def get_surface(self, line_text, lang="*", remove_subword=True):
        """Creates a single image from an entire line and returns the surface."""
        # Note: lang is currently never changed from default "*". self.fonts["*"] is set to whatever is passed in
        
        # Remove subwords if present
        if remove_subword and "▁" in line_text:
            line_text = line_text.replace(" ", "").replace("▁", " ").strip()
        # Arabic reshaper and bidi proc:
        if "Arabic" in self.fonts[lang].path:
            line_text = self.arabic_proc(line_text)

        # Set the surface width based on a liberal estimate of the space needed for rendering
        surface_width = max(self.window, self.estimated_max_char_width * len(line_text))
        if surface_width > MAX_SURFACE_WIDTH:
            logger.warning(f"Reducing surface width from {surface_width} to {MAX_SURFACE_WIDTH} or {16384//self.stride} slices")
            surface_width = MAX_SURFACE_WIDTH

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

        if crop_width > surface_width:
            old_width = crop_width
            # smallest number <= surface_width that self.stride factorizes into
            while crop_width > surface_width:
                crop_width -= self.stride
            logger.warning(f"{len(line_text)} raw text chars ({old_width // self.window} slices) too wide, truncating {old_width} -> {crop_width}")
            logger.warning(f"line: {line_text.strip()}")

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
        A better way is to call get_tensors(), which slices the tensor directly,
        instead of the image.

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

        return whole_image, image_pieces

    def get_tensor(self, text):
        """Returns a single tensor for the image.
        Shape: (channels=1 x height x width)
        """
        image = self.get_image(text)
        image_tensor = self.transform(image)
        return image_tensor

    def slice(self, image_tensor):
        """Slices a tensor according to stride and window.

        image_tensor: Shape (channels, height, width).
        """
        num_channels, height, width = image_tensor.shape

        tensors = []
        # print("IMAGE_TENSOR:", image_tensor.shape)
        for i in range(0, width - self.window + 1, self.stride):
            # print(f"-> {i}:{i+self.window} of {image_tensor.shape[2]}")
            slice_tensor = image_tensor[:,:,i:i+self.window]
            tensors.append(slice_tensor)

        return torch.stack(tensors)

    def get_tensors(self, text):
        """Returns a stack of sliced tensor produced from rendered text.
        Shape: (num_slices x channels=1 x height x width)
        """
        # image_tensor = self.get_tensor(text)
        # print("GET_TENSORS", image_tensor.shape)
        # print(" SLICES", self.slice(image_tensor).shape)
        return self.slice(self.get_tensor(text))

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

    def dump(self, text, prefix):
        """
        Creates sample images.
        """
        whole_image, image_pieces = self.get_images(text)

        dirname = os.path.dirname(prefix)
        if dirname and not os.path.exists(dirname):
            logger.info(f"Creating samples directory {dirname}")
            os.makedirs(dirname)

        imagepath = f"{prefix}.png"
        cv2.imwrite(imagepath, whole_image)

        logger.info(f"Dumping sample to {imagepath} ({len(image_pieces)} pieces)")
        for i, image in enumerate(image_pieces, 1):
            imagepath = f"{prefix}.{i:02}.png"
            cv2.imwrite(imagepath, image)


def main(args):
    gen = TextImageGenerator(window=args.window,
                             stride=args.stride,
                             font_size=args.font_size,
                             font_file=args.font_file,
    )

#    if args.text is None:
#        num_pixels = 0
#        nonempty_pixels = 0
#        for lineno, line in enumerate(args.input):
#            image_tensor = gen.get_tensor(line).view(-1)
#            num_pixels += image_tensor.shape[0]
#            nonempty_pixels += int(torch.sum(image_tensor < 1))
#            if args.interval and lineno % args.interval == 0:
#                gen.dump(line.rstrip(), f"{args.prefix}.{lineno}")
#        print(f"Pixel density: {nonempty_pixels} / {num_pixels} = {nonempty_pixels / num_pixels:.2f}", file=sys.stderr)
    if args.text is None:
        num_pixels = 0
        nonempty_pixels = 0
        pixelavg = 0
        for lineno, line in enumerate(args.input):
            image_tensor = gen.get_tensor(line).view(-1)
            num_pixels += image_tensor.shape[0]
            nonempty_pixels += int(torch.sum(image_tensor < 1))
            pixelavg += int(torch.sum(image_tensor))
            if args.interval and lineno % args.interval == 0:
                gen.dump(line.rstrip(), f"{args.prefix}.{lineno}")
        print(f"Pixel density (nonempty): {nonempty_pixels} / {num_pixels} = {nonempty_pixels / num_pixels:.2f}", file=sys.stderr)
        print(f"Pixel density (averaged): {pixelavg} / {num_pixels} = {pixelavg / num_pixels:.2f}", file=sys.stderr)

    else:
        gen.dump(args.text, args.prefix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-file", type=str, default="/home/hltcoe/mpost/code/fairseq-ocr/fairseq/data/visual/fonts/NotoMono-Regular.ttf")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--prefix", type=str, default="test_image")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--interval", type=int, default=0, help="Dump every Nth image", metavar="N")
    parser.add_argument("input", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
