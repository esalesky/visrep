#!/usr/bin/env python3

import cv2
import logging
import os
import sys
import random

import numpy as np
import torch
import torchvision.transforms as transforms

import glob
import math
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import cairo
import gi
import manimpango
from fontTools import ttLib

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo


# default values
DEFAULT_FONT_SIZE = 10
DEFAULT_PAD_SIZE = 2
DEFAULT_PPB = 24
DEFAULT_WINDOW = DEFAULT_PPB
DEFAULT_STRIDE = 12
MAX_SEQ_LENGTH = 529
MAX_PIXELS_LEN = MAX_SEQ_LENGTH * DEFAULT_PPB

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
os.environ["PANGOCAIRO_BACKEND"] = "fontconfig"
os.environ["FONTCONFIG_FILE"] = "/exp/esalesky/newrender/visrep/fairseq/data/visual/abc.conf"

fallback_fonts_dir = '/exp/esalesky/newrender/visrep/fairseq/data/visual/fonts/fallback_fonts'
logger = logging.getLogger(__name__)

class TextImageGenerator():
    def __init__(
        self,
        font_file: str,
        font_size: int = DEFAULT_FONT_SIZE,
        font_color: str = "black",
        background_color: str = "white",
        rgb: bool = False,
        dpi: int = 120,
        pad_size: int = DEFAULT_PAD_SIZE,
        pixels_per_patch: int = DEFAULT_PPB,
        stride: int = DEFAULT_STRIDE, 
        max_seq_length: int = MAX_SEQ_LENGTH,
        fallback_fonts_dir: Optional[str] = fallback_fonts_dir,
        **kwargs,
    ):
        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color
        self.rgb = rgb

        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.image_height = pixels_per_patch
        self.image_width = pixels_per_patch
        self.window = pixels_per_patch
        self.stride = stride
        self.dpi = dpi
        self.PANGO_SCALE = 1024

        self.font = None
        self.fonts_list = None
        self.fallback_fonts_dir = fallback_fonts_dir
        self.load_font()

        # This normalizes the 0--255 grayscale values
        # and creates the tensors.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        logger.info(f"Image generator created with {self.font_size}pt {font_file} with image height {self.pixels_per_patch}")

    @property
    def height(self):
        return self.image_height

    @property
    def width(self):
        return self.image_width

    @property
    def max_pixels_len(self):
        return self.max_seq_length * self.pixels_per_patch

    def px2patch_ceil(self, px: int):
        return math.ceil(px / self.pixels_per_patch)

    def px2patch_floor(self, px: int):
        return math.floor(px / self.pixels_per_patch)

    def patch2px(self, patch: int):
        return patch * self.pixels_per_patch

    @staticmethod
    def is_rtl(text: str) -> bool:
        """
        Returns whether a piece of text is written in a right-to-left (RTL) script based on a majority vote of the
        first, middle, and last characters in the text after removing whitespace, punctuation, and numbers

        Returns:
            Whether the piece of text is RTL, type `bool`
        """
        text = text.translate(str.maketrans("", "", string.whitespace))
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.translate(str.maketrans("", "", string.digits))

        if len(text) == 0:
            return False

        vote = 0
        for char in [text[0], text[-1], text[len(text) // 2]]:
            if Pango.unichar_direction(char) == Pango.Direction.RTL:
                vote += 1

        is_rtl = vote >= 2
        return is_rtl

    def _get_offset_to_next_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the next patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return min(
            math.ceil(x / self.pixels_per_patch) * self.pixels_per_patch,
            self.max_pixels_len - self.pixels_per_patch,
        )

    def _get_offset_to_previous_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the previous patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return math.floor(x / self.pixels_per_patch) * self.pixels_per_patch

    def get_empty_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context, List[int]]:
        """
        Create and return a tuple containing
        (1) an empty surface that we will later render the text to,
        (2) a context object used to draw on the surface, and
        (3) an empty list in which we keep track of where to insert black separator patches

        Returns:
            A tuple of type (`~cairo.ImageSurface`, `~cairo.Context`, `List[int]`) containing the blank surface,
            the context object, an the empty list for keeping track of black separator patches, respectively
        """

        cairo_format = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        surface = cairo.ImageSurface(cairo_format, self.max_pixels_len, self.pixels_per_patch)
        context = cairo.Context(surface)
        if self.rgb:
            context.set_source_rgb(1.0, 1.0, 1.0)
            context.rectangle(0, 0, self.max_pixels_len, self.pixels_per_patch)
            context.fill()
            context.set_source_rgb(0.0, 0.0, 0.0)
        sep_patches = []
        return surface, context, sep_patches

    def get_cluster_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the grapheme cluster level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, cluster-by-cluster).
        """
        logical_extents = layout_iter.get_cluster_extents()[1]
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_char_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the character level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, character-by-character).
        """
        logical_extents = layout_iter.get_char_extents()
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_text_offset_mapping(
        self, layout: Pango.Layout, offset: int, text_width: int, text_shift: int = 0, rtl: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Returns an offset mapping, i.e. a list that keeps track of where in the rendered image each character of
        the input text is located. It has the form [(start_character_index, end_character_index)] with an entry for
        every image patch.

        Args:
            layout (`Pango.Layout`):
                The layout containing the rendered text.
            offset (`int`):
                The offset in pixels of the first character of the text from the beginning of the first patch.
            text_width (`int`):
                The logical width of the rendered text in pixels.
            text_shift (`int`, *optional*, defaults to 0):
                The number of pixels that a text is shifted to the right on the layout, i.e. the starting position
                as pixel offset of the first image patch corresponding to this text. This value is typically set when
                obtaining the offset_mapping for the second text in a rendered text pair.
            rtl (`bool`, *optional*, defaults to False):
                Indicates whether the text is rendered right-to-left (RTL), in which case the offset mapping needs to
                account for the fact that the actual beginning of the text is on the right.
        """
        # Find starting positions for each character in the text
        layout_iter = layout.get_iter()
        # Get offset for first character
        idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
        character_positions = [x_offset + offset]
        # Loop through remaining characters
        while layout_iter.next_char():
            idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
            character_positions.append(x_offset + offset)

        # Find starting positions for each cluster in the text. A cluster may consist of multiple characters rendered
        # as one glyph
        layout_iter = layout.get_iter()
        # Get offset for first cluster
        idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
        cluster_positions = [x_offset + offset]
        # Loop through remaining clusters
        while layout_iter.next_cluster():
            idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
            cluster_positions.append(x_offset + offset)

        # In case clusters exist, the length of the cluster list will be shorter than the length of the character list.
        # However, the offset mapping maps between clusters in the rendered image and characters in the written text,
        # so we need to assign a starting position to each character in the cluster position list. We do this by
        # assigning the starting position of a cluster to each character in that cluster.
        if len(character_positions) != len(cluster_positions):
            buffer = []
            cluster_idx = 0
            for idx in range(len(character_positions)):
                if cluster_idx == len(cluster_positions) or character_positions[idx] != cluster_positions[cluster_idx]:
                    buffer.append(cluster_positions[cluster_idx - 1])
                else:
                    buffer.append(character_positions[idx])
                    cluster_idx += 1

            buffered_cluster_positions = buffer
        else:
            buffered_cluster_positions = character_positions

        # Retrieve the rendered text from the layout. This is necessary for RTL scripts
        text = layout.get_text()

        # This means we add a full blank patch
        if self._get_offset_to_next_patch(text_width) - text_width < offset - self._get_offset_to_previous_patch(
            offset
        ):
            is_blank_patch_inserted = True
        else:
            is_blank_patch_inserted = False

        buffered_cluster_positions.append(self._get_offset_to_next_patch(text_width + offset))

        offset_mapping = []
        patch_start = 0
        cleared = 0
        for k, v in enumerate(buffered_cluster_positions):
            if v - text_shift >= self.pixels_per_patch * (len(offset_mapping) + 1):
                if v - text_shift == self.pixels_per_patch * (len(offset_mapping) + 1):
                    patch_end = k
                else:
                    patch_end = k - 1
                offset_mapping.append(
                    (
                        (len(text) - patch_start) if rtl else patch_start,
                        (len(text) - patch_end) if rtl else patch_end,
                    )
                )

                patch_start = patch_end
                cleared += 1

        # The `cleared` variable counts how many times we have added a character span to the offset mapping, i.e.,
        # cleared the cluster buffer. If at the end of processing the buffered_cluster_positions we still have clusters
        # in the buffer, we add the remainder to the offset mapping
        if cleared < self.px2patch_ceil(text_width + offset - text_shift):
            if rtl:
                offset_mapping.append((len(text) - patch_start, 0))
            else:
                offset_mapping.append((patch_start, len(buffered_cluster_positions)))

        # We add padding between the end of the rendered sequence and the final black separator patch. If this padding
        # happens to be a full patch, this means that we need to merge the penultimate and last patches in the offset
        # mapping and add a buffer to the offset mapping
        if is_blank_patch_inserted:
            offset_mapping[-2] = (
                offset_mapping[-2][0],
                offset_mapping[-1][1],
            )
            offset_mapping[-1] = (-1, -1)

        # print(f"{len(offset_mapping) = }")

        return offset_mapping

    def pad_or_truncate_offset_mapping(self, offset_mapping: List[Tuple[int, int]]):
        if len(offset_mapping) >= self.max_seq_length:
            offset_mapping = offset_mapping[: self.max_seq_length - 1] + [(0, 0)]
        if len(offset_mapping) < self.max_seq_length:
            offset_mapping += (self.max_seq_length - len(offset_mapping)) * [(0, 0)]
        return offset_mapping

    def _render_single_word(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)

        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in word: {word}. Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        offset = self._get_offset_to_next_patch(offset + width)

        return context, layout, offset

    def _render_single_sentence(
        self, sentence: str, offset: int, context, max_length: Optional[int] = None, rtl: bool = False, debpe: bool = True
    ) -> Tuple[cairo.Context, Tuple[Pango.Layout, Pango.Layout], int]:
        """
        Renders a single sentence to a surface with a horizontal offset, i.e. the rendered
        sentence begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        text vertically on the surface

        Args:
            sentence (`str`):
                The sentence to be rendered
            offset (`int`):
                The horizontal starting position of the rendered sentence on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            max_length (`int`, *optional*, defaults to None):
                Maximum number of patches that the rendered sentence may fill on the surface. If set, anything longer
                than this number of patches will be truncated.

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the width of the rendered
            sentence in pixels, type `int`
        """
        pango_context = PangoCairo.create_context(context)
        pango_context.set_font_description(self.font)
        layout = Pango.Layout(pango_context)

        # Remove sentencepiece subword tokenization if present
        if debpe and "▁" in sentence:
            sentence = sentence.replace(" ", "").replace("▁", " ").strip()

        if rtl:
            layout.set_auto_dir(False)
            pango_context.set_base_dir(Pango.Direction.RTL)
            layout.set_alignment(Pango.Alignment.RIGHT)
        layout.set_text(sentence, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in sentence: {sentence}. Consider"
                f" double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()
        full_width = width
        full_layout = layout
        truncated_layout = layout

        if max_length is not None:
            if self.px2patch_ceil(offset + width) > max_length:
                truncated_layout = Pango.Layout(pango_context)

                # print(
                #     f"Truncating {sentence} ({self.px2patch_ceil(offset + width)} patches) to fit {max_length = }."
                # )

                # Run binary search to find truncation point
                lo = 0
                hi = len(sentence)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    new_sentence = sentence[:mid]
                    truncated_layout.set_text(new_sentence, -1)
                    width, height = truncated_layout.get_pixel_size()
                    if self.px2patch_ceil(offset + width) < max_length:
                        lo = mid + 1
                    elif self.px2patch_ceil(offset + width) > max_length:
                        hi = mid - 1
                    else:
                        break
                # print(f"New sentence = {new_sentence}, width = {self.px2patch_ceil(offset + width)} patches")

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0)
        context.move_to(*position)

        PangoCairo.show_layout(context, truncated_layout)

        return context, (full_layout, truncated_layout), full_width


    def _render_text_to_surface(self, text:str):
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a surface and keeps track of
        metadata, e.g. how many patches in the rendered surface contain text, i.e. are neither blank nor black separator
        patches
        Args:
            text (`str`):
                The piece of text to be rendered


        Returns:
            A numpy array of pixel_values
        """

        # Clean text
        text = text.replace("\n", " ")

        surface, context, sep_patches = self.get_empty_surface()

        offset = 2

        # Render text
        context, (_, layout), text_width = self._render_single_sentence(text, offset, context)

        # Adding eos patch
        eos_patch_offset = self._get_offset_to_next_patch(2 + text_width + 2)
        num_text_patches = self.px2patch_floor(eos_patch_offset)
        sep_patches.append(eos_patch_offset)

        pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches)

        return pixel_values[:,0:eos_patch_offset+self.pixels_per_patch] # actual seq len + eos patch
##        return pixel_values[:,0:eos_patch_offset] # (no eos patch)

    def get_image_from_surface(self, surface: cairo.ImageSurface, sep_patches: List[int] = []) -> np.ndarray:
        """
        Transforms a surface containing a rendered image into a numpy image and inserts black separator patches.

        Args:
            surface (`cairo.ImageSurface`):
                The cairo surface containing the rendered text
            sep_patches (`List[int]`):
                A list of offset values at which black separator patches will be inserted
        Returns:
            An image of type `np.ndarray` of size [self.pixels_per_patch, self.max_pixels_len]
        """

        # Get image data from surface
        data = surface.get_data()
        if self.rgb:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len, 4))
            data = data[:, :, :3]
            # Reverse channels BGR -> RGB
            image = data[:, :, ::-1]
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch, :] = 0
        else:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len))
            image = np.invert(data)
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch] = 0

        return image


    def load_font(self) -> None:
        """
        Loads the font from specified font file with specified font size and color.
        """

        logger.info(f"Loading font from {self.font_file}")

        manimpango.register_font(self.font_file)
        if self.fallback_fonts_dir is not None:
            for fallback_font in glob.glob(os.path.join(self.fallback_fonts_dir, "*tf")):
                logger.info(f"Loading fallback font {fallback_font}")
                manimpango.register_font(fallback_font)
        self.fonts_list = manimpango.list_fonts()

        font_family_name = ttLib.TTFont(self.font_file)["name"].getDebugName(1)

        scaled_font_size = (self.dpi / 72) * self.font_size
        font_str = f"{font_family_name} {scaled_font_size}px"
        self.font = Pango.font_description_from_string(font_str)

    def get_image(self, text: str):
        """
        Render a piece of text to a surface, convert the surface into an image and return the image

        Args:
            text (str):
                The text to be rendered

        Returns:
            A numpy array of pixel_values
        """
        pixel_values = self._render_text_to_surface(text)

        return pixel_values

    def get_images(self, text: str):
        """
        Returns 'tokenized' images of fixed height and width from image of a line of text.
        A better (faster) way is to call get_tensors(), which slices the tensor directly,
        instead of the image.

        Shape: slices x height x width
        """

        sent_image = self.get_image(text)
        (height, width) = sent_image.shape
        
        # Slide a window over the image
        image_pieces = []

        for start in range(0, width - self.window + 1, self.stride):
            token = sent_image[:,start:start+self.window]
            image_pieces.append(token)

        return sent_image, image_pieces

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
        for start in range(0, width - self.window + 1, self.stride):
            slice_tensor = image_tensor[:,:,start:start+self.window]
            tensors.append(slice_tensor)

        return torch.stack(tensors)

    def get_tensors(self, text):
        """Returns a stack of sliced tensor produced from rendered text.
        Shape: (num_slices x channels=1 x height x width)
        """
        return self.slice(self.get_tensor(text))
    
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
    generator = TextImageGenerator(
        font_file=args.font_file,
        fallback_fonts_dir=args.fallback_fonts_dir,
        font_size=args.font_size,
        pixels_per_patch=args.pixels_per_patch,
        rgb=args.rgb,
        max_seq_length=args.max_seq_length
    )
    if args.text is not None:
        generator.dump(args.text, args.prefix)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-file", type=str, default="/exp/esalesky/newrender/visrep/fairseq/data/visual/fonts/NotoSans-Regular.ttf")
    parser.add_argument("--fallback-fonts-dir", type=str, default="/exp/esalesky/newrender/visrep/fairseq/data/visual/fonts/fallback_fonts")
    parser.add_argument("--font-size", type=int, default=DEFAULT_FONT_SIZE)
    parser.add_argument("--pixels-per-patch", type=int, default=DEFAULT_PPB)
    parser.add_argument("--rgb", type=bool, default=False)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--prefix", type=str, default="test_image")
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)
