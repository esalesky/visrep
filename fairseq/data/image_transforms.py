import torch
import math
import random
import numpy as np
import numbers
import types
import collections
import cv2
from scipy.interpolate import griddata


class Scale(object):
    """Rescales the input image to the given 'size'.
    If 'size' is a 2-element tuple or list in the order of (width, height), it will be the exactly size to scale.
    If 'size' is a number, it will indicate the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the exactly size or the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(
        self,
        size=None,
        new_h=None,
        new_w=None,
        preserve_apsect_ratio=True,
        interpolation=cv2.INTER_CUBIC,
    ):
        assert (
            isinstance(size, int)
            or (isinstance(size, collections.Iterable) and len(size) == 2)
            or isinstance(new_h, int)
            or isinstance(new_w, int)
        )
        self.size = size
        self.new_h = new_h
        self.new_w = new_w
        self.preserve_apsect_ratio = preserve_apsect_ratio
        self.interpolation = interpolation

    def __call__(self, img):
        if len(img.shape) == 2:
            h, w = img.shape
        else:
            h, w, c = img.shape

        # First check if we specified specific height and/or width
        if not (self.new_h is None and self.new_w is None):
            local_new_h = self.new_h or h
            local_new_w = self.new_w or w

            if self.preserve_apsect_ratio and self.new_h is None:
                local_new_h = int(h * float(self.new_w / w))
            if self.preserve_apsect_ratio and self.new_w is None:
                local_new_w = int(w * float(self.new_h / h))

            if local_new_w <= 0 or local_new_h <= 0:
                print(
                    "Warning!! local_new_h = %d, local_new_w = %d; orig h = %d, orig w = %d"
                    % (local_new_h, local_new_w, h, w)
                )
                # let's fallback to have a non-zero width; this will give junk results but at least not crash
                local_new_w = 1

            return cv2.resize(img, (local_new_w, local_new_h), self.interpolation)

        # Next, fall back to old tuple API
        if isinstance(self.size, int):
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return cv2.resize(img, (ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return cv2.resize(img, (ow, oh), self.interpolation)
        else:
            return cv2.resize(img, self.size, self.interpolation)

    def __repr__(self):
        return self.__class__.__name__ + "()"

