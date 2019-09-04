import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
import sys
import argparse
import time
from PIL import Image
import cv2
import csv
import os


class ImageAug(object):

    def __init__(self):
        sometimes = lambda aug: iaa.Sometimes(.90, aug)
        seq = iaa.Sequential(
            [
                sometimes(iaa.CropAndPad(
                    percent=(-0.03, 0.03),
                    pad_mode=["constant", "edge"],
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    rotate=(-3, 3),  # rotate by -45 to +45 degrees
                    shear=(-3, 3),  # shear by -16 to +16 degrees
                )),
                iaa.SomeOf((0, 3),
                    [
                        iaa.GaussianBlur((0, 0.75)),  # blur images with a sigma between 0 and 3.0
                        iaa.Sharpen(alpha=(0, 0.5), lightness=(0.75, 1.5)),  # sharpen images
                        iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5)),  # emboss images
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add gaussian noise to images
                        iaa.Dropout((0.01, 0.05), per_channel=0.5),  # randomly remove up to 10% of the pixels
                        iaa.Invert(0.05, per_channel=True),  # invert color channels
                        iaa.Add((-5, 5), per_channel=0.5),  # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                        iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),  # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.0, 1.0)),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug


class GaussianBlurAug(object):

    def __init__(self, gaussian_sigma):
        seq = iaa.GaussianBlur(gaussian_sigma)
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug


class EdgeDetectAug(object):

    def __init__(self, edgedetect_alpha):
        seq = iaa.EdgeDetect(edgedetect_alpha)
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug

