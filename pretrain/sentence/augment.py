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
        print('...ImageAug v1')
        def sometimes(aug): return iaa.Sometimes(.90, aug)
        seq = iaa.Sequential(
            [
                iaa.SomeOf((0, 3),
                           [
                    iaa.GaussianBlur((0, 0.75)),
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(
                        0.75, 1.5)),  # sharpen images
                    iaa.Emboss(alpha=(0, 0.5), strength=(
                        0, 1.5)),  # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(
                        0.0, 0.05*255), per_channel=0.5),
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-5, 5), per_channel=0.5),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.ContrastNormalization((0.5, 1.0), per_channel=0.5),
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


class OcrAug(object):

    def __init__(self):
        print('...OcrAug')
        def sometimes(aug): return iaa.Sometimes(.90, aug)

        seq = iaa.Sequential(
            [
                sometimes(
                    iaa.OneOf([
                        iaa.Affine(shear=(-1, 1)),
                        iaa.imgcorruptlike.GaussianBlur(severity=(1, 1)),
                        iaa.imgcorruptlike.ElasticTransform(severity=(1, 2)),
                        iaa.SaltAndPepper((.01, .05)),
                        iaa.imgcorruptlike.JpegCompression(severity=(1, 2)),
                        iaa.imgcorruptlike.Pixelate(severity=(1, 2)),
                        iaa.Crop(percent=(.01, .02)),
                    ]),

                ),
            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug
