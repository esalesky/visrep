# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import sys
import cv2

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    LanguagePairDataset,
    PrependTokenDataset,
)

from fairseq.data.image_dataset import (
    ImageDataset,
    ImagePairDataset,
    ImageGenerator,
)

from . import FairseqTask, register_task
import torchvision.transforms as transforms

from imgaug import augmenters as iaa


""" class ImageAug(object):

    def __init__(self):

        def sometimes(aug):
            return iaa.Sometimes(.90, aug)
        seq = iaa.Sequential(
            [
                sometimes(
                    iaa.CropAndPad(
                        percent=(-0.03, 0.03),
                        pad_mode=["constant", "edge"],
                        pad_cval=(0, 255)
                    )
                ),
                sometimes(
                    iaa.Affine(
                        rotate=(-4, 4),  # rotate by -45 to +45 degrees
                        shear=(-3, 3),  # shear by -16 to +16 degrees
                    )
                ),
                iaa.OneOf([
                    iaa.GaussianBlur((0, 0.75)),
                    iaa.Dropout((0.01, 0.03), per_channel=0.5)
                ])
            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug """


class ImageAug(object):

    def __init__(self):
        def sometimes(aug): return iaa.Sometimes(.50, aug)
        seq = iaa.Sequential(
            [
                iaa.OneOf([
                    iaa.GaussianBlur((0.25, 1.0)),  # blur images with a sigma
                    # randomly remove up to n% of the pixels
                    iaa.Dropout((0.01, 0.05), per_channel=0.5),
                    iaa.CropAndPad(
                        percent=(-0.05, 0.05),
                        pad_mode=["constant"],
                        pad_cval=255
                    ),
                    iaa.Affine(
                        shear=(-3, 3),  # shear by -16 to +16 degrees
                    ),
                    iaa.Affine(
                        rotate=(-4, 4),  # rotate by -45 to +45 degrees
                    )
                ]),

            ],
            random_order=True
        )
        self.seq = seq

    def __call__(self, img):
        images_aug = self.seq.augment_image(img)
        return images_aug


def load_visual_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    args=None,
    image_cache=None


):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    print('....loading dataset, split', split)
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(
                data_path, '{}.{}-{}.'.format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    'Dataset not found: {} ({})'.format(split, data_path))

        path = prefix + src
        dictionary = src_dict
        print('...loading ImageDataset', path, len(dictionary))
        if args.image_type:
            if args.image_augment:
                transform = transforms.Compose([
                    # ImageAug(),
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ])
        else:
            transform = None

        img_dataset = ImageDataset(path, dictionary, append_eos=True, reverse_order=False,
                                   transform=transform,
                                   image_type=args.image_type,
                                   image_font_path=args.image_font_path,
                                   image_height=args.image_height,
                                   image_width=args.image_width,
                                   image_cache=image_cache)
        src_datasets.append(
            img_dataset
        )

        tgt_datasets.append(
            data_utils.load_indexed_dataset(
                prefix + tgt, tgt_dict, dataset_impl)
        )

        print('| {} {} {}-{} {} examples'.format(data_path,
                                                 split_k, src, tgt, len(src_datasets[-1])))

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets)

    if len(src_datasets) == 1:
        src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(
            tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(
            data_path, '{}.align.{}-{}'.format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl)

    print('...loading ImagePairDataset', split)

    return ImagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=args.left_pad_source,
        left_pad_target=args.left_pad_target,
        max_source_positions=args.max_source_positions,
        max_target_positions=args.max_target_positions,
    )


@register_task('visualmt')
class VisualMTTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--load-alignments', action='store_true',
                            help='load the binarized alignments')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # Image dataset loader parameters
        parser.add_argument('--image-type', type=str, default=None,
                            help='use word or line image dataset (None | word | line)')
        parser.add_argument('--image-font-path', default=None, type=str,
                            help='Font path')
        parser.add_argument('--image-width', default=64, type=int,
                            help='Image width')
        parser.add_argument('--image-height', default=32, type=int,
                            help='Image height')
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Image Samples path')
        parser.add_argument("--image-use-cache", default=False, action='store_true',
                            help='Cache image dictionary')
        parser.add_argument("--image-augment", default=False, action='store_true',
                            help='Use image augmentation')

    def __init__(self, args, src_dict, tgt_dict, image_cache):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.image_cache = image_cache

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        if getattr(args, 'raw_text', False):
            utils.deprecation_warning(
                '--raw-text is deprecated, please use --dataset-impl=raw')
            args.dataset_impl = 'raw'
        elif getattr(args, 'lazy_load', False):
            utils.deprecation_warning(
                '--lazy-load is deprecated, please use --dataset-impl=lazy')
            args.dataset_impl = 'lazy'

        paths = args.data.split(':')
        assert len(paths) > 0
        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(
                paths[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception(
                'Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(
            paths[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(
            args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(
            args.target_lang, len(tgt_dict)))

        image_cache = None
        write_dict = False
        if args.image_samples_path:
            image_dir = os.path.join(args.image_samples_path, 'dict')
            if os.path.exists(image_dir):
                print(
                    f'* Not Dumping dictionary since {image_dir} exists', file=sys.stderr)
            else:
                os.makedirs(image_dir)
                print(
                    f'* Dumping {len(src_dict)} word images to {image_dir}', file=sys.stderr)
                write_dict = True

        if write_dict or args.image_use_cache:
            image_generator = ImageGenerator(font_file_path=args.image_font_path,
                                             image_width=args.image_width,
                                             image_height=args.image_height)
            image_cache = {}
            print('...BUILDING CACHE')
            for index in range(len(src_dict)):
                word = src_dict[index]
                img_data = image_generator.get_image(word,
                                                     font_name=image_generator.font_list[0],
                                                     font_size=16, font_style=1,
                                                     font_color='black', bkg_color='white', font_rotate=0,
                                                     pad_top=5, pad_bottom=5, pad_left=5, pad_right=5)
                img_data = image_generator.resize_or_pad(
                    img_data, height=args.image_height, width=args.image_width)
                if write_dict:
                    image_path = os.path.join(image_dir, f'{index}_{word}.png')
                    cv2.imwrite(image_path, img_data)
                if args.image_use_cache:
                    image_cache[word] = img_data
            print('CACHE COMPLETE', len(image_cache))

        return cls(args, src_dict, tgt_dict, image_cache)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_visual_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=self.args.load_alignments,
            args=self.args,
            image_cache=self.image_cache,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
