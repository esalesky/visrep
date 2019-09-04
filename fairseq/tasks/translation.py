# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    IndexedImageLineDataset,
    IndexedImageWordDataset,
    IndexedImageDataset,
    IndexedImageCachedDataset,
    LanguagePairDataset,
    ImagePairDataset,
    ImageAug,
    GaussianBlurAug,
    EdgeDetectAug,
)

from . import FairseqTask, register_task
import torchvision.transforms as transforms


@register_task('translation')
class TranslationTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

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
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='False', type=str, metavar='BOOL',
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
        parser.add_argument('--image-font-size', default=16, type=int,
                            help='Font size')
        parser.add_argument('--image-font-color', default='black', type=str,
                            help='Font color')
        parser.add_argument('--image-bkg-color', default='white', type=str,
                            help='Background color')
        parser.add_argument('--image-channels', default=3, type=int,
                            help='image channels')
        parser.add_argument('--image-width', default=30, type=int,
                            help='Image width')
        parser.add_argument('--image-height', default=150, type=int,
                            help='Image height')
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Image Samples path')

        # Image encoder parameters
        parser.add_argument('--image-stride', default=1, type=int,
                            help='Image stride')
        parser.add_argument('--image-pad', default=1, type=int,
                            help='Image padding')
        parser.add_argument('--image-kernel', default=3, type=int,
                            help='Image kernel size')
        parser.add_argument('--image-maxpool-width', default=0.5, type=float,
                            help='Image frac maxpool ratio width')
        parser.add_argument('--image-maxpool-height', default=0.7, type=float,
                            help='Image frac maxpool ratio height')

        parser.add_argument('--image-embed-dim', default=512, type=int,
                            help='image embedding dimension')

        parser.add_argument("--image-rand-augment", action='store_true',
                            default=False, help="Add image aug library")
        parser.add_argument('--image-verbose', action='store_true',
                            help='image verbose output')
        parser.add_argument("--image-rand-font", action='store_true',
                            default=False, help="Select random font from list")
        parser.add_argument("--image-rand-style", action='store_true',
                            default=False, help="Select random font style")

        parser.add_argument("--image-use-cache", action='store_true',
                            default=False, help="Use image caching")

        parser.add_argument("--image-gaussianblur", action='store_true',
                            default=False, help="Add GaussianBlur")
        parser.add_argument('--image-gaussianblur-sigma', default=0.0, type=float,
                            help='Image aug GaussianBlur sigma')
        parser.add_argument("--image-edgedetect", action='store_true',
                            default=False, help="Add Edge Detection")
        parser.add_argument('--image-edgedetect-alpha', default=0.0, type=float,
                            help='Image aug EdgeDetect alpha')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        print('....loading...', os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        print('Dictionary load complete...')
        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary, is_src=False):
            if self.args.image_type is not None:
                print('Image_type %s, is src %s' % (self.args.image_type, is_src))
                if not is_src:
                    print('...binary image text TGT')
                    if self.args.lazy_load:
                        return IndexedDataset(path, fix_lua_indexing=True, flatten=True)
                    else:
                        return IndexedCachedDataset(path, fix_lua_indexing=True, flatten=True)
                else:
                    print('...binary image text SRC')
                    if self.args.lazy_load:
                        return IndexedImageDataset(path, dictionary,
                                    self.args.image_verbose,
                                    self.args.image_font_path, self.args.image_font_size, self.args.image_font_color, self.args.image_bkg_color,
                                    self.args.image_width, self.args.image_height, fix_lua_indexing=True, flatten=True,
                                    image_use_cache=self.args.image_use_cache,
                                    image_rand_font=self.args.image_rand_font, image_rand_style=self.args.image_rand_style)
                    else:
                        return IndexedImageCachedDataset(path, dictionary,
                                    self.args.image_verbose,
                                    self.args.image_font_path, self.args.image_font_size, self.args.image_font_color, self.args.image_bkg_color,
                                    self.args.image_width, self.args.image_height, fix_lua_indexing=True, flatten=True,
                                    image_use_cache=self.args.image_use_cache,
                                    image_rand_font=self.args.image_rand_font, image_rand_style=self.args.image_rand_style)

            elif self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        print('\n...load datasets ', split)
        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        print('Dataset not found: {} ({}) \n'.format(split, data_path))
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict, is_src=True))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict, is_src=False))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        if self.args.image_type != None:
            print('loading ImagePairDataset', split)

            simple_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

            aug_transform = transforms.Compose([
                ImageAug(),
                transforms.ToTensor(),
            ])

            blur_transform = transforms.Compose([
                GaussianBlurAug(self.args.image_gaussianblur_sigma),
                transforms.ToTensor(),
            ])

            edge_transform = transforms.Compose([
                EdgeDetectAug(self.args.image_edgedetect_alpha),
                transforms.ToTensor(),
            ])

            if self.args.image_rand_augment:
                print('...USING random transform')
                transform = aug_transform
            elif self.args.image_gaussianblur:
                print('...USING gaussian blur')
                transform = blur_transform
            elif self.args.image_edgedetect:
                print('...USING edge detect')
                transform = edge_transform
            else:
                print('...USING simple_transform')
                transform = simple_transform

            self.datasets[split] = ImagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                transform,
                self.args.image_verbose,
                self.args.image_samples_path,
                self.args.image_font_path, self.args.image_font_size,
                self.args.image_width, self.args.image_height,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
        else:
            print('loading LanguagePairDataset', split)
            self.datasets[split] = LanguagePairDataset(
                src_dataset, src_dataset.sizes, self.src_dict,
                tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            print('split complete', split)

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
