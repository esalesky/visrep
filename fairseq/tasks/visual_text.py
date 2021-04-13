# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

from fairseq.data import Dictionary, encoders
from fairseq.data.visual import VisualTextDataset, TextImageGenerator
from fairseq.tasks import LegacyFairseqTask, register_task

logger = logging.getLogger(__name__)

from fairseq.options import add_visual_text_args

import fairseq.data.visual.image_generator as imgen


DEFAULT_FONT_SIZE = imgen.DEFAULT_FONT_SIZE
DEFAULT_WINDOW = imgen.DEFAULT_WINDOW
DEFAULT_STRIDE = imgen.DEFAULT_STRIDE


@register_task("visual_text")
class VisualTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument("-s", "--source-lang", default=None, metavar="SOURCE",
                       help="source language")
        parser.add_argument("-t", "--target-lang", default=None, metavar="TARGET",
                       help="target language")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--target-dict",
            type=str,
            metavar="DICT",
            help="path to target language dictionary",
        )
        parser.add_argument(
            "--shuffle",
            action="store_true",
            help="Shuffle the training data",
        )

        parser.add_argument("--image-verbose", action='store_true',
                            help='Display verbose debug')
        parser.add_argument("--image-pretrain-path", type=str, default=None,
                            help='Load pretrain sentence embeddings')
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Directory to dump image samples to')
        parser.add_argument('--image-samples-interval', default=1000, type=int, metavar="N",
                            help='Dump every Nth sample image')

        add_visual_text_args(parser)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang

        self.image_generator = None
        self.build_image_generator(args)

        if self.source_lang is None or self.target_lang is None:
            raise ValueError("You have to set --source-lang and --target-lang")

        assert args.image_stride > 0 and args.image_stride <= args.image_window, "Stride must be nonzero and not greater than the window size"

    @classmethod
    def setup_task(cls, args, **kwargs):
        dict_path = op.join(args.data, args.target_dict)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"dictionary size ({args.target_dict}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict)

    def build_criterion(self, args):
        # TODO

        from fairseq import criterions

        # if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
        #     raise ValueError(
        #         'Please set "--ignore-prefix-size 1" since '
        #         "target language ID token is prepended as BOS."
        #     )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")

        self.datasets[split] = VisualTextDataset.from_text_path(
            self.args.data,
            self.args,
            split,
            self.source_lang,
            self.target_lang,
            self.image_generator,
            self.tgt_dict,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_image_generator(self, args):
        """Builds an image generator.

        At training time, args.image_font_path will be defined.  At
        test time, it shouldn't be, so this will not get built until
        interactive.py calls this function on the task, after loading
        the model args from the model. This is ugly, but it ensures
        parameter continuity between training and inference. It's
        basically a workaround for the fact that parameters in fairseq
        are task-level, but are stored at the model level (in the
        checkpoint), but models aren't loaded until after the task
        is instantiated.
        """
        if args.image_font_path is not None:
            self.image_generator = TextImageGenerator(window=args.image_window,
                                                      stride=args.image_stride,
                                                      font_size=args.image_font_size,
                                                      font_file=args.image_font_path,
            )


    def build_model(self, args):
        # args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        # args.input_channels = self.data_cfg.input_channels
        return super(VisualTextTask, self).build_model(args)

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        return lines, [len(line) for line in lines]

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """
        Turn the source tokens into images, and create a dictionary
        object with
        { id, net_input: { src_tokens, src_lengths } }

        """
        dataset = VisualTextDataset.from_text(
            self.args,
            src_tokens,
            self.image_generator,
            self.tgt_dict,
            kwargs["constraints"],
        )

        return dataset
