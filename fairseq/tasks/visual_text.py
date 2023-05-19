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
            default=1024,
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

        parser.add_argument('--image-embed-type', type=str, default='1layer',
                            choices=["vista", "visonly", "direct", "1layer", "2layer", "smallvista"],
                            help='OCR embedding method (visonly is for backwards compat, means vista)')
        parser.add_argument("--image-embed-normalize", action="store_true", default=False,
                            help='Apply batch norm to convolutions (always true for "vista")')
        parser.add_argument("--image-channel-increment", type=int, default=1,
                            help='Amount to increment channel capacity in odd channel layers')
        parser.add_argument("--image-bridge-relu", action="store_true",
                            help='add ReLU to bridge')

        parser.add_argument("--image-verbose", action='store_true',
                            help='Display verbose debug')
        parser.add_argument("--image-pretrain-path", type=str, default=None,
                            help='Load pretrain sentence embeddings')

        add_visual_text_args(parser)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang

        self.image_generator = VisualTextTask.build_image_generator(args)

        if self.source_lang is None or self.target_lang is None:
            raise ValueError("You have to set --source-lang and --target-lang")


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

        self.datasets[split] = VisualTextDataset.load(
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

    @staticmethod
    def build_image_generator(args):
        """Builds an image generator.

        At training time, args.image_font_path will be defined.  At
        test time, it shouldn't be, so this will not get built until
        it is called from build_model(), which has access to the
        params loaded from the saved model.

        This is ugly, but it ensures parameter continuity between
        training and inference. It's basically a workaround for the
        fact that parameters in fairseq are task-level, but are stored
        at the model level (in the checkpoint), but models aren't
        loaded until after the task is instantiated.
        """
        image_generator = None
        if args.image_font_path is not None:
            image_generator = TextImageGenerator(pixels_per_patch=args.pixels_per_patch,
                                                 font_size=args.image_font_size,
                                                 font_file=args.image_font_path,
            )

        return image_generator

    def build_model(self, args):
        # At inference, we have to build the image generator here to replace defaults,
        # after we have the parameters loaded from the model
        self.image_generator = VisualTextTask.build_image_generator(args)

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
