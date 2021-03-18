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
        parser.add_argument("--image-checkpoint-path", type=str,
                            default=None, help="Image checkpoint path")
        parser.add_argument("--image-pretrain-path", type=str, default=None,
                            help='Load pretrain sentence embeddings')
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Image samples path')
        parser.add_argument('--image-samples-interval', default=1000, type=int,
                            help='Image sample frequency')

        parser.add_argument("--image-embed-type", default='avg', type=str,
                            help='Image embed type [add, avg, concat, visonly, tokonly]')
        parser.add_argument('--image-embed-dim', default=512, type=int,
                            help='Image embed dim')
        parser.add_argument('--image-embedding-normalize',  action='store_true',
                            help='Image embedding l2 normalize')

        parser.add_argument('--image-pretrain-eval-only', action='store_true',
                            help='OCR pretrain model in eval only mode')
        parser.add_argument("--image-display-mod", type=int,
                            default=400, help="display ocr ref/hyp and write image")
        parser.add_argument('--image-augment', action='store_true',
                            help='Augment images during training')
        parser.add_argument('--image-enable-src-loss', action='store_true',
                            help='Enable src loss')
        parser.add_argument('--image-src-loss-scale', type=float, default=1.0,
                            help='Image src loss scale')
        parser.add_argument('--image-tgt-loss-scale', type=float, default=1.0,
                            help='Image tgt loss scale')

        parser.add_argument('--image-font-path', type=str,
                            default='', help='Input font file')
        parser.add_argument("--image-font-size", type=int,
                            default=16, help="Font size")
        parser.add_argument("--image-height", type=int,
                            default=32, help="Image height")
        parser.add_argument("--image-width", type=int,
                            default=32, help="Image width")
        parser.add_argument("--image-surface-width", type=int,
                            default=7000, help="Image surface width")
        parser.add_argument("--image-surface-height", type=int,
                            default=200, help="Image surface height")
        parser.add_argument("--image-start-x", type=int,
                            default=25, help="Image start x")
        parser.add_argument("--image-start-y", type=int,
                            default=25, help="Image start y")
        parser.add_argument("--image-pad-size", type=int,
                            default=2, help="Image pad size")
        parser.add_argument("--image-dpi", type=int,
                            default=120, help="Image dpi")

        parser.add_argument("--image-stride", type=int, default=30,
                            help="Stride width in image pixels")
        parser.add_argument("--image-stride-overlap", type=int, default=10,
                            help="Stride overlap in image pixels")

        parser.add_argument('--image-cache-path', default=None, type=str,
                            help='Image cache path')


    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        self.image_generator = TextImageGenerator(stride=args.image_stride,
                                                  overlap=args.image_stride_overlap,
                                              )

        if self.source_lang is None or self.target_lang is None:
            raise ValueError("You have to set --source-lang and --target-lang")

        assert args.image_stride > args.image_stride_overlap, "Stride must be larger than overlap"
        assert args.image_stride % args.image_stride_overlap == 0, "overlap must be a factor of stride"

        # self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))

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
        # pre_tokenizer = self.build_tokenizer(self.args)
        # bpe_tokenizer = self.build_bpe(self.args)
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

    def build_model(self, args):
        # args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        # args.input_channels = self.data_cfg.input_channels
        return super(VisualTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        lang_token_ids = {
            i
            for s, i in self.trg_dict.indices.items()
            if VisualTextDataset.is_lang_tag(s)
        }
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    # def build_tokenizer(self, args):
    #     logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
    #     return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    # def build_bpe(self, args):
    #     logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
    #     return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return VisualTextDataset(
            "interactive", False, self.args, src_tokens, src_lengths
        )
