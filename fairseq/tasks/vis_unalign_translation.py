
import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    indexed_dataset,
    PrependTokenDataset,
)

#from . import FairseqTask, register_task
from fairseq.tasks import FairseqTask, register_task

from fairseq.vis_unalign_sequence_generator import (
    VisUnAlignSequenceGenerator,
    VisUnAlignSequenceGeneratorWithAlignment
)

from fairseq.data.vis_unalign_indexed_dataset import (
    VisUnAlignIndexedRawTextDataset,
    make_dataset,
    dataset_exists
)

from fairseq.data.vis_unalign_language_pair_dataset import (
    VisUnAlignLanguagePairDataset
)

from torch.utils.tensorboard import SummaryWriter

import logging
LOG = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path, split,
    src, src_dict,
    tgt, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions,
    args=None,
    prepend_bos=False, load_alignments=False,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(
            data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
        return dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []

    for k in itertools.count():

        split_k = split + (str(k) if k > 0 else '')

        LOG.info('split %s split_k %s %s', split, split_k, k)

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

        LOG.info('prefix %s src %s', prefix, src)
        img_dataset = VisUnAlignIndexedRawTextDataset(
            split=split,
            path=prefix + src,
            dictionary=src_dict,
            image_pretrain_path=args.image_pretrain_path,
            append_eos=True,
            reverse_order=False,
            font_size=args.image_font_size,
            surf_width=args.image_surface_width,
            surf_height=args.image_surface_height,
            start_x=args.image_start_x,
            start_y=args.image_start_y,
            dpi=args.image_dpi,
            pad_size=args.image_pad_size,
            font_file=args.image_font_path,
            image_height=args.image_height,
            augment=args.image_augment
        )

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

    return VisUnAlignLanguagePairDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset.sizes, tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
        align_dataset=align_dataset,
    )


@register_task('visunaligntranslation')
class VisUnAlignTranslationTask(FairseqTask):
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
        # fmt: on

        parser.add_argument("--image-verbose", action='store_true',
                            help='Display verbose debug')

        parser.add_argument("--image-checkpoint-path", type=str,
                            default=None, help="Image checkpoint path")
        parser.add_argument("--image-pretrain-path", type=str, default=None,
                            help='Load pretrain sentence embeddings')
        parser.add_argument('--image-samples-path', default=None, type=str,
                            help='Image Samples path')

        parser.add_argument("--image-embed-type", default='avg', type=str,
                            help='Image embed type [add, avg, concat, visonly, tokonly]')
        parser.add_argument('--image-embed-dim', default=512, type=int,
                            help='Image embed dim')
        parser.add_argument('--image-embedding-normalize',  action='store_true',
                            help='Image embedding l2 normalize')

        parser.add_argument('--image-pretrain-eval-only', action='store_true',
                            help='OCR pretrain model in eval only mode')

        parser.add_argument('--score-disable-attn', action='store_true',
                            help='disable attn')

        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')

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

        parser.add_argument("--ocr-fract-width-perc", type=float,
                            default=0.30, help="OCR fractional max pool percentage for width")

        parser.add_argument("--decoder-lstm-layers", type=int,
                            default=3, help="Number of LSTM layers in model")
        parser.add_argument("--decoder-lstm-units", type=int,
                            default=256, help="Number of LSTM hidden units ")  # 640 256
        parser.add_argument("--decoder-lstm-dropout", type=float,
                            default=0.50, help="Number of LSTM layers in model")

        parser.add_argument('--image-font-path', type=str,
                            default='', help='Input font file')
        parser.add_argument("--image-font-size", type=int,
                            default=16, help="Font size")
        parser.add_argument("--image-height", type=int,
                            default=32, help="Image height")
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

        parser.add_argument("--tensorboard-path", type=str,
                            default='', help="Path to tensorboard output")

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.tensorboard_writer = SummaryWriter(
            log_dir=self.args.tensorboard_path)

        if args.image_verbose:
            logging.basicConfig(
                format='%(levelname)s: %(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(
                format='%(levelname)s: %(message)s', level=logging.INFO)

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

        return cls(args, src_dict, tgt_dict)

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

        self.datasets[split] = load_langpair_dataset(
            data_path, split, src, self.src_dict, tgt, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            args=self.args,
            load_alignments=self.args.load_alignments,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return VisUnAlignLanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def build_generator(self, args):
        if getattr(args, 'score_reference', False):
            from fairseq.sequence_scorer import SequenceScorer
            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.vis_unalign_sequence_generator import VisUnAlignSequenceGenerator, VisUnAlignSequenceGeneratorWithAlignment
            if getattr(args, 'print_alignment', False):
                seq_gen_cls = VisUnAlignSequenceGeneratorWithAlignment
            else:
                seq_gen_cls = VisUnAlignSequenceGenerator
            return seq_gen_cls(
                args.score_disable_attn,
                self.target_dictionary,
                beam_size=getattr(args, 'beam', 5),
                max_len_a=getattr(args, 'max_len_a', 0),
                max_len_b=getattr(args, 'max_len_b', 200),
                min_len=getattr(args, 'min_len', 1),
                normalize_scores=(not getattr(args, 'unnormalized', False)),
                len_penalty=getattr(args, 'lenpen', 1),
                unk_penalty=getattr(args, 'unkpen', 0),
                sampling=getattr(args, 'sampling', False),
                sampling_topk=getattr(args, 'sampling_topk', -1),
                sampling_topp=getattr(args, 'sampling_topp', -1.0),
                temperature=getattr(args, 'temperature', 1.),
                diverse_beam_groups=getattr(args, 'diverse_beam_groups', -1),
                diverse_beam_strength=getattr(
                    args, 'diverse_beam_strength', 0.5),
                match_source_len=getattr(args, 'match_source_len', False),
                no_repeat_ngram_size=getattr(args, 'no_repeat_ngram_size', 0),
            )

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
