import itertools
import os

import json

import torchvision.transforms as transforms

from fairseq.data import image_transforms
from fairseq import options, utils
from fairseq.data import ConcatDataset, indexed_dataset, PrependTokenDataset, Dictionary

from fairseq.data import (
    data_utils,
    FairseqDataset,
    iterators,
    Dictionary,
)
from fairseq.data.ocr_iterators import OcrIterator

# from . import FairseqTask, register_task
from fairseq.tasks import FairseqTask, register_task


from fairseq.data.lmdb_dataset import OcrLmdbDataset

from fairseq.data.lmdb_pair_dataset import OcrLmdbPairDataset


from fairseq.data.datautils import OcrGroupedSampler


from torch.utils.tensorboard import SummaryWriter

import logging

LOG = logging.getLogger(__name__)


def load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    args=None,
    prepend_bos=False,
    load_alignments=False,
):

    if split == "train":
        train_transforms_list = []
        train_transforms_list.append(
            image_transforms.Scale(new_h=args.ocr_height))
        train_transforms_list.append(transforms.ToTensor())
        train_transforms = transforms.Compose(train_transforms_list)

        src_dataset = OcrLmdbDataset(
            split,
            args.data,
            src_dict,
            train_transforms,
            args.ocr_height,
            args.max_allowed_width,
        )
    else:
        valid_transforms_list = []
        valid_transforms_list.append(
            image_transforms.Scale(new_h=args.ocr_height))
        valid_transforms_list.append(transforms.ToTensor())
        valid_transforms = transforms.Compose(valid_transforms_list)

        src_dataset = OcrLmdbDataset(
            split,
            args.data,
            src_dict,
            valid_transforms,
            args.ocr_height,
            args.max_allowed_width,
        )

    return OcrLmdbPairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        None,
        None,
        None,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        max_source_positions=max_source_positions,
        max_target_positions=max_target_positions,
    )


@register_task("ocr")
class OcrTask(FairseqTask):
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

        parser.add_argument(
            "--image-verbose", action="store_true", help="Display verbose debug"
        )
        parser.add_argument(
            "--image-samples-path", default=None, type=str, help="Image Samples path"
        )
        parser.add_argument(
            "--tensorboard-path",
            type=str,
            default="",
            help="Path to tensorboard output",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.args = args
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

        self.tensorboard_writer = SummaryWriter(
            log_dir=self.args.tensorboard_path)

        if args.image_verbose:
            logging.basicConfig(
                format="%(levelname)s: %(message)s", level=logging.DEBUG
            )
        else:
            logging.basicConfig(
                format="%(levelname)s: %(message)s", level=logging.INFO)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """

        task_dict = Dictionary()

        with open(os.path.join(filename, "desc.json"), "r") as fh:
            data_desc = json.load(fh)

        for data_split in ["train", "validation", "test"]:
            for entry in data_desc[data_split]:
                for char in entry["trans"].split():
                    task_dict.add_symbol(char)

        return task_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)
        src_dict = cls.load_dictionary(args.data)
        tgt_dict = src_dict

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = self.args.data.split(":")
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            args=self.args,
            load_alignments=self.args.load_alignments,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return OcrLmdbDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def build_generator(self, args):
        if getattr(args, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(self.target_dictionary)
        else:
            from fairseq.vis_unalign_sequence_generator import (
                VisUnAlignSequenceGenerator,
                VisUnAlignSequenceGeneratorWithAlignment,
            )

            if getattr(args, "print_alignment", False):
                seq_gen_cls = VisUnAlignSequenceGeneratorWithAlignment
            else:
                seq_gen_cls = VisUnAlignSequenceGenerator
            return seq_gen_cls(
                args.score_disable_attn,
                self.target_dictionary,
                beam_size=getattr(args, "beam", 5),
                max_len_a=getattr(args, "max_len_a", 0),
                max_len_b=getattr(args, "max_len_b", 200),
                min_len=getattr(args, "min_len", 1),
                normalize_scores=(not getattr(args, "unnormalized", False)),
                len_penalty=getattr(args, "lenpen", 1),
                unk_penalty=getattr(args, "unkpen", 0),
                sampling=getattr(args, "sampling", False),
                sampling_topk=getattr(args, "sampling_topk", -1),
                sampling_topp=getattr(args, "sampling_topp", -1.0),
                temperature=getattr(args, "temperature", 1.0),
                diverse_beam_groups=getattr(args, "diverse_beam_groups", -1),
                diverse_beam_strength=getattr(
                    args, "diverse_beam_strength", 0.5),
                match_source_len=getattr(args, "match_source_len", False),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices,
                dataset,
                max_positions,
                raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        # batch_sampler = data_utils.batch_by_size(
        #    indices, dataset.num_tokens, max_tokens=max_tokens, max_sentences=max_sentences,
        #    required_batch_size_multiple=required_batch_size_multiple,
        # )

        # group_sampler = (GroupedSampler(dataset, rand=True),)

        # return a reusable, sharded iterator
        epoch_iter = OcrIterator(
            dataset=dataset,
            batch_size=self.args.max_sentences,
            collate_fn=dataset.collater,
            # group_sampler=group_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
