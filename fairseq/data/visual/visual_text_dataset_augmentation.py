# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import cv2
import itertools
import io
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from functools import lru_cache

import numpy as np
import torch
import random

from fairseq.data.indexed_dataset import dataset_exists as indexed_dataset_exists

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.language_pair_dataset import LanguagePairDataset


logger = logging.getLogger(__name__)


DEFAULT_FONT_SIZE = 8
DEFAULT_FONT_PATH = "."
DEFAULT_IMAGE_STRIDE = 30
DEFAULT_IMAGE_STRIDE_OVERLAP = 10


def _collate_slices(
    slices: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a list of 2D slices into a padded 3D tensor
    Args:
        slices (list): list of 2D slices of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(slices)*len_max*f_dim where len_max is max of L[i]
    """
    batch_size = len(slices)
    max_len = max(slice.size(0) for slice in slices)

    # for 3 channels: image_dims = [channels x height x width]
    image_dims = slices[0].size()[1:]
    out = slices[0].new_zeros(batch_size, max_len, *image_dims)
    for i, v in enumerate(slices):
        out[i, : v.size(0)] = v
    return out


class VisualTextAugDataset(LanguagePairDataset):
    """
    A class containing source texts, source images, and target IDs.
    The use of source texts is required because we don't have a source
    side dictionary.
    """

    LANG_TAG_TEMPLATE = "<lang:{}>"
    image_generator = None

    def __init__(
            self,
            src_images,
            src_sizes,
            slice_height,
            slice_width,
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            tgt_sizes=None,
            tgt_dict: Optional[Dictionary] = None,
            constraints=None,
            shuffle=False,
    ):
        """
        src_images is a list of 1d tensors which will be reshaped later
        into shape (num_slices x channels x height x width)
        """
        super().__init__(
            src=src_images,
            src_sizes=src_sizes,
            src_dict=tgt_dict, # unused!
            tgt=tgt_texts,
            tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict,
            constraints=constraints,
            eos=tgt_dict.eos_index,
            shuffle=shuffle,
        )
        self.src_text = src_texts
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.slice_area = slice_width * slice_height if slice_width and slice_height else 1

        # TODO: make sure lengths are all valid


    @property
    def can_reuse_epoch_itr_across_epochs(self):
        """
        Whether we can reuse the :class:`fairseq.data.EpochBatchIterator` for
        this dataset across epochs.
        This needs to return ``False`` if the sample sizes can change across
        epochs, in which case we may need to regenerate batches at each epoch.
        If your dataset relies in ``set_epoch`` then you should consider setting
        this to ``False``.
        """
        return False


    @lru_cache(maxsize=8)
    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns a single example in the form of a dictionary.
        The image tensor is 1d, and has to be scaled to the
        right shape (slices, channels, height, width).
        """

        example = super().__getitem__(index)
        if self.src_text is not None:
            example["source_text"] = self.src_text[index]

        assert VisualTextAugDataset.image_generator is not None, "UNEXPECTED: RAW DATA BUT CLASS IMAGE_GENERATOR IS NONE"
        if VisualTextAugDataset.image_generator is not None:
            ## in-progress: first, font_size regularization
            font_size_choice = random.choice([4,8,12,16])

            # update image generator
            VisualTextAugDataset.image_generator.font_size=font_size_choice
            VisualTextAugDataset.image_generator.load_font()

            # not necessary to update source_images, straight to tensors (1d pixel stream, flattened image_tensor)
            image_tensor = VisualTextAugDataset.image_generator.get_tensors(self.src_text[index])
            self.src_sizes[index] = image_tensor.shape[0]
#            print(image_tensor.shape)

            # flatten and resize to follow orig code (overkill)
            flattened = image_tensor.view(-1)
            num_slices = flattened.shape[0] // self.slice_area

            example["source"] = flattened.view(num_slices, 1, self.slice_height, self.slice_width)
#            print(example["source"].shape)
        else:
            # print("GETITEM:", example["source"].shape, self.slice_area)
            # Binarization writes this to a 1d pixel stream, so reshape (harmless for raw text)
            if self.slice_area:
                num_slices = example["source"].shape[0] // self.slice_area
                example["source"] = example["source"].view(num_slices, 1, self.slice_height, self.slice_width)

                # print(" -> RESHAPED:", example["source"].shape)
                # print(" -> DATA:")
                # for t in example["source"]:
                #     print("     ->", t)

        return example #  { id, source, target, source_text }

    def collater(self, samples) -> Dict:
        """Merge a list of samples to form a mini-batch."""

        # print("visual_text_dataset::collater: SAMPLES", len(samples), samples[0])

        if len(samples) == 0:
            return {}
        indices = torch.tensor([i["id"] for i in samples], dtype=torch.long)
        slices = _collate_slices(
            [s["source"] for s in samples],
        )
        # sort samples by descending number of slices
        n_slices = torch.tensor([s.size(0) for s in slices], dtype=torch.long)
        n_slices, order = n_slices.sort(descending=True)
        indices = indices.index_select(0, order)
        slices = slices.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if samples[0]["target"] is not None:
            target = fairseq_data_utils.collate_tokens(
                [t["target"] for t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [t.size(0) for t in target], dtype=torch.long
            )
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [t["target"] for t in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(s["target"].size(0) for s in samples)

        out = {
            "id": indices,
            "net_input": {
                "src_tokens": slices,
                "src_lengths": n_slices,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    @classmethod
    def from_raw(
            cls,
            root: str,
            args,
            split: str,
            source,
            target,
            image_generator,
            tgt_dict,
            is_train_split: bool,
            epoch: int,
            seed: int,
    ) -> "VisualTextAugDataset":
        """
        Builds a VisualTextAugDataset from an input path and some command-line parameters.

        There are two ways to build: from plain text, or from a preprocessed dataset.
        """

        samples = []

        source_path = os.path.join(root, f"{split}.{args.source_lang}-{args.target_lang}.{args.source_lang}")
        target_path = os.path.join(root, f"{split}.{args.source_lang}-{args.target_lang}.{args.target_lang}")

        source_texts, source_images, source_sizes = [], [], []
        targets, target_sizes = [], []

        slice_height = image_generator.height
        slice_width = image_generator.width

        total_source_len = 0
        for sampleno, (source, target) in enumerate(zip(open(source_path, "rt"), open(target_path, "rt")), 1):
            source = source.strip()
            target = target.strip()

            source_texts.append(source)
            image_tensor = image_generator.get_tensors(source)
            # VisualTextAugDataset expects the tensor to be flattened
            source_images.append(image_tensor.view(-1))
            source_sizes.append(image_tensor.shape[0])
            total_source_len += image_tensor.shape[0]

            target_tokens = tgt_dict.encode_line(
                target, add_if_not_exist=False,
                append_eos=True, reverse_order=False,
            ).long()
            targets.append(target_tokens)
            target_sizes.append(len(target_tokens))

            if args.image_samples_interval > 0 and sampleno % args.image_samples_interval == 0:
                image_generator.dump(source, os.path.join(args.image_samples_path, f"{split}.{sampleno}"))

        logger.info(f"Read {sampleno} samples for split {split}, mean length {total_source_len / sampleno:.1f}")

        source_sizes = np.array(source_sizes)
        target_sizes = np.array(target_sizes)

        shuffle = args.shuffle if is_train_split else False
        dataset = VisualTextAugDataset(source_images, source_sizes,
                                    slice_height, slice_width,
                                    src_texts=source_texts,
                                    tgt_texts=targets,
                                    tgt_sizes=target_sizes,
                                    tgt_dict=tgt_dict,
                                    constraints=None,
                                    shuffle=shuffle)

        return dataset

    @classmethod
    def from_text(
            cls,
            args,
            source_texts,
            image_generator,
            tgt_dict,
            constraints,
    ) -> "VisualTextAugDataset":
        """
        Used at inference time to create a dataset from STDIN.
        """
        source_images = []
        source_sizes = []
        for lineno, source in enumerate(source_texts, 1):
            image_tensor = image_generator.get_tensors(source)
            # VisualTextAugDataset expects the tensor to be flattened
            source_images.append(image_tensor.view(-1))
            source_sizes.append(image_tensor.shape[0])

            if args.image_samples_interval > 0 and lineno % args.image_samples_interval == 0:
                image_generator.dump(source, os.path.join(args.image_samples_path, f"{split}.{sampleno}"))

        return VisualTextAugDataset(source_images, source_sizes,
                                 image_generator.height,
                                 image_generator.width,
                                 src_texts=source_texts,
                                 tgt_dict=tgt_dict,
                                 constraints=constraints)

    def from_indexed(
            data_path,
            split,
            src,
            trg,
            trg_dict,
            dataset_impl,
            image_window,
            image_height,
            truncate_source=False,
            shuffle=True,
            combine=True,
    ):
        def split_exists(split, src, trg, lang, data_path):
            filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, trg, lang))
            return indexed_dataset_exists(filename, impl=dataset_impl)

        src_datasets = []
        trg_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")
            split_k_vis = split + ".vis" + (str(k) if k > 0 else "")

            # infer langcode
            if split_exists(split_k_vis, src, trg, src, data_path):
                image_prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k_vis, src, trg))
                target_prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, trg))
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, os.path.join(data_path, "{}.{}-{}.".format(split_k_vis, src, trg)))
                    )

            # This is important. We use standard fairseq utils to load the dataset, but each
            # entry has collapsed the (slice, pixel width, pixel height) entries into a sequence
            # of pixels, which we need to map back to their original sizes.
            src_dataset = fairseq_data_utils.load_indexed_dataset(
                image_prefix + src, None, dataset_impl
            )
            # print("LOADED: first five examples")
            # for i in range(5):
            #     print(f"  ->", len(src_dataset[i]), src_dataset[i].shape, src_dataset[i])

            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)

            # TODO: you could load the source text dataset here, if you needed access to it.
            # I'm not going to waste our time.

            trg_dataset = fairseq_data_utils.load_indexed_dataset(
                target_prefix + trg, trg_dict, dataset_impl
            )
            if trg_dataset is not None:
                trg_datasets.append(trg_dataset)

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, trg, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        assert len(src_datasets) == len(trg_datasets) or len(trg_datasets) == 0

        if len(src_datasets) == 1:
            src_dataset = src_datasets[0]
            trg_dataset = trg_datasets[0] if len(trg_datasets) > 0 else None
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            if len(trg_datasets) > 0:
                trg_dataset = ConcatDataset(trg_datasets, sample_ratios)
            else:
                trg_dataset = None

        trg_dataset_sizes = trg_dataset.sizes if trg_dataset is not None else None
        dataset = VisualTextAugDataset(src_dataset,
                                    src_dataset.sizes // (image_height * image_window),
                                    image_height, image_window,
                                    tgt_texts=trg_dataset,
                                    tgt_sizes=trg_dataset.sizes,
                                    tgt_dict=trg_dict,
                                    constraints=None,
                                    shuffle=shuffle)
        return dataset

    @classmethod
    def load(
            cls,
            root: str,
            args,
            split: str,
            source,
            target,
            image_generator,
            tgt_dict,
            is_train_split: bool,
            epoch: int,
            seed: int,
    ) -> "VisualTextAugDataset":
        """
        Loads plain text or indexed data, based on the requested dataset implementation.
        """
        if args.dataset_impl == "raw":
            return cls.from_raw(root, args, split, source, target,
                                image_generator, tgt_dict, is_train_split,
                                epoch, seed)

        elif args.dataset_impl == "mmap":
            return cls.from_indexed(data_path=root, split=split, src=source, trg=target,
                                    trg_dict=tgt_dict, dataset_impl=args.dataset_impl,
                                    image_window=image_generator.image_width,
                                    image_height=image_generator.image_height)

        else:
            raise Exception(f"No such dataset implementation {args.dataset_impl}")
