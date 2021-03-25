# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import io
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.language_pair_dataset import LanguagePairDataset

import cv2

logger = logging.getLogger(__name__)


def _collate_frames(
    frames: List[torch.Tensor],
) -> torch.Tensor:
    """
    Convert a list of 2D frames into a padded 3D tensor
    Args:
        frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
            length of i-th frame and f_dim is static dimension of features
    Returns:
        3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
    """
    batch_size = len(frames)
    max_len = max(frame.size(0) for frame in frames)

    # for 3 channels: image_dims = [channels x height x width]
    image_dims = frames[0].size()[1:]
    out = frames[0].new_zeros(batch_size, max_len, *image_dims)
    for i, v in enumerate(frames):
        out[i, : v.size(0)] = v
    return out


class VisualTextDataset(LanguagePairDataset):
    """
    A class containing source texts, source images, and target IDs.
    The use of source texts is required because we don't have a source
    side dictionary.
    """

    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
            self,
            src_images,
            src_sizes,
            src_texts: Optional[List[str]] = None,
            tgt_texts: Optional[List[str]] = None,
            tgt_sizes=None,
            tgt_dict: Optional[Dictionary] = None,
            constraints=None,
            shuffle=False,
    ):
        super().__init__(
            src=src_images,
            src_sizes=src_sizes,
            src_dict=tgt_dict, # actually, unused
            tgt=tgt_texts,
            tgt_sizes=tgt_sizes,
            tgt_dict=tgt_dict,
            constraints=constraints,
            eos=tgt_dict.eos_index,
            shuffle=shuffle,
        )
        self.src_text = src_texts

        # TODO: make sure lengths are all valid

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:

        example = super().__getitem__(index)
        example["source_text"] = self.src_text[index]
        return example #  index, source, source_image, target

    def collater(self, samples) -> Dict:
        """Merge a list of samples to form a mini-batch."""

        if len(samples) == 0:
            return {}
        indices = torch.tensor([i["id"] for i in samples], dtype=torch.long)
        frames = _collate_frames(
            [s["source"] for s in samples],
        )
        # sort samples by descending number of frames
        n_frames = torch.tensor([s.size(0) for s in frames], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

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
                "src_images": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
        }
        return out

    @classmethod
    def from_text_path(
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
    ) -> "VisualTextDataset":
        """
        Builds a VisualTextDataset from an input path and some command-line parameters.
        """

        samples = []

        source_path = os.path.join(root, f"{split}.{args.source_lang}")
        target_path = os.path.join(root, f"{split}.{args.target_lang}")

        source_texts, source_images, source_sizes = [], [], []
        targets, target_sizes = [], []

        if args.image_samples_path is not None and not os.path.exists(args.image_samples_path):
            logger.info(f"Creating {args.image_samples_path}")
            os.makedirs(args.image_samples_path)

        for sampleno, (source, target) in enumerate(zip(open(source_path, "rt"), open(target_path, "rt")), 1):
            source = source.strip()
            target = target.strip()

            source_texts.append(source)
            if not args.image_cache_path:
                image_tensor = image_generator.get_tensor(source)
                source_images.append(image_tensor)
                source_sizes.append(image_tensor.shape[0])

            target_tokens = tgt_dict.encode_line(
                target, add_if_not_exist=False,
                append_eos=True, reverse_order=False,
            ).long()
            targets.append(target_tokens)
            target_sizes.append(len(target_tokens))

            if args.image_samples_path and sampleno % args.image_samples_interval == 0:
                imagepath = os.path.join(args.image_samples_path, f"{split}.{sampleno}.png")
                whole_image, image_pieces = image_generator.get_images(source)
                cv2.imwrite(imagepath, whole_image)

                for i, image in enumerate(image_pieces, 1):
                    imagepath = os.path.join(args.image_samples_path, f"{split}.{sampleno}.{i}.png")
                    logger.info(f"Saving sample #{sampleno} to {imagepath}")
                    cv2.imwrite(imagepath, image)

            # logger.info('gen sent images w font %s, size %s',
            #          self.font_file, self.font_size)

        source_sizes = np.array(source_sizes)
        target_sizes = np.array(target_sizes)

        shuffle = args.shuffle if is_train_split else False
        dataset = VisualTextDataset(source_images, source_sizes, source_texts,
                                    targets, target_sizes,
                                    tgt_dict=tgt_dict,
                                    constraints=None,
                                    shuffle=shuffle)

        return dataset
