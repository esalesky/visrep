# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

# from . import data_utils, FairseqDataset
from fairseq.data import data_utils, FairseqDataset


import logging

LOG = logging.getLogger(__name__)


def ocr_collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
    use_ctc_loss=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
        )

    samples = sorted(samples, key=lambda s: s["width"], reverse=True)

    image_list = [s["image"] for s in samples]
    image_height_list = [s["height"] for s in samples]
    image_width_list = [s["width"] for s in samples]
    image_group_list = [s["group"] for s in samples]
    image_name_list = [s["image_name"] for s in samples]
    uxxx_trans_list = [s["uxxx_trans"] for s in samples]
    utf8_trans_list = [s["utf8_trans"] for s in samples]
    # target_len_list = [s["target_len"] for s in samples]
    id_list = [s["id"] for s in samples]

    prev_output_tokens = None
    if use_ctc_loss:
        targets = [s["target"] for s in samples]
    else:
        targets = merge("target", left_pad=left_pad_source)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                "target", left_pad=left_pad_source, move_eos_to_beginning=True,
            )

    max_width = max(image_width_list)
    num_channels, image_height, image_width = image_list[0].shape
    images_tensor = torch.ones(len(samples), num_channels, image_height, max_width)
    images_width_tensor = torch.IntTensor(len(samples))
    for sample_idx, sample_image in enumerate(image_list):
        width = sample_image.shape[2]  # channel, height, width
        images_tensor[sample_idx, :, :, :width] = sample_image[0]
        images_width_tensor[sample_idx] = width

    target_transcription_widths = torch.IntTensor(len(samples))
    for idx, target_item in enumerate(targets):
        target_transcription_widths[idx] = len(target_item)

    target_transcription = torch.IntTensor(target_transcription_widths.sum().item())
    cur_offset = 0
    for idx, target_item in enumerate(targets):
        for j, char in enumerate(target_item):
            target_transcription[cur_offset] = char
            cur_offset += 1

    batch = {
        "id": id_list,
        "group": image_group_list,
        "height": image_height_list,
        "width": image_width_list,
        "image_name": image_name_list,
        "uxxx_trans": uxxx_trans_list,
        "utf8_trans": utf8_trans_list,
        "net_input": {
            "src_images": images_tensor,
            "src_images_width": images_width_tensor,
        },
        "target": target_transcription,
        "target_length": target_transcription_widths,
    }

    if prev_output_tokens is not None:
        batch["net_input"]["prev_output_tokens"] = prev_output_tokens

    return batch


class OcrLmdbPairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict,
        tgt=None,
        tgt_sizes=None,
        tgt_dict=None,
        left_pad_source=True,
        left_pad_target=False,
        max_source_positions=1024,
        max_target_positions=1024,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"

    def __getitem__(self, index):

        # src_metadata = {
        #     'transcription': transcription,
        #     'uxxx_trans': entry['trans'],
        #     'utf8_trans': uxxxx_to_utf8(entry['trans']),
        #     'width': original_width,
        #     'height': original_height,
        #     'group': group_id,
        #     'path': image_name,
        #     'id': index,
        # }

        src_metadata = self.src[index]

        return src_metadata

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return ocr_collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
