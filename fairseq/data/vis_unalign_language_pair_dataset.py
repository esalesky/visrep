# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

#from . import data_utils, FairseqDataset
from fairseq.data import data_utils, FairseqDataset


import logging
LOG = logging.getLogger(__name__)


def vis_align_collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    def check_alignment(alignment, src_len, tgt_len):
        if alignment is None or len(alignment) == 0:
            return False
        if alignment[:, 0].max().item() >= src_len - 1 or alignment[:, 1].max().item() >= tgt_len - 1:
            LOG.info("| alignment size mismatch found, skipping alignment!")
            return False
        return True

    def compute_alignment_weights(alignments):
        """
        Given a tensor of shape [:, 2] containing the source-target indices
        corresponding to the alignments, a weight vector containing the
        inverse frequency of each target index is computed.
        For e.g. if alignments = [[5, 7], [2, 3], [1, 3], [4, 2]], then
        a tensor containing [1., 0.5, 0.5, 1] should be returned (since target
        index 3 is repeated twice)
        """
        align_tgt = alignments[:, 1]
        _, align_tgt_i, align_tgt_c = torch.unique(
            align_tgt, return_inverse=True, return_counts=True)
        align_weights = align_tgt_c[align_tgt_i[np.arange(len(align_tgt))]]
        return 1. / align_weights.float()

    left_pad_source = False
    left_pad_target = False

    # sort by desc src tokens length
    samples = sorted(samples, key=lambda s: s['source'].numel(), reverse=True)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        tgt_lengths = torch.LongTensor(
            [s['target'].numel() for s in samples])
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens
    else:
        ntokens = sum(len(s['source']) for s in samples)

    src_embeddings_tensor = None
    if 'src_embedding' in samples[0]:
        embedding_shapes = [list(s['src_embedding'].shape)
                            for s in samples]
        src_embeddings_width_tensor = torch.LongTensor(
            [s['src_embedding'].shape[0] for s in samples])
        max_shape = np.max(embedding_shapes, axis=0)
        src_embeddings_tensor = torch.zeros(
            len(samples), max_shape[0], max_shape[1])  # (batch, tstep, dim)
        for sample_idx, sample in enumerate(samples):
            sample_shape = sample['src_embedding'].shape
            np_embedding = torch.from_numpy(sample['src_embedding'])
            src_embeddings_tensor[sample_idx,
                                  0:np_embedding.shape[0], :] = np_embedding

    image_id_list = [s['image_id'] for s in samples]
    utf8_ref_text_list = [s['utf8_ref_text'] for s in samples]
    image_list = [s['image'] for s in samples]

    #num_channels, word_height, word_width = image_list[0][0].shape
    # token_images = torch.ones(len(samples),
    #                          len(image_list[0]), num_channels, word_height, word_width)
    # for sample_idx, sample_image_list in enumerate(image_list):
    #    for word_idx, word_sample in enumerate(sample_image_list):
    #        width = word_sample.shape[2]  # 32, 32, 3
    #        token_images[sample_idx, word_idx, :, :, :width] = word_sample

    # unaligned, setnence with max tokens may not be the same as max image width
    max_width = max([s[0].size(2) for s in image_list])

    num_channels, word_height, word_width = image_list[0][0].shape

    token_images = torch.ones(len(samples),
                              num_channels, word_height, max_width)
    for sample_idx, sample_image in enumerate(image_list):
        # sample_image[0] since only 1 image in list with unaligned
        width = sample_image[0].shape[2]  # 32, 32, 3
        token_images[sample_idx, :, :, :width] = sample_image[0]

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'src_images': token_images,
        },
        'image_id_list': image_id_list,
        'utf8_ref_text_list': utf8_ref_text_list,
        'target': target,
    }

    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    return batch


class VisUnAlignLanguagePairDataset(FairseqDataset):
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
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
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
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"

    def __getitem__(self, index):
        tgt_item = self.tgt[index] if self.tgt is not None else None
        #src_item, src_embedding = self.src[index]

        src_metadata = self.src[index]
        src_item = src_metadata['src_item']

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat(
                    [self.tgt[index], torch.LongTensor([eos])])

        if self.remove_eos_from_source:
            eos = self.src_dict.eos()
            if self.src[index][-1] == eos:
                src_item = self.src[index][:-1]

        # 'embedding': src_embedding,
        example = {
            'id': index,
            'source': src_item,
            'target': tgt_item,
        }

        # 'src_item': self.tokens_list[i],
        # 'image_id': decode_metadata['image_id'],
        # 'utf8_ref_text': decode_metadata['utf8_ref_text'],
        # 'uxxxx_ref_text': decode_metadata['uxxxx_ref_text'],
        # 'image': decode_metadata['image'],

        if 'image_id' in src_metadata:
            example['image_id'] = src_metadata['image_id']

        if 'utf8_ref_text' in src_metadata:
            example['utf8_ref_text'] = src_metadata['utf8_ref_text']

        if 'uxxxx_ref_text' in src_metadata:
            example['uxxxx_ref_text'] = src_metadata['uxxxx_ref_text']

        if 'image' in src_metadata:
            example['image'] = src_metadata['image']

        if self.align_dataset is not None:
            example['alignment'] = self.align_dataset[index]

        return example

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
        return vis_align_collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(
                self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)
