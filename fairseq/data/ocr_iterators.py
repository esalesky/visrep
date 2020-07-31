import numpy as np
import torch

from fairseq import tokenizer
from fairseq.data import (
    data_utils,
    FairseqDataset,
    iterators,
    Dictionary,
)
from fairseq.data.iterators import EpochBatchIterator, CountingIterator

from fairseq.data.datautils import OcrGroupedSampler


class OcrIterator(EpochBatchIterator):
    def __init__(
        self,
        dataset,
        batch_size,
        collate_fn,
        # group_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
    ):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        # self.frozen_batches = tuple(batch_sampler)
        # self.group_sampler = group_sampler
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        self.num_workers = num_workers

        self.epoch = epoch
        self.shuffle = True
        self._cur_epoch_itr = None
        self._next_epoch_itr = None
        self._supports_prefetch = getattr(dataset, "supports_prefetch", False)

    def _get_iterator_for_epoch(
        self, epoch, shuffle, fix_batches_to_gpus=False, offset=0
    ):
        ocr_iter = CountingIterator(
            torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                sampler=OcrGroupedSampler(self.dataset, rand=True),
                collate_fn=self.collate_fn,
                pin_memory=True,
                drop_last=True,
            ),
            start=offset,
        )
        return ocr_iter
