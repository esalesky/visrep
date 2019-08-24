# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from .dictionary import Dictionary, TruncatedDictionary
from .fairseq_dataset import FairseqDataset
from .backtranslation_dataset import BacktranslationDataset
from .concat_dataset import ConcatDataset
from .indexed_dataset import (
    IndexedCachedDataset, IndexedDataset, IndexedRawTextDataset,
)
from .indexed_image_dataset import (
    IndexedImageWordDataset, IndexedImageLineDataset,
    IndexedImageDataset, IndexedImageCachedDataset
)
from .language_pair_dataset import LanguagePairDataset
from .image_pair_dataset import ImagePairDataset
from .monolingual_dataset import MonolingualDataset
from .round_robin_zip_datasets import RoundRobinZipDatasets
from .token_block_dataset import TokenBlockDataset
from .transform_eos_dataset import TransformEosDataset
from .ocr_dataset import OCRDataset
from .json_dictionary import JSONDictionary
from .augment import ImageAug

from .iterators import (
    CountingIterator,
    EpochBatchIterator,
    GroupedIterator,
    ShardedIterator,
    OCREpochBatchIterator
)

__all__ = [
    'BacktranslationDataset',
    'ConcatDataset',
    'CountingIterator',
    'Dictionary',
    'EpochBatchIterator',
    'FairseqDataset',
    'GroupedIterator',
    'IndexedCachedDataset',
    'IndexedDataset',
    'IndexedRawTextDataset',
    'IndexedImageWordDataset',
    'IndexedImageDatset',
    'IndexedImageCachedDataset',
    'IndexedImageLineDataset',
    'LanguagePairDataset',
    'ImagePairDataset',
    'MonolingualDataset',
    'RoundRobinZipDatasets',
    'ShardedIterator',
    'TokenBlockDataset',
    'TransformEosDataset',
    'OCRDataset',
    'JSONDictionary',
    'OCREpochBatchIterator',
    'ImageAug',
]
