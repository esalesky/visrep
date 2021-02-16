from functools import lru_cache
import os
import shutil
import struct

import numpy as np
import torch
import re

from fairseq.data.datautils import utf8_to_uxxxx, uxxxx_to_utf8
import cv2

from fairseq.data import FairseqDataset

import json
import lmdb

import logging

LOG = logging.getLogger(__name__)


class OcrLmdbDataset(FairseqDataset):
    """Takes a text file as input and binarizes it in memory at instantiation.
    Original lines are also kept in memory"""

    def __init__(
        self, split, data_dir, dictionary, transforms, image_height, max_allowed_width,
    ):

        LOG.info("...OcrLmdbDataset %s", data_dir)

        self.data_dir = data_dir
        self.split = split
        self.dictionary = dictionary
        self.preprocess = transforms
        self.image_height = image_height
        self.max_allowed_width = max_allowed_width

        with open(os.path.join(self.data_dir, "desc.json"), "r") as fh:
            self.data_desc = json.load(fh)

        self.sizes = []
        for entry in self.data_desc[self.split]:
            self.sizes.append(len(entry["trans"].split()))
        self.sizes = np.array(self.sizes)

        self.lmdb_env = lmdb.Environment(
            os.path.join(self.data_dir, "line-images.lmdb"),
            map_size=1e6,
            readonly=True,
            lock=False,
        )
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

        self.size_group_limits = [150, 200, 300, 350, 450, 600, np.inf]

        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()

        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()

        for idx, entry in enumerate(self.data_desc[self.split]):
            width_orig, height_orig = entry["width"], entry["height"]

            normalized_width = width_orig * (self.image_height / height_orig)

            for cur_limit in self.size_group_limits:
                if (
                    normalized_width < cur_limit
                    and normalized_width < self.max_allowed_width
                ):
                    self.size_groups[cur_limit].append(idx)
                    self.size_groups_dict[cur_limit][idx] = 1
                    break

        # Now get final size (might have dropped large entries!)
        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])

            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max

        print("...finished loading, size {}".format(self.nentries))

        print("count by group")
        total_group_cnt = 0
        for cur_limit in self.size_group_limits:
            print("group", cur_limit, len(self.size_groups[cur_limit]))
            total_group_cnt += len(self.size_groups[cur_limit])
        print("TOTAL...", total_group_cnt)

    def __getitem__(self, index):

        entry = self.data_desc[self.split][index]
        max_width = 0
        for cur_limit in self.size_group_limits:
            if index in self.size_groups_dict[cur_limit]:
                max_width = cur_limit
                break

        group_id = max_width
        image_name = entry["id"]

        img_bytes = np.asarray(
            self.lmdb_txn.get(entry["id"].encode("ascii")), dtype=np.uint8
        )
        line_image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)  # -1)
        # Do a check for RGBA images; if found get rid of alpha channel
        if len(line_image.shape) == 3 and line_image.shape[2] == 4:
            line_image = cv2.cvtColor(line_image, cv2.COLOR_BGRA2BGR)

        line_image = self.preprocess(line_image)

        # Sanity check: make sure width@30px lh is long enough not to crash our model; we pad to at least 15px wide
        # Need to do this and change the "real" image size so that pack_padded doens't complain
        if line_image.size(2) < 15:
            line_image_ = torch.ones(
                line_image.size(0), line_image.size(1), 15)
            line_image_[:, :, : line_image.size(2)] = line_image
            line_image = line_image_

        # Add padding up to max-width, so that we have consistent size for cudnn.benchmark to work with
        original_width = line_image.size(2)
        original_height = line_image.size(1)

        transcription = []
        for char in entry["trans"].split():
            transcription.append(self.dictionary.index(char))

        src_metadata = {
            "target": transcription,
            # "target_len": len(transcription),
            "uxxx_trans": entry["trans"],
            "utf8_trans": uxxxx_to_utf8(entry["trans"]),
            "width": original_width,
            "height": original_height,
            "group": group_id,
            "image_name": image_name,
            "image": line_image,
            "id": index,
        }

        return src_metadata

    def __len__(self):
        return self.nentries
