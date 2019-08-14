
import numpy as np
import torch
from fairseq.data import data_utils, FairseqDataset
import json
import os
import lmdb
import cv2
import sys
#from ocr.data.data_utils import default_loader
from torch.utils.data.sampler import Sampler
from fairseq import utils

class OCRGroupedSampler(Sampler):
    """Dataset is divided into sub-groups, G_1, G_2, ..., G_k
       Samples Randomly in G_1, then moves on to sample randomly into G_2, etc all the way to G_k

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, rand=True, max_items=-1, fixed_rand=False):
        self.size_group_keys = data_source.size_group_keys
        self.size_groups = data_source.size_groups
        self.num_samples = len(data_source)
        self.rand = rand
        self.fixed_rand = fixed_rand
        self.max_items = max_items
        self.rand_perm = dict()

    def __iter__(self):
        n_items = 0
        for g in self.size_group_keys:
            if len(self.size_groups[g]) == 0:
                continue

            if self.fixed_rand:
                if g not in self.rand_perm:
                    self.rand_perm[g] = torch.randperm(len(self.size_groups[g])).long()
                g_idx_iter = iter(self.rand_perm[g])
            else:
                if self.rand:
                    g_idx_iter = iter(torch.randperm(len(self.size_groups[g])).long())
                else:
                    g_idx_iter = iter(range(len(self.size_groups[g])))

            while True:
                try:
                    g_idx = next(g_idx_iter)
                except StopIteration:
                    break

                n_items += 1
                if self.max_items > 0 and n_items > self.max_items:
                    raise StopIteration
                yield self.size_groups[g][g_idx]

        raise StopIteration

    def __len__(self):
        return self.num_samples
    

def collate(
    samples, pad_idx, eos_idx, left_pad=False,
    input_feeding=True, use_ctc_loss=False,
):
    #print('collate')

    """collate samples of images and targets."""
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    samples.sort(key=lambda d: d['width'], reverse=True)
    biggest_size = samples[0]['source'].size()
    smallest_size = samples[-1]['source'].size()
    
    #print('\n \t batch len {}, max {}, min {}'.format(len(samples), biggest_size, smallest_size))
    
    #images = torch.zeros(len(samples), biggest_size[0], biggest_size[1], 900) #biggest_size[2])
    images = torch.zeros(len(samples), biggest_size[0], biggest_size[1], biggest_size[2])
    for idx, img_sample in enumerate(samples):
        width = img_sample['source'].size(2)
        images[idx, :, :, :width] = img_sample['source']
        
    #images = torch.stack([s['source'] for s in samples])

    prev_output_tokens = None

    if use_ctc_loss:
        targets = [s['target'] for s in samples]
    else:
        targets = merge('target', left_pad=left_pad)
        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad,
                move_eos_to_beginning=True,
            )

    tgt_lengths = [s['target_length'] for s in samples]
    id = torch.LongTensor([s['id'] for s in samples])
    ntokens = sum(tgt_lengths)
    tgt_lengths = torch.IntTensor(tgt_lengths)
    names = [s['name'] for s in samples]

    # TODO: pin-memory
    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': images,
        },
        'target': targets,
        'target_length': tgt_lengths,
        'names': names
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def uxxxx_to_utf8(in_str):
    idx = 0
    result = ''
    if in_str.strip() == "":
        return ""

    for uxxxx in in_str.split():
        if uxxxx == '':
            continue

        if uxxxx == "<unk>" or uxxxx == "<s>" or uxxxx == "</s>":
            cur_utf8_char = uxxxx
        else:
            # First get the 'xxxx' part out of the current 'uxxxx' char

            cur_char = uxxxx[1:]

            # Now decode the hex code point into utf-8
            try:
                cur_utf8_char = chr(int(cur_char, 16))
            except:
                print("Exception converting cur_char = [%s]" % cur_char)
                sys.exit(1)

        # And add it to the result buffer
        result = result + cur_utf8_char

    return result

class OCRDataset(FairseqDataset):
    def __init__(
        self, data_dir, height, tgt_dict, 
        shuffle=True, transform=None, 
        use_ctc_loss=False, left_pad=False,
        input_feeding=True, append_eos_to_target=False, 
        max_allowed_width=900, min_allowed_width=25, split='train',
        max_source_positions=1024, max_target_positions=1024,
    ):
        self.tgt_dict = tgt_dict
        self.src_dict = tgt_dict
        self.height = height
        self.max_allowed_width = max_allowed_width
        self.min_allowed_width = min_allowed_width
        
        self.transform = transform
        self.shuffle = shuffle
        self.use_ctc_loss = use_ctc_loss
        self.left_pad = left_pad
        self.input_feeding = input_feeding
        self.append_eos_to_target = append_eos_to_target

        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions

        self.split = split
        if self.split.startswith('valid'):
            self.split='validation' # used in SCALE 2018 lmdb splits
        print('...OCRDataset loading json for split {}'.format(self.split))
        with open(os.path.join(data_dir, 'desc.json'), 'r') as fh:
            self.data_desc = json.load(fh)

        self.size_group_limits = [60, 150, 200, 300, 350, 450, 600, np.inf]
        self.size_group_keys = self.size_group_limits
        self.size_groups = dict()
        self.size_groups_dict = dict()
        for cur_limit in self.size_group_limits:
            self.size_groups[cur_limit] = []
            self.size_groups_dict[cur_limit] = dict()
                    
        self.src = []
        self.src_width = []
        self.src_height = []
        self.tgt = []
        self.tgt_sizes = []
        drop_cnt = 0
        
        keep_idx = 0
        for idx, entry in enumerate(self.data_desc[self.split]):
            
            image_uxxxx_trans = entry['trans']            
            image_uxxxx_trans_list = image_uxxxx_trans.split()

            image_width_orig, image_height_orig = entry['width'], entry['height']
            
            normalized_height = self.height
            normalized_width = int(image_width_orig * (normalized_height / image_height_orig))
            
            if normalized_width < self.max_allowed_width and normalized_width > self.min_allowed_width:
                self.src_height.append(normalized_height)
                self.src_width.append(normalized_width)
                self.tgt_sizes.append(len(image_uxxxx_trans_list))
                self.tgt.append(image_uxxxx_trans_list)
                self.src.append(entry['id'])

                for cur_limit in self.size_group_limits:
                    if normalized_width < cur_limit and normalized_width < self.max_allowed_width:
                        self.size_groups[cur_limit].append(keep_idx)
                        self.size_groups_dict[cur_limit][keep_idx] = 1
                        
                        keep_idx += 1
                        break
            else:
                drop_cnt += 1

        self.nentries = 0
        self.max_index = 0
        for cur_limit in self.size_group_limits:
            self.nentries += len(self.size_groups[cur_limit])

            if len(self.size_groups[cur_limit]) > 0:
                cur_max = max(self.size_groups[cur_limit])
                if cur_max > self.max_index:
                    self.max_index = cur_max                    
                    
                    
        print('setup lmdb {}, height {}, read {}, added {}, dropped {}'.format(self.split, self.height, idx + 1, len(self.src), drop_cnt))
        self.lmdb_env = lmdb.Environment(os.path.join(data_dir, 'line-images.lmdb'), map_size=1e6, readonly=True, lock=False)
        self.lmdb_txn = self.lmdb_env.begin(buffers=True)

    def __len__(self):
        return self.nentries

    def __getitem__(self, index):
        image_name = self.src[index]
        
        img_bytes = np.asarray(self.lmdb_txn.get(image_name.encode('ascii')), dtype=np.uint8)
        image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR) #-1)
        
        dim = (self.src_width[index], self.src_height[index])
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        
        if self.transform is not None:
            image = self.transform(image)

        original_width = image.size(2)
        
        tgt_item = self.tgt[index]
        tgt_length = self.tgt_sizes[index]
        tgt_item = torch.IntTensor([self.tgt_dict.index(i) for i in tgt_item])
        if self.append_eos_to_target:
            tgt_item = tgt_item.to(torch.int64)
            eos = self.tgt_dict.eos()
            if self.tgt and self.tgt[index][-1] != eos:
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])

        return {
            'id': index,
            'source': image,
            'width': original_width,
            'target': tgt_item,
            'target_length': tgt_length,
            'name' : image_name,
        }

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        pad_idx = None if self.use_ctc_loss is True else self.tgt_dict.pad()
        eos_idx = None if self.use_ctc_loss is True else self.tgt_dict.eos()
        left_pad = self.left_pad
        input_feeding = self.input_feeding

        return collate(
            samples, pad_idx=pad_idx, eos_idx=eos_idx, left_pad=left_pad,
            input_feeding=input_feeding, use_ctc_loss=self.use_ctc_loss,
        )

    def ordered_indices(self):
        """
        Return an ordered list of indices. Batches will be constructed based
        on this order.
        """
        if self.shuffle:
            return np.random.permutation(len(self))
        else:
            return np.arange(len(self))

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        #print('size', index)
        return self.tgt_sizes[index] if self.tgt_sizes is not None else 0

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        image = self.transform(np.zeros((32, 150, 3), np.uint8))
        index = 0
        tgt_item = self.tgt[index]
        tgt_length = self.tgt_sizes[index]
        tgt_item = torch.IntTensor([self.tgt_dict.index(i) for i in tgt_item])
        #print('eos?')
        if self.append_eos_to_target:
            tgt_item = tgt_item.to(torch.int64)
            eos = self.tgt_dict.eos()
            print('add eos')
            if self.tgt and self.tgt[index][-1] != eos:
                print(tgt_item)
                tgt_item = torch.cat([tgt_item, torch.LongTensor([eos])])
                print(tgt_item)
        image_name = self.src[index]
        original_width = image.size(2)
        bsz = 32 
        return self.collater([
            {
                'id': index,
                'source': image,
                'width': original_width,
                'target': tgt_item,
                'target_length': tgt_length,
                'name' : image_name,
            }
            for i in range(bsz)
        ])
