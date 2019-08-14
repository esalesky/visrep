import os
import torchvision.transforms as transforms

from fairseq.tasks import FairseqTask, register_task
from fairseq.data import (
    Dictionary, data_utils, iterators, FairseqDataset,
    OCRDataset, JSONDictionary, OCREpochBatchIterator, ImageAug
)


@register_task('ocr')
class OCRTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--max-positions', default=1024, type=int,
                            help='max input length')
        parser.add_argument('--height', type=int, default=32,
                            help='image height size used for training (default: 32)')
        parser.add_argument('--max-width', type=int, default=2500,
                            help='image height size used for training (default: 32)')
        parser.add_argument("--lmdb", type=str, 
                            help="specify the location to data.")
        
        parser.add_argument("--dictionary", type=str,
                            help="Dictionary path")
        parser.add_argument("--result_path", type=str,
                            help="Generate results output path")
        parser.add_argument('--eval', action='store_true', default=False, 
                            help='Eval mode for dataset')

        parser.add_argument("--augment", action='store_true',
                            default=False, help="Add image aug library")
           
    @classmethod
    def setup_task(cls, args, **kwargs):
        use_ctc_loss = True if args.criterion == 'ctc_loss' else False
        tgt_dict = cls.load_dictionary(args.dictionary)
        print('| target dictionary: {} types'.format(len(tgt_dict)))

        return cls(args, tgt_dict)

    def __init__(self, args, tgt_dict):
        super().__init__(args)
        self.tgt_dict = tgt_dict

    @classmethod
    def load_dictionary(cls, filename):
        dict_json = JSONDictionary.load(filename)
        return dict_json

    def load_dataset(self, split='test', **kwargs):
        valid_datasets = ('train', 'valid', 'test')
        if split not in valid_datasets:
            raise FileNotFoundError('Dataset not found: {} '.format(split))

        print('...load_dataset {}'.format(split))
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        aug_transform = transforms.Compose([
            ImageAug(),
            transforms.ToTensor(),
        ])
        if self.args.augment:
            transform = aug_transform

        shuffle = True if split == 'train' else False
        append_eos_to_target = False if self.args.criterion == 'ctc_loss' else True
        use_ctc_loss = True if self.args.criterion == 'ctc_loss' else False
        self.datasets[split] = OCRDataset(
            self.args.lmdb, self.args.height, self.tgt_dict, 
            shuffle=shuffle, transform=transform, use_ctc_loss=use_ctc_loss,
            input_feeding=True, append_eos_to_target=append_eos_to_target,
            max_allowed_width=self.args.max_width, split=split
        )
                
    def build_generator(self, args):
        print('Task: OCR build_generator')
        if args.criterion == 'ctc_loss':
            print('build_generator.ctc_loss')
            from ocr.ctc_loss_generator import CTCLossGenerator
            return CTCLossGenerator(self.target_dictionary)
        else:
            print('build_generator.SequenceGenerator')
            from fairseq.sequence_generator import SequenceGenerator
            return SequenceGenerator(
                self.target_dictionary,
                beam_size=args.beam,
                max_len_a=args.max_len_a,
                max_len_b=args.max_len_b,
                min_len=args.min_len,
                stop_early=(not args.no_early_stop),
                normalize_scores=(not args.unnormalized),
                len_penalty=args.lenpen,
                unk_penalty=args.unkpen,
                sampling=args.sampling,
                sampling_topk=args.sampling_topk,
                temperature=args.temperature,
                diverse_beam_groups=args.diverse_beam_groups,
                diverse_beam_strength=args.diverse_beam_strength,
                match_source_len=args.match_source_len,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

    def max_positions(self):
        """Return the max input length allowed by the task."""
        # The source should be less than *args.max_positions* and
        # In order to use `CuDNN`, the "target" has max length 256,
        return (self.args.max_positions, 256)

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=0
    ):

        return OCREpochBatchIterator(
            dataset=dataset,
            batch_size=max_sentences,
            collate_fn=dataset.collater,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            eval=self.args.eval
        )
        
