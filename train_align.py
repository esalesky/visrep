
"""
Train a new model on one or across multiple GPUs.
"""
import torchvision.transforms.functional as F
import logging
import os
import collections
import math
import random
import cv2
import numpy as np
import torch

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

fb_pathmgr_registerd = False

LOG = logging.getLogger(__name__)


def main(args, init_distributed=False):
    utils.import_user_module(args)

    try:
        from fairseq.fb_pathmgr import fb_pathmgr
        global fb_pathmgr_registerd
        if not fb_pathmgr_registerd:
            fb_pathmgr.register()
            fb_pathmgr_registerd = True
    except (ModuleNotFoundError, ImportError):
        pass

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    if distributed_utils.is_master(args):
        checkpoint_utils.verify_checkpoint_directory(args.save_dir)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print(model)
    print('| model {}, criterion {}'.format(
        args.arch, criterion.__class__.__name__))
    print('| num. model params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(
                args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(
                args, trainer, epoch_itr, valid_losses[0])

        reload_dataset = ':' in getattr(args, 'data', '')
        # sharded data: get train iterator for next epoch
        epoch_itr = trainer.get_train_iterator(
            epoch_itr.epoch, load_dataset=reload_dataset)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = args.update_freq[epoch_itr.epoch - 1] \
        if epoch_itr.epoch <= len(args.update_freq) else args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=(epoch_itr.epoch >= args.curriculum),
    )
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf

    if task.args.image_samples_path:
        samples_train_output = os.path.join(
            task.args.image_samples_path, 'train')
        if not os.path.exists(samples_train_output):
            os.makedirs(samples_train_output)

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        if args.image_verbose:
            LOG.debug('DATA: id %s' % (len(samples[0]['id'])))
            LOG.debug('DATA: nsentences %s', samples[0]['nsentences'])
            LOG.debug('DATA: ntokens %s', samples[0]['ntokens'])
            LOG.debug('DATA: src_tokens %s', samples[0]
                      ['net_input']['src_tokens'].shape)
            LOG.debug('DATA: src_lengths %s',
                      samples[0]['net_input']['src_lengths'].shape)
            if type(samples[0]['net_input']['src_images']) != type(None):
                LOG.debug('DATA: src_images %s', samples[0]
                          ['net_input']['src_images'].shape)
            LOG.debug('DATA: target %s', samples[0]['target'].shape)
            LOG.debug('DATA: prev_output_tokens %s',
                      samples[0]['net_input']['prev_output_tokens'].shape)

        log_output = trainer.train_step(samples)

        if i % args.image_display_mod == 0:
            # sample is an array of len update-freq
            for img_idx in range(len(samples)):
                tokens_list = samples[img_idx]['net_input']['src_tokens'].cpu(
                ).numpy()
                image_list = samples[img_idx]['net_input']['src_images'].cpu(
                ).numpy()

                # log is also is an array per update-freq
                logits = log_output['ocr'][img_idx]['logits']
                logits_view = logits.view(
                    tokens_list.shape[0], tokens_list.shape[1], logits.size(-1))

                # tokens_list has batch size
                for token_idx in range(tokens_list.shape[0]):
                    sent_list = []
                    for word_idx in tokens_list[token_idx]:
                        sent_list.append(task.src_dict[word_idx])
                        src_text = ''.join(sent_list)
                    LOG.info('RESULT OCR ref: %s', src_text)

                    curr_logit = logits_view[token_idx:token_idx+1].squeeze()
                    if len(curr_logit.shape) == 1:
                        curr_logit = curr_logit.unsqueeze(0)
                    _, curr_pred = curr_logit.topk(1, 1, True, True)
                    curr_pred = curr_pred.t()

                    asci_pred = []
                    for curr_pred_item in curr_pred[0]:
                        asci_pred.append(task.src_dict[curr_pred_item])
                    LOG.info('RESULT OCR hyp: %s', ''.join(asci_pred))

                    if task.args.image_samples_path:
                        for image_idx in range(image_list.shape[1]):
                            image = np.uint8(
                                image_list[token_idx][image_idx].transpose((1, 2, 0)) * 255)
                            # rgb to bgr
                            image = image[:, :, ::-1].copy()
                            curr_out_path = os.path.join(samples_train_output, str(i) + '_' + str(img_idx) + '_' +
                                                         str(token_idx) + '_' + str(image_idx) +
                                                         '_' + sent_list[image_idx] + '.png')
                            cv2.imwrite(curr_out_path, image)

                            if image_idx < 3:
                                task.tensorboard_writer.add_image(
                                    'images/train/{}_{}_{}_{}'.format(str(i), str(img_idx), str(token_idx), str(image_idx)), F.to_tensor(image), 0)

                    break  # display 1 and then break
                break

        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size', 'ocr']:
                continue  # these are already logged above
            if 'loss' in k or k == 'accuracy':
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        task.tensorboard_writer.add_scalar(
            'train/train_loss', stats['loss'].avg, (epoch_itr.epoch - 1) * len(epoch_itr) + epoch_itr.iterations_in_epoch)
        if 'total_loss' in stats:
            task.tensorboard_writer.add_scalar(
                'train/total_loss', stats['total_loss'], (epoch_itr.epoch - 1) * len(epoch_itr) + epoch_itr.iterations_in_epoch)
        if 'src_loss' in stats:
            task.tensorboard_writer.add_scalar(
                'train/source_loss', stats['src_loss'], (epoch_itr.epoch - 1) * len(epoch_itr) + epoch_itr.iterations_in_epoch)
        if 'tgt_loss' in stats:
            task.tensorboard_writer.add_scalar(
                'train/target_loss', stats['tgt_loss'], (epoch_itr.epoch - 1) * len(epoch_itr) + epoch_itr.iterations_in_epoch)

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(
                args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(
                args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()

    if trainer.args.image_enable_src_loss:
        stats['loss'] = trainer.get_meter('train_loss')
        if trainer.get_meter('train_nll_loss').count > 0:
            nll_loss = trainer.get_meter('train_nll_loss')
            stats['nll_loss'] = nll_loss
        else:
            nll_loss = trainer.get_meter('train_loss')
        stats['wps'] = trainer.get_meter('wps')
        stats['wpb'] = trainer.get_meter('wpb')
        stats['num_updates'] = trainer.get_num_updates()
        stats['lr'] = trainer.get_lr()

        stats['total_loss'] = trainer.get_meter('total_loss')
        stats['src_loss'] = trainer.get_meter('src_loss')
        stats['tgt_loss'] = trainer.get_meter('tgt_loss')
    else:
        stats['loss'] = trainer.get_meter('train_loss')
        if trainer.get_meter('train_nll_loss').count > 0:
            nll_loss = trainer.get_meter('train_nll_loss')
            stats['nll_loss'] = nll_loss
        else:
            nll_loss = trainer.get_meter('train_loss')
        stats['ppl'] = utils.get_perplexity(nll_loss.avg)
        stats['wps'] = trainer.get_meter('wps')
        stats['ups'] = trainer.get_meter('ups')
        stats['wpb'] = trainer.get_meter('wpb')
        stats['bsz'] = trainer.get_meter('bsz')
        stats['num_updates'] = trainer.get_num_updates()
        stats['lr'] = trainer.get_lr()

        stats['gnorm'] = trainer.get_meter('gnorm')
        stats['clip'] = trainer.get_meter('clip')
        stats['oom'] = trainer.get_meter('oom')
        if trainer.get_meter('loss_scale') is not None:
            stats['loss_scale'] = trainer.get_meter('loss_scale')
        stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
        stats['train_wall'] = trainer.get_meter('train_wall')

    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""

    if task.args.image_samples_path:
        samples_valid_output = os.path.join(
            task.args.image_samples_path, 'valid')
        if not os.path.exists(samples_valid_output):
            os.makedirs(samples_valid_output)

    if args.fixed_validation_seed is not None:
        # set fixed seed for every validation
        utils.set_torch_seed(args.fixed_validation_seed)

    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        valid_batch_ctr = 0

        correct_total_view = 0
        batch_size_total = 0
        image_total = 0

        for sample in progress:

            log_output = trainer.valid_step(sample)

            # sample {'id': ... 'nsentences': 184, 'ntokens': 1562, 'net_input':
            #
            # valid log {'loss': 7.187730188194559, 'nll_loss': 6.168642271884396,
            #   'ntokens': 1562, 'nsentences': 184, 'sample_size': 1562,
            #   'ocr': [{'input_shape': [184, 11, 1, 32, 32],
            #   'encoder_cnn_shape': [2024, 256, 8, 8],
            #   'embeddings': tensor([[5.8015, 0.0000, 0.0000,  ..., 2.8886, 0.0000, 0.0000],
            #   'logits':

            tokens_list = sample['net_input']['src_tokens'].cpu()  # .numpy()

            logits = log_output['ocr'][0]['logits']
            logits_view = logits.view(
                tokens_list.shape[0], tokens_list.shape[1], logits.size(-1)).cpu()

            targets = sample['target']

            batch_size = targets.size(0)  # of targets
            batch_size_total += batch_size  # len(sample)
            image_total += (int(sample['net_input']['src_tokens'].size(0))
                            * int(sample['net_input']['src_tokens'].size(1)))

            for token_idx in range(tokens_list.shape[0]):
                curr_logit = logits_view[token_idx:token_idx+1].squeeze()
                curr_target = tokens_list[token_idx].squeeze()

                if len(curr_logit.shape) == 1:
                    curr_logit = curr_logit.unsqueeze(0)

                _, curr_pred = curr_logit.topk(1, 1, True, True)
                curr_pred = curr_pred.t()

                correct_view = curr_pred.eq(
                    curr_target.view(1, -1).expand_as(curr_pred))
                correct_k_view = correct_view[:1].view(
                    -1).float().sum(0, keepdim=True)
                correct_total_view += int(correct_k_view)

            if valid_batch_ctr % 10 == 0:
                image_list = sample['net_input']['src_images'].cpu(
                ).numpy()

                # tokens_list has batch size
                for token_idx in range(tokens_list.shape[0]):
                    sent_list = []
                    for word_idx in tokens_list[token_idx]:
                        sent_list.append(task.src_dict[word_idx])
                    src_text = ''.join(sent_list)
                    LOG.info('RESULT OCR ref: %s', src_text)

                    curr_logit = logits_view[token_idx:token_idx+1].squeeze()
                    if len(curr_logit.shape) == 1:
                        curr_logit = curr_logit.unsqueeze(0)
                    _, curr_pred = curr_logit.topk(1, 1, True, True)
                    curr_pred = curr_pred.t()

                    asci_pred = []
                    for curr_pred_item in curr_pred[0]:
                        asci_pred.append(task.src_dict[curr_pred_item])
                    LOG.info('RESULT OCR hyp: %s', ''.join(asci_pred))

                    if task.args.image_samples_path:
                        for image_idx in range(image_list.shape[1]):
                            image = np.uint8(
                                image_list[token_idx][image_idx].transpose((1, 2, 0)) * 255)
                            # rgb to bgr
                            image = image[:, :, ::-1].copy()
                            curr_out_path = os.path.join(samples_valid_output, str(valid_batch_ctr) + '_' +
                                                         str(token_idx) + '_' + str(image_idx) +
                                                         '_' + sent_list[image_idx] + '.png')
                            cv2.imwrite(curr_out_path, image)

                            if image_idx < 3:
                                task.tensorboard_writer.add_image(
                                    'images/valid/{}_{}_{}_{}'.format(str(valid_batch_ctr), str(token_idx), str(image_idx), str(sent_list[image_idx])), F.to_tensor(image), 0)

                    break  # display 1 and then break

            valid_batch_ctr += 1

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size', 'ocr']:
                    continue
                extra_meters[k].update(v)

        LOG.info('Validation sub images %d, images %d, correct %d, accuracy %.2f',
                 image_total, batch_size_total, correct_total_view, (correct_total_view/image_total))

        # log validation stats
        stats = get_valid_stats(trainer, args, extra_meters)

        task.tensorboard_writer.add_scalar(
            'valid/accuracy', (correct_total_view/image_total), epoch_itr.epoch)
        task.tensorboard_writer.add_scalar(
            'valid/loss', stats['loss'].avg, epoch_itr.epoch)

        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(
            stats[args.best_checkpoint_metric].avg
            if args.best_checkpoint_metric == 'loss'
            else stats[args.best_checkpoint_metric]
        )
    return valid_losses


def get_valid_stats(trainer, args, extra_meters=None):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        key = 'best_{0}'.format(args.best_checkpoint_metric)
        best_function = max if args.maximize_best_checkpoint_metric else min

        current_metric = None
        if args.best_checkpoint_metric == 'loss':
            current_metric = stats['loss'].avg
        elif args.best_checkpoint_metric in extra_meters:
            current_metric = extra_meters[args.best_checkpoint_metric].avg
        elif args.best_checkpoint_metric in stats:
            current_metric = stats[args.best_checkpoint_metric]
        else:
            raise ValueError("best_checkpoint_metric not found in logs")

        stats[key] = best_function(
            checkpoint_utils.save_checkpoint.best,
            current_metric,
        )
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)


def cli_main():
    parser = options.get_training_parser()
    args = options.parse_args_and_arch(parser)

    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(
            port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


if __name__ == '__main__':
    cli_main()
