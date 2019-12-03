# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(
            model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(
            0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        #print('LabelSmoothedCrossEntropyCriterion loss', loss)
        return loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        #print('agg loss', agg['loss'])
        return agg


@register_criterion('visual_label_smoothed_cross_entropy')
class VisualLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.image_disable = args.image_disable
        self.image_verbose = args.image_verbose
        self.image_src_loss_scale = args.image_src_loss_scale
        self.image_tgt_loss_scale = args.image_tgt_loss_scale

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if self.image_verbose:
            print('CRITERION: model forward')
        net_output, encoder_prelogits = model(**sample['net_input'])
        if self.image_verbose:
            print('CRITERION: model forward end')

        tgt_loss, tgt_nll_loss, src_loss, src_nll_loss = self.compute_loss(
            model, net_output, encoder_prelogits, sample, reduce=reduce)
        sample_size = sample['target'].size(
            0) if self.args.sentence_avg else sample['ntokens']

        if not self.image_disable:
            src_images = sample['net_input']['src_images']
            src_images_view = src_images.view(-1, src_images.size(-3),
                                              src_images.size(-2), src_images.size(-1))
            loss = (tgt_loss * self.image_tgt_loss_scale) + \
                (src_loss * self.image_src_loss_scale)
            nll_loss = (tgt_nll_loss * self.image_tgt_loss_scale) + \
                (src_nll_loss * self.image_src_loss_scale)

            src_image_count = src_images_view.size(0)
        else:
            loss = tgt_loss
            nll_loss = tgt_nll_loss
            src_image_count = 0

        tgt_loss = utils.item(tgt_loss.data) if reduce else tgt_loss.data
        tgt_nll_loss = utils.item(
            tgt_nll_loss.data) if reduce else tgt_nll_loss.data

        if not self.image_disable:
            src_loss = utils.item(src_loss.data) if reduce else src_loss.data
            src_nll_loss = utils.item(
                src_nll_loss.data) if reduce else src_nll_loss.data
        else:
            src_loss = None
            src_nll_loss = None

        #print('input images', src_images_view.shape)
        logging_output = {
            'loss': loss,
            'nll_loss': nll_loss,

            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,

            'tgt_loss': tgt_loss,
            'tgt_nll_loss': tgt_nll_loss,
            'tgt_loss_scale': self.image_tgt_loss_scale,

            'src_loss': src_loss,
            'src_nll_loss': src_nll_loss,
            'src_loss_scale': self.image_src_loss_scale,
            'src_ntokens': src_image_count,
        }
        if self.image_verbose:
            print(logging_output)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, encoder_prelogits, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        if not self.image_disable:
            src_lprobs = model.get_src_normalized_probs(
                encoder_prelogits, log_probs=True)
            src_target = model.get_src_targets(sample, net_output).view(-1, 1)
            src_loss, src_nll_loss = label_smoothed_nll_loss(
                src_lprobs, src_target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
            )
            if self.image_verbose:
                print('CRITERION: source loss %s, nll_loss %s' %
                      (src_loss, src_nll_loss))
                print('CRITERION: compute_loss source, logits %s, labels %s' %
                      (src_lprobs.shape, src_target.shape))
        else:
            src_loss = None
            src_nll_loss = None

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        if self.image_verbose:
            print('CRITERION: compute_loss decoder, logits %s, labels %s' %
                  (lprobs.shape, target.shape))
            print('CRITERION: decoder loss %s, nll_loss %s' %
                  (loss, nll_loss))

        return loss, nll_loss, src_loss, src_nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        # if self.image_verbose:
        # print('AGG LOSS: %s' % (logging_outputs))

        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return agg
