# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import utils

# from . import FairseqCriterion, register_criterion
from fairseq.criterions import FairseqCriterion, register_criterion

import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
LOG = logging.getLogger(__name__)


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


@register_criterion('vis_align_label_smoothed_cross_entropy')
class VisAlignLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args
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

        if self.args.image_enable_src_loss:
            tgt_loss, tgt_nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce)

            # src_loss, src_nll_loss = self.compute_src_loss(
            #    model, net_output, sample, reduce=reduce)

            src_loss = self.compute_src_loss(
                model, net_output, sample, reduce=reduce)

            loss = (tgt_loss * self.args.image_tgt_loss_scale) + \
                (src_loss * self.args.image_src_loss_scale)

            # nll_loss = (tgt_nll_loss * self.args.image_tgt_loss_scale) + \
            #    (src_nll_loss * self.args.image_src_loss_scale)

            nll_loss = tgt_nll_loss

            src_loss = utils.item(src_loss.data) if reduce else src_loss.data
            tgt_loss = utils.item(tgt_loss.data) if reduce else tgt_loss.data
            total_loss = utils.item(loss.data) if reduce else loss.data
            # total_nll_loss = utils.item(
            #    nll_loss.data) if reduce else nll_loss.data

            LOG.debug('src_loss %s', float(src_loss))
            #LOG.debug('src_nll_loss %s', float(src_nll_loss))
            LOG.debug('tgt_loss %s', float(tgt_loss))
            LOG.debug('tgt_nll_loss %s', float(tgt_nll_loss))
            LOG.debug('total_loss %s', float(total_loss))
            LOG.debug('image_tgt_loss_scale %s',
                      self.args.image_tgt_loss_scale)
            LOG.debug('image_src_loss_scale %s',
                      self.args.image_src_loss_scale)
            LOG.debug('loss %s', float(loss))

        else:
            loss, nll_loss = self.compute_loss(
                model, net_output, sample, reduce=reduce)
            src_loss = None
            src_nll_loss = None
            tgt_loss = None
            tgt_nll_loss = None
            total_loss = None
            total_nll_loss = None

        sample_size = sample['target'].size(
            0) if self.args.sentence_avg else sample['ntokens']

        LOG.debug('sample_size- ntokens %s', sample['ntokens'])

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
            'ocr': net_output[1]['ocr_out'],
        }

        if self.args.image_enable_src_loss:
            logging_output['total_loss'] = total_loss
            logging_output['tgt_loss'] = tgt_loss
            logging_output['tgt_loss_scale'] = self.args.image_tgt_loss_scale
            logging_output['src_loss'] = src_loss
            logging_output['src_loss_scale'] = self.args.image_src_loss_scale

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def compute_src_loss(self, model, net_output, sample, reduce=True):
        logits = net_output[1]['ocr_out']['logits']
        tokens_list = sample['net_input']['src_tokens']
        tokens_view = tokens_list.view(-1)

        # reduction='mean' or 'sum'
        loss = F.cross_entropy(logits, tokens_view, reduction='sum')

        # loss, nll_loss = label_smoothed_nll_loss(
        #    logits, tokens_view, self.eps, reduce=reduce,
        # )

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        ocr_logits = [log.get('ocr') for log in logging_outputs]

        agg = {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2) if sample_size > 0 else 0.,
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'ocr': ocr_logits,
        }

        if 'src_loss' in logging_outputs[0]:
            src_loss = sum(log.get('src_loss', 0) for log in logging_outputs)
            tgt_loss = sum(log.get('tgt_loss', 0) for log in logging_outputs)
            total_loss = sum(log.get('total_loss', 0)
                             for log in logging_outputs)

            agg['src_loss'] = src_loss
            agg['tgt_loss'] = tgt_loss
            agg['total_loss'] = total_loss

        return agg
