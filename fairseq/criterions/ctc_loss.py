import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion("ctc_loss")
class CTCLoss(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.args = args

    def forward(self, model, sample, reduction="mean"):
        net_output = model(**sample["net_input"])
        loss = self.compute_loss(model, net_output, sample, reduction=reduction)
        sample_size = sample["target"].size(0)
        token_size = sample["target_length"].size(0)
        logging_output = {
            "loss": loss.item(),
            "ntokens": token_size,
            "nsentences": sample_size,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(
        self, model, net_output, sample, reduction="mean", zero_infinity=False,
    ):

        targets = sample["target"]
        target_lengths = sample["target_length"]

        logits = net_output["logits"]

        log_probs = F.log_softmax(logits, dim=2)

        batch_size = len(sample["id"])
        input_lengths = torch.full((batch_size,), logits.size(0), dtype=torch.int32)

        # CUDA, PyTorch native implementation: OK
        torch.backends.cudnn.enabled = False
        loss = F.ctc_loss(
            log_probs.to("cuda"),
            targets.to("cuda"),
            input_lengths,
            target_lengths,
            reduction="mean",
            zero_infinity=True,
        )

        return loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        assert len(logging_outputs) == 1
        log = logging_outputs[0]
        loss = log.get("loss", 0)
        ntokens = log.get("ntokens", 0)
        batch_sizes = log.get("nsentences", 0)
        sample_size = log.get("sample_size", 0)
        agg_output = {
            "loss": loss,
            "ntokens": ntokens,
            "nsentences": batch_sizes,
            "sample_size": sample_size,
        }
        return agg_output
