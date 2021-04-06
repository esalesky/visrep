"""
Defines configuration for the visual apparatus.
Can be used in pretraining and regular training.

NOTE: This class is currently not used! It's been copied from
the audio directory. It could be updated to replace VisualTextConfig
in visual_text_dataset.py
"""

from dataclasses import dataclass, field

from omegaconf import MISSING
from typing import Optional, Any

from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

@dataclass
class VisualConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    lang: str = field(
        default=MISSING,
        metadata={"help": "language (two-character code)"},
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to crop to for batching"}
    )

    image_height: int = field(
        default=32,
        metadata={"help": "image height"},
    )
    font_size: int = field(
        default=14,
        metadata={"help": "font size"},
    )

    # Options for reporting WER metrics during validation. Only applicable to
    # Seq2Seq models during fine-tuning
    eval_wer: bool = field(
        default=False, metadata={"help": "compute WER for Seq2Seq models"}
    )
    eval_wer_config: GenerationConfig = field(
        default_factory=lambda: GenerationConfig(),
        metadata={"help": "beam search config for evaluating wer during training"},
    )
    eval_wer_tokenizer: Any = field(
        default=None,
        metadata={"help": "tokenizer config for evaluating wer during training"},
    )
    eval_wer_post_process: str = field(
        default="letter",
        metadata={
            "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
        },
    )


