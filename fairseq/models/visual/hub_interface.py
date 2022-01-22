import argparse
import copy
import logging
import os
from typing import Any, Dict, Iterator, List

import torch
from fairseq import utils
from fairseq.hub_utils import GeneratorHubInterface
from omegaconf import open_dict
from torch import nn
import sentencepiece as spm


class VisrepHubInterface(GeneratorHubInterface):
    """
    PyTorch Hub interface for generating sequences from a pre-trained
    translation or language model.
    """

    def __init__(self, cfg, task, models, tgt_spm):
        super().__init__(cfg, task, models)
        self.cfg = cfg
        self.task = task
        self.models = nn.ModuleList(models)
        self.tgt_dict = task.target_dictionary

        # optimize model for generation
        for model in self.models:
            model.prepare_for_inference_(cfg)

        # load alignment dictionary for unknown word replacement (typically not used in visrep)
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(cfg.generation.replace_unk)

        # initializes the image_generator from the serialized parameters in task
        self.image_generator = task.image_generator
        self.bpe = spm.SentencePieceProcessor(model_file=tgt_spm)

        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in models]
        )

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    @property
    def device(self):
        return self._float_tensor.device

    def translate(
        self, sentences: List[str], target_lang: str ='en', beam: int = 5, verbose: bool = False, **kwargs
    ) -> List[str]:
        return self.sample(sentences, target_lang, beam, verbose, **kwargs)

    def sample(
        self, sentences: List[str], target_lang: str ='en', beam: int = 1, verbose: bool = False, **kwargs
    ) -> List[str]:
        if isinstance(sentences, str):
            return self.sample([sentences], target_lang=target_lang, beam=beam, verbose=verbose, **kwargs)[0]
        batched_hypos = self.generate(sentences, target_lang, beam, verbose, **kwargs)
        return [self.decode(hypos[0]["tokens"]) for hypos in batched_hypos]


    # note that unlike fairseq text models, the input 'sentences' is still the initial List[str]:
    #  images are generated when batches are built
    def generate(
        self,
        sentences: List[str],
        target_lang: str ='en', 
        beam: int = 5,
        verbose: bool = False,
        skip_invalid_size_inputs=False,
        inference_step_args=None,
        **kwargs
    ) -> List[List[Dict[str, torch.Tensor]]]:

        # build generator using current args as well as any kwargs
        gen_args = copy.deepcopy(self.cfg.generation)
        with open_dict(gen_args):
            gen_args.beam = beam
            for k, v in kwargs.items():
                setattr(gen_args, k, v)
        generator = self.task.build_generator(self.models, gen_args)

        inference_step_args = inference_step_args or {}
        results = []
        for batch in self._build_batches(sentences, skip_invalid_size_inputs):
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            translations = self.inference_step(
                generator, self.models, batch, self.task, target_lang, **inference_step_args
            )
            for id, hypos in zip(batch["id"].tolist(), translations):
                results.append((id, hypos))

        # sort output to match input order
        outputs = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
        return outputs

    # given a text sentence, render and generate tensors for image slices
    def encode(self, sentence: str) -> torch.LongTensor:
        return self.image_generator.get_tensors(sentence)

    # decode output text
    def decode(self, tokens: torch.LongTensor) -> str:
        sentence = self.string(tokens)
        sentence = self.remove_bpe(sentence)
        return sentence

    def remove_bpe(self, sentence: str) -> str:
        sentence = self.bpe.decode(sentence.split())
        return sentence

    def string(self, tokens: torch.LongTensor) -> str:
        return self.tgt_dict.string(tokens)

    # lengths is None because it is set in this step after rendering, when src_lengths is known
    def _build_batches(
        self, tokens: List[List[int]], skip_invalid_size_inputs: bool
    ) -> Iterator[Dict[str, Any]]:
        lengths = None
        batch_iterator = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens,lengths,constraints=None),
            max_tokens=self.cfg.dataset.max_tokens,
            max_sentences=self.cfg.dataset.batch_size,
            max_positions=self.max_positions,
            ignore_invalid_inputs=skip_invalid_size_inputs,
            disable_iterator_cache=True,
        ).next_epoch_itr(shuffle=False)
        return batch_iterator

    def inference_step(
        self, generator, models, sample, task, target_lang, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            bos_token = task.target_dictionary.eos()
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
            )
