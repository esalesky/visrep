# visrep

**This branch focuses on multilingual training and is under development. Use with caution.**

This repository is an extension of [fairseq](https://github.com/pytorch/fairseq) to enable training with visual text representations. 

For further information, please see:
- [Salesky et al. (2021): Robust Open-Vocabulary Translation from Visual Text Representations.](https://arxiv.org/abs/2104.08211)  
  In *Proceedings of EMNLP 2021*.


## Overview 

Our approach replaces the source embedding matrix with visual text representations, computed from rendered text with (optional) convolutions. 
This creates a 'continuous' vocabulary, in place of the fixed-size embedding matrix, which takes into account visual similarity, which together improve model robustness. 
There is no preprocessing before rendering text: on the source side, we directly render raw text, which we slice into overlapping, fixed-width image tokens. 

![Model diagram showing rendered text input at the sentence-level, which is sliced into overlapping, fixed-width image tokens, from which source representations for translation are computed via a convolutional block, before being passed to a traditional encoder-decoder model for translation.](https://user-images.githubusercontent.com/4117932/133522748-9fd1858d-c40f-4018-8bd7-b9e9c5f4e302.png)

Given typical parallel text, the data loader renders a complete source sentence according to a given font and size, and then creates strided patches or image tokens of a fixed square size. 

Because image tokens are generated in the data loader from provided text, to train and evaluate typical fairseq code remains largely unchanged. 


## Installation

The installation is the same as [fairseq](https://github.com/pytorch/fairseq), plus additional requirements specific to visual text.  
This branch has additional package requirements for pangocairo. 

**Requirements:**
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also want an NVIDIA GPU, and [NCCL](https://github.com/NVIDIA/nccl) (for multi-gpu training)

**To install and develop locally:**
``` bash
git clone https://github.com/esalesky/visrep
cd visrep
conda install -c conda-forge pycairo pygobject manimpango
pip install --editable ./
pip install -r examples/visual_text/requirements.txt
```


## Running the code

Training and evaluation can be called as with normal fairseq. 
For multilingual models, we use the `pixel_translation_multi_simple_epoch` task. 
Patch size is currently specified in `fairseq/data/visual/image_generator.py`. 
Pangocairo will attempt to use the specified font and if unsupported characters are present given a directory of fallback fonts it will fallback to supported fonts. 

```
--task pixel_translation_multi_simple_epoch
--arch visual_text_transformer 
--image-font-path {VALUE} (we have included the NotoSans fonts we used in this repo: see fairseq/data/visual/fonts/)
--image-font-size {VALUE}
--image-embed-normalize
--image-embed-type {VALUE} (options for number of convolutional blocks: e.g., direct, 1layer, 2layer, ..
```
Visual text parameters are serialized into saved models and do not need to be specified at inference time.  
Image samples can also optionally be written to the MODELDIR/samples/ subdirectory using `--image-samples-path` (directory to write to) and `--image-samples-interval` N (write every Nth image). 


### Binarization 
In addition to running on raw text and rendering on the fly, you may want to preprocess (binarize) data for larger experiments. This can be done as normal using fairseq `preprocess` but with the necessary visual text parameters, as below, and then passing `--dataset-impl mmap` instead of `--dataset-impl raw` during training. If you point to files with sentencepiece preprocessing applied, it will be removed on the source side before rendering (note: this only applies to sentencepiece). 
```
./preprocess.sh ${DATADIR} ${BINDIR-PREFIX} ${SRC} ${TGT}
```


## License

fairseq(-py) is MIT-licensed.

## Citation

Please cite as:

``` bibtex
@inproceedings{salesky-etal-2021-robust,
    title = "Robust Open-Vocabulary Translation from Visual Text Representations",
    author = "Salesky, Elizabeth  and
      Etter, David  and
      Post, Matt",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2104.08211",
}

@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
