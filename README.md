# visrep

This repository is an extension of [fairseq](https://github.com/pytorch/fairseq) to enable training with visual text representations. 

For further information, please see:
- [Salesky et al. (2021): Robust Open-Vocabulary Translation from Visual Text Representations.](https://arxiv.org/abs/2104.08211)  
  In *Proceedings of EMNLP 2021*.

## Overview 

Our approach replaces the source embedding matrix with visual text representations, computed from rendered text with (optional) convolutions. 
This creates a 'continuous' vocabulary, in place of the fixed-size embedding matrix, which takes into account visual similarity, which together improve model robustness. 
There is no preprocessing before rendering text: on the source side, we directly render raw text, which we slice into overlapping, fixed-width image tokens. 

![Model diagram showing rendered text input at the sentence-level, which is sliced into overlapping, fixed-width image tokens, from which source representations for translation are computed via a convolutional block, before being passed to a traditional encoder-decoder model for translation.](https://user-images.githubusercontent.com/4117932/133522748-9fd1858d-c40f-4018-8bd7-b9e9c5f4e302.png)

Given typical parallel text, the data loader renders a complete source sentence and then creates strided slices according to the values of `--image-window` (width) and `--image-stride` (stride). 
Image height is determined automatically from the font size (`--font-size`), and slices are created using the full image height. 
This creates a set of image 'tokens' for each sentence, one per slice, with size 'window width' x 'image height.'

Because the image tokens are generated completely in the data loader, to train and evaluate typical fairseq code remains largely unchanged. 
Our VisualTextTransformer (enabled with `--task visual_text`) produces the source representations for training from the rendered text (one per image token). 
After that, everything proceeds as per normal fairseq.


## Installation

The installation is the same as [fairseq](https://github.com/pytorch/fairseq), plus additional requirements specific to visual text.

**Requirements:**
* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

**To install and develop locally:**
``` bash
git clone https://github.com/esalesky/visrep
cd visrep
pip install --editable ./
pip install -r examples/visual_text/requirements.txt
```

## Running the code

Training and evaluation can be called as with normal fairseq. 
The following parameters are unique to visrep: 

```
--task visual_text 
--arch visual_text_transformer 
--image-window {VALUE} 
--image-stride {VALUE} 
--image-font-path {VALUE} (we have included the NotoSans fonts we used in this repo: see fairseq/data/visual/fonts/)
--image-embed-normalize
--image-embed-type {VALUE} (options for number of convolutional blocks: e.g., direct, 1layer, 2layer, ..
```
Visual text parameters are serialized into saved models and do not need to be specified at inference time.  
Image samples can also optionally be written to the MODELDIR/samples/ subdirectory using `--image-samples-path` (directory to write to) and `--image-samples-interval` N (write every Nth image). 

<details>
  <summary><strong>Best visual text parameters</strong></summary><p>
  
  * **MTTT**
    + ar-en: 1layer, window 27, stride 10, fontsize 14, batch 20k
    + de-en: 1layer, window 20, stride 5, fontsize 10, batch 20k
    + fr-en: 1layer, window 15, stride 10, fontsize 10, batch 20k
    + ko-en: 1layer, window 25, stride 8, fontsize 12, batch 20k
    + ja-en: 1layer, window 25, stride 8, fontsize 10, batch 20k
    + ru-en: 1layer, window 20, stride 10, fontsize 10, batch 20k
    + zh-en: 1layer, window 30, stride 6, fontsize 10, batch 20k
  * **WMT (filtered)**
    + de-en: direct, window 30, stride 20, fontsize 8, batch 40k
    + zh-en: direct, window 25, stride 10, fontsize 8, batch 40k
  
  </p>
</details>

### Grid scripts

We include our grid scripts, which use the UGE scheduler, in [grid_scripts](https://github.com/esalesky/visrep/tree/main/grid_scripts).  
These include `*.qsub`, `train.sh`, `train-big.sh`, `translate.sh`, `translate-big.sh`, and `translate-all-testsets.sh` to bulk queue translation of multiple test sets. 
The .sh scripts have the hyperparameters for the small ([MTTT](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/)) and larger datasets. 

**Example:**
```
export lang=fr; export window=25; export stride=10; 
qsub train.qsub /exp/esalesky/visrep/exp/$lang-en/1layernorm.window$window.stride$stride.fontsize10.batch20k $lang en --image-font-size 10 --image-window $window --image-stride $stride --image-embed-type 1layer --update-freq 2
```


### Important Files

* [fairseq/tasks/visual_text.py](https://github.com/esalesky/visrep/blob/main/fairseq/tasks/visual_text.py)

  The visual text task. Does data loading, instantiates the model for training, and creates the data for inference.

* [fairseq/data/visual/visual_text_dataset.py](https://github.com/esalesky/visrep/blob/main/fairseq/data/visual/visual_text_dataset.py)

  Creates a visual text dataset object for fairseq.

* [fairseq/data/visual/image_generator.py](https://github.com/esalesky/visrep/blob/main/fairseq/data/visual/image_generator.py)

  Loads the raw data, and generates images from text. 
  
  To generate individual samples from `image_generator.py` directly, it can be called like so:
  ```
  ./image_generator.py --font-size 10 --font-file fonts/NotoSans-Regular.ttf --text "This is a sentence." --prefix english --window 25 --stride 10
  ```
  `combine.sh` in the same directory can combine the slices into a single image to visualize what the image tokens for a sentence look like (as in Table 6 in the paper). 

* [fairseq/models/visual/visual_transformer.py]()
  (Note: fairseq/models/visual_transformer.py is UNUSED)

  Creates the VisualTextTransformerModel. 
  This has a VisualTextTransformerEncoder and a normal decoder. 
  The only thing that is unique to this encoder is that it calls self.cnn_embedder to create source representations
  
* There may be additional obsolete visual files in the repository. 

## Inducing noise

We induced five types of noise, as below:
- **swap**: swaps two adjacent characters per token. applies to words of length >=2 *(Arabic, French, German, Korean, Russian)*
- **cmabrigde**: permutes word-internal characters with first and last character unchanged. applies to words of length >=4 *(Arabic, French, German, Korean, Russian)*
- **diacritization**: diacritization, applied via [camel-tools](https://github.com/CAMeL-Lab/camel_tools) *(Arabic)*
- **unicode**: substitutes visually similar Latin characters for Cyrillic characters *(Russian)*
- **l33tspeak**: substitutes numbers or other visually similar characters for Latin characters *(French, German)*

The scripts to induce noise are in [scripts/visual_text](https://github.com/esalesky/visrep/tree/main/scripts/visual_text), where -p is the probability of inducing noise per-token, and can be run as below. In our paper we use p from 0.1 to 1.0, in intervals of 0.1.

```
cat test.de-en.de | python3 scripts/visual_text/swap.py -p 0.1 > visual/test-sets/swap_10.de-en.de
cat test.ko-en.ko | python3 scripts/visual_text/cmabrigde.py -p 0.1 > visual/test-sets/cam_10.ko-en.ko
cat test.ar-en.ar | python3 scripts/visual_text/diacritization.py -p 0.1 > visual/test-sets/dia_10.ar-en.ar
cat test.ru-en.ru | python3 scripts/visual_text/cyrillic_noise.py -p 0.1 > visual/test-sets/cyr_10.ru-en.ru
cat test.fr-en.fr | python3 scripts/visual_text/l33t.py -p 0.1 > visual/test-sets/l33t_10.fr-en.fr
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
