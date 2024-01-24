# multilingual visrep

**This branch focuses on training multilingual translation models with pixel representations,** or <ins>vis</ins>ual <ins>rep</ins>resentations (visrep). Visrep models process input text rendered as images in place of tokenization with a fixed source vocabulary. 

We trained multilingual pixel translation models using two TED datasets (TED-7 and TED-59), and showed improved performance compared to subword embeddings. Pixel representations increased positive transfer across languages with a shared script, and enable cross-lingual transfer to unseen scripts without changes to the model. Additionally, we found pixel models to be more data-efficient than subwords, once sufficient samples to learn a new script have been observed. 

For further details, please see our paper [Multilingual Pixel Representations for Translation and Effective Cross-lingual Transfer](https://arxiv.org/abs/2305.14280) from EMNLP 2023.  Information on how to cite our work can be found [below](#citation-and-contact).

## Setup

The installation is the same as [fairseq](https://github.com/pytorch/fairseq), plus additional requirements specific to rendering text.  
This branch uses a different rendering library (pangocairo) than the main branch (pygame), which has additional dependencies. We suggest creating separate conda environments to work with each renderer. 

**Requirements:**
* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also want an NVIDIA GPU, and [NCCL](https://github.com/NVIDIA/nccl) (for multi-gpu training)

**To install and develop locally:**

In a fresh conda environment... 

``` bash
git clone -b multi git@github.com:esalesky/visrep.git
cd visrep
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install -c conda-forge pycairo pygobject manimpango
pip install --editable ./
pip install -r examples/visual_text/requirements.txt
```

## Training & Inference

**Pixel models**  
We extend the fairseq `translation_multi_simple_epoch` task to use pixel representations on the source side as `pixel_translation_multi_simple_epoch`. This task expects the same parameters and files as the base task; source dict files are expected but unused and can be empty or linked to the target dict. 

Scripts for training and inference with our exact parameters for TED-59 can be found in the `grid_scripts/` directory [here](https://github.com/esalesky/visrep/blob/multi/grid_scripts/). 

**Subword models**  
Our subword baseline follows the fairseq `translation_multi_simple_epoch` recipe linked [here](https://github.com/facebookresearch/fairseq/blob/main/examples/multilingual/README.md), using the sentencepiece vocabularies shared on Zenodo below.

## Data & Models

To facilitate working with our models and further comparisons using the TED-59 dataset, we have posted our pixel and subword models and the dataset in expected fairseq format on [Zenodo](https://zenodo.org/records/10086264). 

## Expected Results

Full results reported by individual language pair for three metrics (BLEU, chrF, and COMET) on TED-59 and TED-7 can be found in [Appendix E of our paper](https://arxiv.org/pdf/2305.14280.pdf#page=14).

## Citation and Contact

```bibtex
@inproceedings{salesky-etal-2023-multilingual,
    title = "Multilingual Pixel Representations for Translation and Effective Cross-lingual Transfer",
    author = "Salesky, Elizabeth  and
      Verma, Neha  and
      Koehn, Philipp  and
      Post, Matt",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.854",
    doi = "10.18653/v1/2023.emnlp-main.854",
    pages = "13845--13861",
}

@inproceedings{salesky-etal-2021-robust,
    title = "Robust Open-Vocabulary Translation from Visual Text Representations",
    author = "Salesky, Elizabeth  and
      Etter, David  and
      Post, Matt",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.576",
    doi = "10.18653/v1/2021.emnlp-main.576",
}
```
<hr> 

**Contact person:**  Elizabeth Salesky ([elizabeth.salesky@gmail.com](mailto:elizabeth.salesky@gmail.com))

Please feel free to send an email or open an issue here with questions about our code or models. We emphasize that this is experimental research code and we may provide advice for new feature requests rather than implementation. 
