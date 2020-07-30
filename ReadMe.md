
# Introduduction

Fairseq for Visual Machine Translation

## Supported training modes

Fairseq-ocr supports a number of experimental variables:

- *granularity*: supervised (token-based) or unsupervised (sentence-based); and
- *training*: pretrained, end-to-end, and multitask.

The supervised approach generates images at the token level; these
token-level representations are computed at run-time and replace the
embedding lookup. In the unsupervised setting, the entire sentence is
visualized, and embeddings are generated for each "frame". There is no
alignment between these frames and the input tokens unless generated
by CTC loss or some other technique.

For training, there are three modes. We can start with pretrained word
embeddings, which then become the input to the computation graph. We
can also train in an end-to-end fashion, where the OCR code is
actually run, but there is only the downstrain loss of
(target-language) cross-entropy. Finally, we can do multitask, in
which we add a loss function that reconstructs the original input
tokens in addition to computing normal cross-entropy loss.

### What's New:

Mar 14, 2020:  
- pretrain embeddings  
Finished pretrain code and renamed folder (pretrain folder)  
Load pretrain embeddings (image_dataset.py line 702)  
- write image training samples  
Write sample images when args.image_verbose is enabled (train.py line 160)  
- train time cache generation  
Added train time cache generation (image_dataset.py line 734)  
New flag for preload cache (visualmt.py 283 --image-preload-cache)  
- Add embedding concat  
Visual line and token concat (visual_transformer.py line 578)  

### Features:
- Pretrain visual embeddings    
- Sentence visual training  

### Scripts:

- pretrain  
train zh - pretrain/scripts/sentence/vismt/train_zh.sh   
extract zh -  pretrain/scripts/sentence/vismt/train_zh.sh  

- train vismt  
concat with pretrain embedding - grid_scripts/dave_scripts/train_pretrain_concat.sh  
visonly with pretrain embedding - grid_scripts/dave_scripts/train_pretrain_visonly.sh  

### To do:

- Auxillary loss (CTC)  
Add auxillary CTC loss for visual embeddings  

### Sentence Results

- arch  
transformer_iwslt_de_en  
- input  
/exp/esalesky/mtocr19/zh-en/data/5k  
- script  
baseline robust/grid_scripts/dave_scripts/train_orig.sh  

Language | Baseline | VisDiable | PreConcat | PreVisOnly | VisConcat | VisOnly
--- | --- | --- | --- | --- | --- | ---
Zh | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0   
