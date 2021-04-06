# Fairseq for Machine Translation via Visual Text

[[_TOC_]]

## Files

The code is implemented via the following files:

* grid_scripts/train.sh
* grid_scripts/train_wrapper.sh
* fairseq/tasks/visual_text.py
* fairseq/data/visual_text_dataset.py
* fairseq/data/image_generator.py
* fairseq/models/visual/visual_transformer.py
  (Note: fairseq/models/visual_transformer.py is UNUSED)

There are other files scattered about that are not used.

## Description
In this work, we learn a visual representation of source tokens that when added to a token embedding improves the final target translation. The visual representation is learned from an image of the source token or sentence, that is generated using a font file and gaming engine.  

We investigate both aligned and unaligned visual representations. For the aligned model, we learn a 512-dimensional visual representation for each word image (dictionary entry) of a source sentence. For the unaligned model, we learn a 512-dimensional visual representation for each step width of a full-sentence image.  

The visual representation model can be learned as an end-to-end system with a machine translation (MT) architecture or as a pre-trained model that is passed to the MT model. In this set of experiments, we pre-train the visual representation model and then feed the visual embeddings to a transformer based encoder-decoder MT model. 

Both the aligned and unaligned pre-trained models use an OCR CNN-based architecture. The OCR model consists of 7 convolution layers with max-pooling after layers 2 and 4. A bridge layer is used to combine the CNN output height and channels into the 512-dimensional embedding at each step width. The aligned model uses an adaptive average pooling layer after the CNN to pool the step width to a single dimension per word. The unaligned model uses fractional max pooling after layers 2 and 4, which controls the final step width of the sentence representation.  

The aligned model is trained with a cross-entropy loss for each visual word representation. The unaligned model is trained with a connectionist temporal classification loss over the sequence of words in the sentence. 

## Major features
- online image generation for both tokens and sentences
- cache token images to speed up training
- ability to load a pretrain visual model or train end-to-end
- ability to enable/disable gradient updates from the pre-train model
- control step width for visual sentence embeddings
- source side loss for visual learning
- tensorboard statistics for loss, accuracy, and character error rate

## Steps to train pre-train aligned models
1. pre-train aligned visual model - fairseq-ocr/dave_grid_scripts/pretrain/aligned/pretrain_aligned.sh (create checkpoint)
2. decode aligned visual model - fairseq-ocr/dave_grid_scripts/pretrain/aligned/extract_aligned.sh (create numpy image files)  
3. train transformer model - fairseq-ocr/dave_grid_scripts/vistrans/aligned/train_vismt_concat.sh (loads pre-train checkpoint and reads numpy image files)
4. score model - fairseq-ocr/dave_grid_scripts/vistrans/aligned/score_aligned.sh


## Steps to train pre-train unaligned models
1. pre-train unaligned visual model - fairseq-ocr/dave_grid_scripts/pretrain/unaligned/pretrain_unaligned.sh (create checkpoint)
2. decode unaligned visual model - fairseq-ocr/dave_grid_scripts/pretrain/unaligned/extract_unaligned.sh (create numpy image files)  
3. train transformer model - fairseq-ocr/dave_grid_scripts/vistrans/unaligned/train_vismt_visonly.sh (loads pre-train checkpoint and reads numpy image files)
4. score model - fairseq-ocr/dave_grid_scripts/vistrans/unaligned/score_unaligned.sh


## Tensorboard with sshfs
- create path on local machine - /LOCAL_PATH
- sudo sshfs -o allow_other USER_ID@REMOTE_ADDRESS:/EXPERIMENT_TENSORBOARD_PATH /LOCAL_PATH
- start tensorboard on local machine - tensorboard --logdir=/LOCAL_PATH
*Note: if you get a local error, "Transport endpoint is not connected", use "sudo umount /LOCAL_PATH"

## Locations      
- code: https://gitlab.hltcoe.jhu.edu/matt.post/fairseq-ocr  
- branch: robust  
- data: /exp/esalesky/mtocr19/zh-en/data/10k   
- font: /exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf (size 16)  
- pretrain checkpoints (zh):   
  - aligned - /expscratch/detter/vismt/zh/20200730/aligned/checkpoints/model_ckpt_best.pth
  - unaligned - /expscratch/detter/vismt/zh/20200730/unaligned/checkpoints/model_ckpt_best.pth 
- pretrain numpy:  
  - aligned - /expscratch/detter/vismt/zh/20200730/aligned/decode_images_[train|valid|test].tar.gz
  - unaligned - /expscratch/detter/vismt/zh/20200730/unaligned/decode_images_[train|valid|test].tar.gz
- scripts:  
  - pretrain - [pretrain branch] fairseq-ocr/scripts
    
## Experiments   
Initial pre-train experiments use data from The Multitarget TED Talks Task.  
Experiments use Chinese (zh) character level input source with English (en) 10K target.

### Baseline

| Language | Description       | BLEU4 | Notes                                                                  |
| -------- | ----------------- |-------|------------------------------------------------------------------------|
| zh-en    | token only        | 18.71 | 100 epochs                                                             |

### Aligned

| Language | Description       | BLEU4 | Notes                                                                  |
| -------- | ----------------- |-------|------------------------------------------------------------------------|
| zh-en    | visual only       | 17.82 | 100 epochs, pre-train, disable gradient updates, normalize embeddings |
|          | visual only e2e   |  8.77 | 100 epochs, end-to-end training                                        |
|          | concat            | 18.60 | 100 epochs, pre-train, disable gradient updates, normalize embeddings |
|          | concat e2e        | 14.80 | 100 epochs, end-to-end training                                        |
|          | add               | 19.10 | 100 epochs, pre-train, disable gradient updates, normalize embeddings |
|          | avg               | 18.68 | 100 epochs, pre-train, disable gradient updates, normalize embeddings |

### Unaligned   
  
| Language | Description       | BLEU4 | Notes                                                                  |
| -------- | ----------------- |-------|------------------------------------------------------------------------|
| zh-en    | visual only       |       |                                                                        |
|          | visual only e2e   |       |                                                                        |

  
