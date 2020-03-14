
2020-03-14

Train and evaluate visual word embeddings

Description
This code will train a model for visual word embeddings.
Input is a seed text file (mt dictionary), training font file, and validation font file.
Word images are generated at train time using random fonts (type, style, size, padding, rotation).
The model uses ResNet18 and the AdaptiveAvgPool2d layer to provide a 512-dim feature vector.

Code
train.py - Train a word embedding model
models.py - ResNet18
dataset.py - Generate synthetic word images from seed text
augment.py - Add random augmentations (noise, pixel dropout) during training 

Scripts
train_visemb.sh - Train model (mt dictionary w train/valid font and augmentation)
extract_embedding.sh - Extract visual embeddings (mt dictionary w test (held out font))

Example
-- Train visual embedding model --
qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N kovis \
-l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
/expscratch/detter/src/fairseq/fairseq-ocr/visual/scripts/train_visemb.sh ko

-- Extract visual embeddings --
qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N koext \
-l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
/expscratch/detter/src/fairseq/fairseq-ocr/visual/scripts/extract_embedding.sh ko

-- Cosine score visual embeddings --
qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N koscr \
-l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
/expscratch/detter/src/fairseq/fairseq-ocr/visual/scripts/score_embedding.sh ko
