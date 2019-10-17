
2019-10-14

Train and evaluate visual word embeddings

Description
This code will train a model for visual word embeddings.
Input is a seed text file, training font file, and validation font file.
Word images are generated at train time using random fonts (type, style, size, padding, rotation).
The model uses ResNet18 and the AdaptiveAvgPool2d layer to provide a 512-dim feature vector.

Code
train.py - Train a word embedding model
model.py - ResNet18
dataset.py - Generate synthetic word images from seed text
augment.py - Add random augmentations (noise, pixel dropout) during training 

Scripts
gird_scripts.sh - Run code on the gpu grid

Example

python /expscratch/detter/src/fairseq-ocr/visual/train.py \
--input=/exp/mpost/mtocr19/data/mttt/de-en/fairseq_binary/dict.de.txt \
--font=/expscratch/detter/fonts/mt/train_font.txt \
--valid_font=/expscratch/detter/fonts/mt/valid_font.txt \
--output=/expscratch/detter/mt/visual

