#!/bin/bash

# Preprocesses the data.
#
# Usage:
#
#     preprocess.sh DATADIR BINDIR SOURCE_LANG TARGET_LANG
#

DATADIR=$1
BINPREFIX=$2
SRC=$3
TGT=$4

WINDOW=24
STRIDE=12
FONTSIZE=10
WORKERS=20

FAIRSEQ=/exp/esalesky/newrender/visrep
export PYTHONPATH=$FAIRSEQ

FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSans-Regular.ttf

FONTNAME=$(basename $FONTPATH .ttf)
FONTNAME=$(basename $FONTNAME .otf)
BINDIR=${BINPREFIX}.ppb${WINDOW}.stride${STRIDE}.font${FONTSIZE}

mkdir -p $BINDIR

echo "FAIRSEQ: $FAIRSEQ"
echo "FONTPATH: $FONTPATH"
echo "FONTSIZE: $FONTSIZE"
echo "WINDOW: $WINDOW"
echo "STRIDE: $STRIDE"
echo "DATADIR: $DATADIR"
echo "BINDIR: $BINDIR"
echo "WORKERS: $WORKERS"

PYTHONPATH=$FAIRSEQ python3 -m fairseq_cli.preprocess \
  --destdir $BINDIR \
  --workers $WORKERS \
  -s $SRC -t $TGT \
  --trainpref $DATADIR/train.${SRC}-${TGT} \
  --validpref $DATADIR/dev.${SRC}-${TGT} \
  --tgtdict $(abspath $DATADIR)/dict.$TGT.txt --thresholdtgt 1 \
  --visual-text \
  --image-font-path $FONTPATH \
  --image-font-size $FONTSIZE \
  --pixels-per-patch $WINDOW 

#  --image-samples-path $BINDIR/samples_${SRC} \
#  --image-samples-interval 1000000 
