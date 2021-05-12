#!/bin/bash

# Preprocesses the data.
#
# Usage:
#
#     preprocess.sh BINDIR SOURCE_LANG TARGET_LANG
#
# Assumes data is found in a fixed location parameterized by $SRC and $TRG.

set -eu

: ${WINDOW=30}
: ${STRIDE=20}
: ${FONTSIZE=8}
DATADIR=$1
BINPREFIX=$2
SRC=$3
TRG=$4

shift
shift
shift
shift

: ${WORKERS=20}

FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr
export PYTHONPATH=$FAIRSEQ
#DATADIR=/exp/mpost/mtocr19/data/unaligned/$SRC-$TRG/5k

case ${SRC} in
  ru | de | fr | en )
    : ${FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSans-Regular.ttf}
    ;;
  ar )
    : ${FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoNaskhArabic-Regular.ttf}
    ;;
  zh | ja | ko )
    FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSansCJKjp-Regular.otf
    ;;
  *)
    echo "You didn't set a font path for language ${SRC}, you turd!"
    exit 1
    ;;
esac

FONTNAME=$(basename $FONTPATH .ttf)
FONTNAME=$(basename $FONTNAME .otf)
BINDIR=$BINPREFIX.window$WINDOW.stride$STRIDE.font-$FONTNAME.size$FONTSIZE

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
  -s $SRC -t $TRG \
  --trainpref $DATADIR/train \
  --validpref $DATADIR/valid \
  --tgtdict $(abspath $DATADIR)/dict.$TRG.txt --thresholdtgt 1 \
  --visual-text \
  --image-font-path $FONTPATH \
  --image-font-size $FONTSIZE \
  --image-window $WINDOW \
  --image-stride $STRIDE \
  --image-samples-path $BINDIR/samples \
  "$@"
