#!/bin/bash

## Settings
SRC=de
TRG=en
FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr

MODELDIR=$1
shift

export PYTHONPATH=$FAIRSEQ

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "FAIRSEQ: $FAIRSEQ"
echo "MODELDIR: $MODELDIR"

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.interactive \
  $MODELDIR \
  --task 'visual_text' \
  --path $MODELDIR/checkpoint_best.pt \
  -s $SRC -t $TRG \
  --target-dict dict.$TRG.txt \
  --beam 5 \
  "$@"
