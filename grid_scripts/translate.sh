#!/bin/bash

set -eu

## Settings
SRC=de
TRG=en
FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr

MODELDIR=$1
INPUT=$2
OUTPUT=$3
REF=$4
shift
shift
shift
shift

export PYTHONPATH=$FAIRSEQ

linesneeded=$(cat $INPUT | wc -l)
linesfound=0
[[ -e $OUTPUT ]] && linesfound=$(grep ^D- $OUTPUT | wc -l)
if [[ $linesneeded -eq $linesfound ]]; then
    echo "Cowardly refusing to regenerate existing and complete file $OUTPUT"
    echo "Full command: $0 $MODELDIR $INPUT $OUTPUT $REF"
    exit 1
fi

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "FAIRSEQ: $FAIRSEQ"
echo "MODELDIR: $MODELDIR"

infile=$(echo $OUTPUT | perl -pe "s|/out\.|/in.|")
reffile=$(echo $OUTPUT | perl -pe "s|/out\.|/ref.|")
ln -sf $INPUT $infile
ln -sf $REF $reffile

cat $INPUT \
| PYTHONPATH=$FAIRSEQ python -m fairseq_cli.interactive \
  $MODELDIR \
  --task 'visual_text' \
  --path $MODELDIR/checkpoint_best.pt \
  -s $SRC -t $TRG \
  --target-dict dict.$TRG.txt \
  --beam 5 \
  "$@" \
> $OUTPUT 2>&1

echo "model epoch for $MODELDIR/checkpoint_best.pt is $(~/bin/getepoch.py $MODELDIR/checkpoint_best.pt)" >> $OUTPUT

cleanfile=$(echo $OUTPUT | perl -pe "s|/out\.|/clean.|")
bleufile=$(echo $OUTPUT | perl -pe "s|/out\.|/bleu.|")

grep ^D- $OUTPUT | sort -V | cut -f 3 | deseg \
| tee $cleanfile \
| sacrebleu -b $REF \
> $bleufile
