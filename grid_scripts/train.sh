#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
#
# End-to-end training with source loss.
#

# module load cuda10.1/toolkit/10.1.105
# module load cudnn/7.6.1_cuda10.1
# module load gcc/7.2.0

set -eu

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

MODELDIR=$1
shift

## Settings
SRC=de
TRG=en
FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr
DATADIR=/exp/mpost/mtocr19/data/unaligned/5k/$SRC-$TRG

export PYTHONPATH=$FAIRSEQ

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DATADIR: $DATADIR"
echo "FAIRSEQ: $FAIRSEQ"
echo "MODELDIR: $MODELDIR"

[[ ! -d $MODELDIR ]] && mkdir -p $MODELDIR

[[ -e $MODELDIR ]] && cp $DATADIR/dict.$SRC.txt $MODELDIR
[[ -e $MODELDIR ]] && cp $DATADIR/dict.$TRG.txt $MODELDIR

cp $0 $MODELDIR/train.sh
echo "$@" > $MODELDIR/args

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.train \
  ${DATADIR} \
  --task 'visual_text' \
  --arch visual_text_transformer \
  -s $SRC -t $TRG \
  --save-dir ${MODELDIR} \
  --target-dict $DATADIR/dict.$TRG.txt \
  --validate-interval-updates 1000 \
  --patience 10 \
  --update-freq=1 \
  --image-samples-path ${MODELDIR}/samples \
  --image-samples-interval 10000 \
  --image-embed-type 'visonly' \
  --image-embedding-normalize \
  --criterion 'label_smoothed_cross_entropy' \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-08 \
  --decoder-attention-heads 4 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 1024 \
  --decoder-layers 6 \
  --dropout 0.3 \
  --encoder-attention-heads 4 \
  --encoder-embed-dim 512 \
  --encoder-ffn-embed-dim 1024 \
  --encoder-layers 6 \
  --label-smoothing 0.1 \
  --lr 5e-4 \
  --lr-scheduler 'inverse_sqrt' \
  --max-epoch 100 \
  --max-source-positions 1024 \
  --max-target-positions 1024 \
  --max-tokens 10000 \
  --max-tokens-valid 2000 \
  --min-loss-scale 0.0001 \
  --no-epoch-checkpoints \
  --num-workers 8 \
  --optimizer 'adam' \
  --dataset-impl raw \
  --share-decoder-input-output-embed \
  --warmup-updates 4000 \
  --weight-decay 0.0001 \
  --log-format json \
  --log-interval 10 \
  "$@" \
> $MODELDIR/log 2>&1

echo done > $MODELDIR/status

# evaluate
outfile=$MODELDIR/out.mttt.test1
cat ~/data/bitext/raw/multitarget-ted/en-de/raw/ted_test1_en-de.raw.de | ./interactive.sh $MODELDIR > $outfile

cleanfile=$MODELDIR/clean.mttt.test1
bleufile=$MODELDIR/bleu.mttt.test1
grep ^D- $outfile | sort -V | cut -f 3 | debpe | tee $cleanfile | sacrebleu -b ~/data/bitext/raw/multitarget-ted/en-de/raw/ted_test > $bleufile

