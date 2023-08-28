#!/bin/bash
# =========

set -eu

SRC=$1
MODELDIR=$BOLT_ARTIFACT_DIR
DATADIR=/mnt/task_runtime/ted_data/${SRC}
TGT=en

shift

FAIRSEQ=/mnt/task_runtime/im2text
FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSans-Regular.ttf
export PYTHONPATH=$FAIRSEQ

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
nvidia-smi

echo "PATH: ${PATH}"
echo "FAIRSEQ: $FAIRSEQ"
echo "FONTPATH: $FONTPATH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DATADIR: $DATADIR"
echo "MODELDIR: $MODELDIR"

mkdir -p $MODELDIR/samples

cp $DATADIR/dict.*.txt $MODELDIR/
cp $0 $MODELDIR/train.sh
echo "$@" > $MODELDIR/args


PYTHONPATH=$FAIRSEQ python -m fairseq_cli.train \
  ${DATADIR} \
  --task visual_text_aug \
  --arch visual_text_transformer \
  --save-dir $MODELDIR \
  --source-lang $SRC \
  --target-lang $TGT \
  --target-dict $DATADIR/dict.$TGT.txt \
  --validate-interval 1 \
  --keep-last-epochs 1 \
  --keep-best-checkpoints 1 \
  --patience 10 \
  --max-epoch 200 \
  --max-tokens 10000 \
  --update-freq=4 \
  --image-embed-type 1layer \
  --image-embed-normalize \
  --image-font-path $FONTPATH \
  --image-font-size 10 \
  --image-samples-interval 100000 \
  --criterion 'label_smoothed_cross_entropy' \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-08 \
  --decoder-attention-heads 4 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 4096 \
  --decoder-layers 3 \
  --dropout 0.3 \
  --encoder-attention-heads 4 \
  --encoder-embed-dim 512 \
  --encoder-ffn-embed-dim 4096 \
  --encoder-layers 12 \
  --label-smoothing 0.2 \
  --lr 0.0005 \
  --lr-scheduler 'inverse_sqrt' \
  --max-source-positions 1024 \
  --max-target-positions 1024 \
  --max-tokens-valid 2000 \
  --min-loss-scale 0.0001 \
  --no-epoch-checkpoints \
  --num-workers 0 \
  --optimizer 'adam' \
  --dataset-impl raw \
  --share-decoder-input-output-embed \
  --warmup-updates 8000 \
  --warmup-init-lr '1e-07' \
  --weight-decay 0.0001 \
  --log-format json \
  --log-interval 10 \
  --skip-invalid-size-inputs-valid-test \
  "$@" \
> $MODELDIR/log 2>&1


echo "Done training."
echo done > $MODELDIR/status
