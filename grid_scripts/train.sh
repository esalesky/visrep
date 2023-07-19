#!/bin/bash
# =========

set -eu

MODELDIR=$1
SRC=$2
TGT=$3
DATADIR=$4

shift
shift
shift
shift

FAIRSEQ=/exp/esalesky/newrender/visrep
FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSans-Regular.ttf
export PYTHONPATH=$FAIRSEQ

echo "HOSTNAME: $(hostname)"
nvidia-smi

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "FAIRSEQ: $FAIRSEQ"
echo "FONTPATH: $FONTPATH"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "DATADIR: $DATADIR"
echo "MODELDIR: $MODELDIR"

mkdir -p $MODELDIR
mkdir -p $MODELDIR/samples

cp $DATADIR/dict.*.txt $MODELDIR/
cp $DATADIR/lang_list $MODELDIR/

cp $0 $MODELDIR/train.sh
echo "$@" > $MODELDIR/args

lang_pairs="ar-en,az-en,be-en,bg-en,bn-en,bs-en,cs-en,da-en,de-en,el-en,eo-en,es-en,et-en,eu-en,fa-en,fi-en,frca-en,fr-en,gl-en,he-en,hi-en,hr-en,hu-en,hy-en,id-en,it-en,ja-en,ka-en,kk-en,ko-en,ku-en,lt-en,mk-en,mn-en,mr-en,ms-en,my-en,nb-en,nl-en,pl-en,ptbr-en,pt-en,ro-en,ru-en,sk-en,sl-en,sq-en,sr-en,sv-en,ta-en,th-en,tr-en,uk-en,ur-en,vi-en,zhcn-en,zh-en,zhtw-en"
lang_list=$DATADIR/lang_list

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.train \
  ${DATADIR} \
  --task pixel_translation_multi_simple_epoch \
  --arch visual_text_transformer \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --sampling-method "temperature" \
  --sampling-temperature 1.5 \
  --save-dir $MODELDIR \
  --source-dict $DATADIR/dict.$TGT.txt \
  --target-dict $DATADIR/dict.$TGT.txt \
  --validate-interval 1 \
  --keep-last-epochs 1 \
  --keep-best-checkpoints 1 \
  --save-interval-updates 10000 \
  --patience 10 \
  --max-epoch 200 \
  --max-tokens 10000 \
  --update-freq=6 \
  --image-embed-type 1layer \
  --image-embed-normalize \
  --image-font-path $FONTPATH \
  --image-font-size 10 \
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
  --dataset-impl mmap \
  --share-decoder-input-output-embed \
  --warmup-updates 8000 \
  --warmup-init-lr '1e-07' \
  --weight-decay 0.0001 \
  --log-format json \
  --log-interval 10 \
  --skip-invalid-size-inputs-valid-test \
  "$@" \
> $MODELDIR/log 2>&1

chmod 444 $MODELDIR/log

echo "Done training."
echo done > $MODELDIR/status
