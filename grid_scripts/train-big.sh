#!/bin/bash

# Does visual end-to-end training with standard MT loss.
#
# Usage:
#
# train.sh MODELDIR SOURCE_LANG TARGET_LANG [FAIRSEQ ARGS...]
#
#     cd ~mpost/exp/mtocr19/runs
#     for lang in ja zh ko de fr; do
#       for window in 15 20 25 30; do
#         let bottom=window-10
#         for stride in $(seq $window -5 $bottom); do
#           qsub train.qsub $lang-en/5k.max-10k.window$window.stride$stride.fontsize10 $lang en --image-font-size 8 --image-window $window --image-stride $stride \
#         done
#       done
#     done

set -eu

MODELDIR=$1
SRC=$2
TRG=$3

shift
shift
shift

FAIRSEQ=/exp/esalesky/visrep/fairseq-ocr
DATADIR=/expscratch/esalesky/visrep21/de-en/bin.window35.stride10.font-NotoSans-Regular.size10
#DATADIR=/exp/mpost/mtocr19/data/unaligned/$SRC-$TRG/5k

case ${SRC} in
  ru | de | fr | en )
    : ${FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSans-Regular.ttf}
    ;;
  ar )
#    : ${FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoNaskhArabic-Regular.ttf}
    : ${FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSansArabic-Regular.ttf}
    ;;
  zh | ja | ko )
    FONTPATH=$FAIRSEQ/fairseq/data/visual/fonts/NotoSansCJKjp-Regular.otf
    ;;
  *)
    echo "You didn't set a font path for language ${SRC}, you turd!"
    exit 1
    ;;
esac

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


#if [[ -e $MODELDIR ]]; then
#    echo "Refusing to run training since $MODELDIR already exists"
#    exit 1
#fi

mkdir -p $MODELDIR

cp $DATADIR/dict.$SRC.txt $MODELDIR
cp $DATADIR/dict.$TRG.txt $MODELDIR

cp $0 $MODELDIR/train.sh
echo "$@" > $MODELDIR/args

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.train \
  ${DATADIR} \
  --task visual_text \
  --arch visual_text_transformer \
  -s $SRC -t $TRG \
  --save-dir $MODELDIR \
  --target-dict $DATADIR/dict.$TRG.txt \
  --validate-interval 1 \
  --patience 10 \
  --max-epoch 200 \
  --max-tokens 10000 \
  --update-freq=1 \
  --image-samples-path ${MODELDIR}/samples \
  --image-samples-interval 100000 \
  --image-embed-type 1layer \
  --image-embed-normalize \
  --image-font-path $FONTPATH \
  --criterion 'label_smoothed_cross_entropy' \
  --adam-betas '(0.9, 0.98)' \
  --adam-eps 1e-08 \
  --decoder-attention-heads 8 \
  --decoder-embed-dim 512 \
  --decoder-ffn-embed-dim 2048 \
  --decoder-layers 6 \
  --dropout 0.1 \
  --encoder-attention-heads 8 \
  --encoder-embed-dim 512 \
  --encoder-ffn-embed-dim 2048 \
  --encoder-layers 6 \
  --label-smoothing 0.1 \
  --lr 5e-4 \
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
  --warmup-updates 4000 \
  --weight-decay 0.0001 \
  --log-format json \
  --log-interval 100 \
  "$@" \
> $MODELDIR/log 2>&1

chmod 444 $MODELDIR/log

echo "Done training."
echo done > $MODELDIR/status

# evaluate
echo "Starting evaluation on test sets..."
for testset in /exp/esalesky/visrep/fairseq-ocr/visual/test-sets/mttt.$SRC-en.$SRC; do
    echo "Evaluating $testset..."
    ref=$(echo $testset | perl -pe "s/.$SRC$/.en/");
    qsub /exp/esalesky/visrep/fairseq-ocr/grid_scripts/translate.qsub $MODELDIR $testset $MODELDIR/out.$(basename $testset) $ref
done
