#!/bin/bash

# Does visual end-to-end training with standard MT loss.
#
# Usage:
#
#     cd ~mpost/exp/mtocr19/runs
#     for lang in ja zh ko de fr; do 
#       for window in 15 20 25 30; do 
#         let bottom=window-10
#         for stride in $(seq $window -5 $bottom); do 
#           qsub -v FONTSIZE=10 -v WINDOW=$window -v STRIDE=$stride train_wrapper.sh train.sh $lang-en/5k.max-10k.window$window.stride$stride.fontsize10 $lang en
#         done
#       done
#     done

set -eu

: ${WINDOW=30}
: ${STRIDE=20}
: ${FONTSIZE=8}
MODELDIR=$1
SRC=$2
TRG=$3
ORIGMODELDIR=$4

shift
shift
shift
shift

if [[ ! -e $ORIGMODELDIR/checkpoint_best.pt ]]; then
    echo "Found no checkpoint at $ORIGMODELDIR/checkpoint_best.pt, quitting"
    exit 1
fi

FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr
DATADIR=/exp/mpost/mtocr19/data/unaligned/$SRC-$TRG/5k

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

export PYTHONPATH=$FAIRSEQ

# echo "HOSTNAME: $(hostname)"
# nvidia-smi

# echo "PATH: ${PATH}"
# echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
# echo "FAIRSEQ: $FAIRSEQ"
# echo "FONTPATH: $FONTPATH"
# echo "WINDOW: $WINDOW"
# echo "STRIDE: $STRIDE"
# echo "FONTSIZE: $FONTSIZE"
# echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
# echo "DATADIR: $DATADIR"
# echo "MODELDIR: $MODELDIR"

if [[ -e $MODELDIR ]]; then
    echo "Refusing to run training since $MODELDIR already exists"
    exit 1
fi

mkdir -p $MODELDIR

for file in train.{$SRC,$TRG} valid.{$SRC,$TRG} dict.$SRC.txt; do
    cp $DATADIR/$file $MODELDIR
done
cp $ORIGMODELDIR/dict.$TRG.txt $MODELDIR

cp $0 $MODELDIR/train.sh
echo "$@" > $MODELDIR/args

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.train \
  $MODELDIR \
  --finetune-from-model $ORIGMODELDIR/checkpoint_best.pt \
  --task 'visual_text' \
  --arch visual_text_transformer \
  -s $SRC -t $TRG \
  --save-dir $MODELDIR \
  --target-dict dict.$TRG.txt \
  --validate-interval-updates 1000 \
  --patience 10 \
  --update-freq=1 \
  --image-samples-path ${MODELDIR}/samples \
  --image-samples-interval 10000 \
  --image-embed-type 'visonly' \
  --image-embedding-normalize \
  --image-font-path $FONTPATH \
  --image-font-size $FONTSIZE \
  --image-window $WINDOW \
  --image-stride $STRIDE \
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
chmod 444 $MODELDIR/log

echo "Done training."
echo done > $MODELDIR/status

# evaluate
echo "Starting evaluation..."
for testset in /exp/esalesky/visrep/fairseq-ocr/visual/test-sets/mttt.$SRC-en.$SRC; do
    echo "Evaluating $testset..."
    ref=$(echo $testset | perl -pe "s/.$SRC$/.en/");
    qsub grid_scripts/translate.qsub $MODELDIR $testset $MODELDIR/out.$(basename $testset) $ref
done
