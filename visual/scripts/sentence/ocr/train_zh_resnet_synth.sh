#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-03-04
# 
# Train sentence embeddings 
#

module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.1_cuda10.1
module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate /expscratch/detter/tools/anaconda3

EXP_DIR=/expscratch/detter/ocr/visual/zh/resnet/synth/20200310
SRC_PATH=/expscratch/detter/src/Mar2020/fairseq/fairseq-ocr/visual/sentence

#read 100000, included 100000, max 255, min 8, mean 56, median 46.0
#TRAIN_DATA=/exp/ocr/seed/zh/zho-simp-tw_web_2014_100K-sentences.txt 
#TRAIN_DATA=/exp/ocr/seed/zh/zho-trad_newscrawl_2011_1M-sentences_clean.txt
#TRAIN_DATA=/exp/scale18/ocr/data/derived/YOMDLE/final_chinese/truth_line_text.txt
TRAIN_DATA=/exp/ocr/seed/zh/zh_wikipedia.txt

TRAIN_FONT=/exp/ocr/fonts/zh.txt
TRAIN_BACKGROUND=/expscratch/detter/bkg_list.txt
VALID_DIR=/expscratch/detter/ocr/data/yomdle

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TRAIN_DATA - ${TRAIN_DATA}"
echo "TRAIN_FONT - ${TRAIN_FONT}"
echo "TRAIN_BACKGROUND - ${TRAIN_BACKGROUND}"
echo "VALID_DIR - ${VALID_DIR}"
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi

mkdir -p $TMPDIR/ocr/zh
pushd $TMPDIR/ocr/zh
tar xf ${VALID_DIR}/zh/lmdb.tar
popd

mkdir -p $EXP_DIR
cd $EXP_DIR

python -u $SRC_PATH/train.py \
--output ${EXP_DIR} \
--train ${TRAIN_DATA} \
--train-font ${TRAIN_FONT} \
--train-background ${TRAIN_BACKGROUND} \
--train-max-image-width 1000 \
--train-min-image-width 32 \
--train-split-text \
--train-max-seed 5000000 \
--train-max-text-width 50 \
--valid-lmdb \
--valid-split validation \
--valid ${TMPDIR}/ocr/zh/lmdb \
--valid-max-image-width 2400 \
--valid-min-image-width 1 \
--valid-batch-mod 2500 \
--valid-epoch-mod 1 \
--image-height 32 \
--augment \
--encoder-dim 512 \
--encoder-arch resnet18 \
--decoder-lstm-units 640 \
--decoder-lstm-layers 3 \
--decoder-lstm-dropout 0.5 \
--batch-size 32 \
--num-workers 8 \
--epochs 125 \
--lr 1e-3

#\
#--use-font-chars \
#--image-verbose

echo "COMPLETE"
