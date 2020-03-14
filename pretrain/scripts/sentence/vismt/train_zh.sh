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

EXP_DIR=/expscratch/detter/vismt/zh/vista_maxpool/20200310
SRC_PATH=/expscratch/detter/src/Mar2020/fairseq/robust/visual/sentence

TRAIN_DATA=/exp/esalesky/mtocr19/zh-en/data/data-raw/raw/ted_train_en-zh.raw.zh

TRAIN_FONT=/exp/ocr/fonts/zh.txt
TRAIN_BACKGROUND=/expscratch/detter/bkg_list.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TRAIN_DATA - ${TRAIN_DATA}"
echo "TRAIN_FONT - ${TRAIN_FONT}"
echo "TRAIN_BACKGROUND - ${TRAIN_BACKGROUND}"
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi

mkdir -p $EXP_DIR
cd $EXP_DIR

python -u $SRC_PATH/train.py \
--output ${EXP_DIR} \
--train ${TRAIN_DATA} \
--train-font ${TRAIN_FONT} \
--train-background ${TRAIN_BACKGROUND} \
--train-max-image-width 1000 \
--train-min-image-width 32 \
--train-max-seed 5000000 \
--train-max-text-width 5000 \
--train-use-default-image \
--valid ${TRAIN_DATA} \
--valid-font ${TRAIN_FONT} \
--valid-background ${TRAIN_BACKGROUND} \
--valid-max-image-width 1000 \
--valid-min-image-width 32 \
--valid-max-seed 500 \
--valid-max-text-width 5000 \
--valid-use-default-image \
--valid-batch-mod 2500 \
--valid-epoch-mod 1 \
--image-height 32 \
--encoder-dim 512 \
--encoder-arch vista_maxpool \
--decoder-lstm-units 256 \
--decoder-lstm-layers 3 \
--decoder-lstm-dropout 0.5 \
--batch-size 32 \
--num-workers 8 \
--epochs 125 \
--lr 1e-3

#--image-verbose

echo "COMPLETE"

