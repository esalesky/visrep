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

EXP_DIR=/expscratch/detter/ocr/visual/zh/vista_maxpool/lmdb/20200310
SRC_PATH=/expscratch/detter/src/Mar2020/fairseq/fairseq-ocr/visual/sentence

TRAIN_DIR=/expscratch/detter/ocr/data/yomdle

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TRAIN_DIR - ${TRAIN_DIR}"
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi

mkdir -p $TMPDIR/ocr/zh
pushd $TMPDIR/ocr/zh
tar xf ${TRAIN_DIR}/zh/lmdb.tar
popd

mkdir -p $EXP_DIR
cd $EXP_DIR

python -u $SRC_PATH/train.py \
--output ${EXP_DIR} \
--train-lmdb \
--train-split train \
--train ${TMPDIR}/ocr/zh/lmdb \
--train-max-image-width 2400 \
--train-min-image-width 1 \
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
--encoder-arch vista_fractional \
--decoder-lstm-units 640 \
--decoder-lstm-layers 3 \
--decoder-lstm-dropout 0.5 \
--batch-size 32 \
--num-workers 8 \
--epochs 125 \
--lr 1e-3



#--image-verbose

echo "COMPLETE"


