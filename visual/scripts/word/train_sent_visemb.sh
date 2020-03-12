#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-02-06
# 
# Train sentence embeddings 
#

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source activate /expscratch/detter/tools/py36
export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH

FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr
TRAIN_DATA=/exp/esalesky/mtocr19/zh-en/data/data-raw/raw/ted_train_en-zh.raw.zh # 170,341
VALID_DATA=/exp/esalesky/mtocr19/zh-en/data/data-raw/raw/ted_dev_en-zh.raw.zh 
EXP_DIR=/expscratch/detter/mt/multitarget-ted/sentemb/zh-en
TRAIN_FONT=/expscratch/detter/fonts/mt/train_zh_font.txt
#TRAIN_FONT=/expscratch/detter/fonts/mt/valid_zh_font.txt
VALID_FONT=/expscratch/detter/fonts/mt/valid_zh_font.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TRAIN_DATA - ${TRAIN_DATA}"
echo "VALID_DATA - ${VALID_DATA}"
echo "TRAIN_FONT - ${TRAIN_FONT}"
echo "VALID_FONT - ${VALID_FONT}"
echo "FAIRSEQ_PATH - ${FAIRSEQ_PATH}"

nvidia-smi

mkdir -p $EXP_DIR

python $FAIRSEQ_PATH/visual/sentence/train.py \
--train $TRAIN_DATA \
--valid $VALID_DATA \
--train-font $TRAIN_FONT \
--valid-font $VALID_FONT \
--batch-size 32 \
--num-workers 4 \
--image-height 30 \
--lr 1e-3 \
--epochs 25 \
--max-seed 500000 \
--output $EXP_DIR \
--augment

#--use-image-cache
#--image-verbose
#--max-image-cache 50000
#--save-images
#--max-cache-write 500

echo "COMPLETE"
