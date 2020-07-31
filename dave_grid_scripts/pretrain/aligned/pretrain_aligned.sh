#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
# 
# Train aligned embeddings 
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

EXP_DIR=/expscratch/detter/vismt/zh/20200730/aligned/chars

SRC_PATH=/home/hltcoe/detter/src/pytorch

TRAIN_DATA=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k/train.zh-en.zh
VALID_DATA=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k/test.zh-en.zh
DICT=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k/dict.zh.txt

#TRAIN_DATA=/exp/esalesky/mtocr19/zh-en/data/10k/train.zh-en.zh
#VALID_DATA=/exp/esalesky/mtocr19/zh-en/data/10k/valid.zh-en.zh
#DICT=/exp/esalesky/mtocr19/zh-en/data/10k/dict.zh.txt

#TRAIN_DATA=/exp/esalesky/mtocr19/zh-en/data/chars/train.zh-en.zh
#VALID_DATA=/exp/esalesky/mtocr19/zh-en/data/chars/valid.zh-en.zh
#DICT=/exp/esalesky/mtocr19/zh-en/data/chars/dict.zh.txt

TRAIN_FONT=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf
VALID_FONT=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "DICT - ${DICT}"
echo "TRAIN_DATA - ${TRAIN_DATA}"
echo "VALID_DATA - ${VALID_DATA}"
echo "TRAIN_FONT - ${TRAIN_FONT}"
echo "VALID_FONT - ${VALID_FONT}"
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi

mkdir -p $EXP_DIR
cd $EXP_DIR

python -u ${SRC_PATH}/fairseq-ocr/visual/aligned/train.py \
--output ${EXP_DIR} \
--dict ${DICT} \
--train ${TRAIN_DATA} \
--train-font ${TRAIN_FONT} \
--valid ${VALID_DATA} \
--valid-font ${TRAIN_FONT} \
--image-height 32 \
--image-width 32 \
--train-max-text-width 40 \
--font-size 16 \
--batch-size 64 \
--num-workers 8 \
--epochs 2 \
--lr 1e-3 \
--write-image-samples

echo "COMPLETE"

