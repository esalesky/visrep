#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
# 
# Extract aligned embeddings 
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
CKPT_PATH=/expscratch/detter/vismt/zh/20200730/aligned/chars/checkpoints/model_ckpt_best.pth

SRC_PATH=/home/hltcoe/detter/src/pytorch

DATA_TYPE=test #train (170341) valid (1958) test (1982)
TEST_DATA=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k/${DATA_TYPE}.zh-en.zh
DICT=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k/dict.zh.txt

#DATA_TYPE=train #train valid test
#TEST_DATA=/exp/esalesky/mtocr19/zh-en/data/chars/${DATA_TYPE}.zh-en.zh
#DICT=/exp/esalesky/mtocr19/zh-en/data/chars/dict.zh.txt

TEST_FONT=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "DICT - ${DICT}"
echo "TEST_DATA - ${TEST_DATA}"
echo "TEST_FONT - ${TEST_FONT}"
echo "SRC_PATH - ${SRC_PATH}"
echo "CKPT_PATH - ${CKPT_PATH}"


nvidia-smi

mkdir -p ${TMPDIR}/${DATA_TYPE}

mkdir -p $EXP_DIR
cd $EXP_DIR

python -u ${SRC_PATH}/fairseq-ocr/visual/aligned/decode.py \
--output ${TMPDIR}/${DATA_TYPE} \
--dict ${DICT} \
--test ${TEST_DATA} \
--test-font ${TEST_FONT} \
--image-height 32 \
--image-width 32 \
--font-size 14 \
--batch-size 1 \
--num-workers 0 \
--write-image-samples \
--write-metadata \
--load-checkpoint-path ${CKPT_PATH} 

#\
#--image-verbose

tar -cf ${TMPDIR}/decode_embeddings_${DATA_TYPE}.tar.gz -C ${TMPDIR}/${DATA_TYPE}/embeddings encoder
tar -cf ${TMPDIR}/decode_images_${DATA_TYPE}.tar.gz -C ${TMPDIR}/${DATA_TYPE}/embeddings images

cp ${TMPDIR}/decode_embeddings_${DATA_TYPE}.tar.gz ${EXP_DIR}
cp ${TMPDIR}/decode_images_${DATA_TYPE}.tar.gz ${EXP_DIR}

echo "COMPLETE"

