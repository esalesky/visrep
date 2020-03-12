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
CKPT_PATH=/expscratch/detter/vismt/zh/vista_maxpool/20200310/checkpoints/model_ckpt_best.pth
SRC_PATH=/expscratch/detter/src/Mar2020/fairseq/fairseq-ocr/visual/sentence

TEST_DATA=/exp/esalesky/mtocr19/zh-en/data/data-raw/raw/ted_train_en-zh.raw.zh
TEST_FONT=/exp/ocr/fonts/zh.txt
TEST_BACKGROUND=/expscratch/detter/bkg_list.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TEST_DATA - ${TEST_DATA}"
echo "TEST_FONT - ${TEST_FONT}"
echo "TEST_BACKGROUND - ${TEST_BACKGROUND}"
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi

mkdir -p $EXP_DIR
cd $EXP_DIR

mkdir -p ${TMPDIR}/vismt

python -u $SRC_PATH/decode_synth.py \
--output ${TMPDIR}/vismt \
--load-checkpoint-pat ${CKPT_PATH} \
--test ${TEST_DATA} \
--test-font ${TEST_FONT} \
--test-background ${TEST_BACKGROUND} \
--test-max-image-width 1000 \
--test-min-image-width 32 \
--test-use-default-image \
--image-height 32 \
--encoder-dim 512 \
--encoder-arch vista_maxpool \
--decoder-lstm-units 256 \
--decoder-lstm-layers 3 \
--decoder-lstm-dropout 0.5 \
--batch-size 32 \
--num-workers 0

#--write-images \
#--image-verbose

tar -cf ${TMPDIR}/vismt/decode_embeddings.tar.gz -C ${TMPDIR}/vismt embeddings
cp ${TMPDIR}/vismt/decode_embeddings.tar.gz ${EXP_DIR}

echo "COMPLETE"

