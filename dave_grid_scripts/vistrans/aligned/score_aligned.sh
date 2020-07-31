#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
# 
# Score pretrain align
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

TYPE=concat # concat add avg tokonly
EXP_DIR=/expscratch/detter/vismt/zh/20200730/aligned/chars

SRC_PATH=/home/hltcoe/detter/src/pytorch

DATA_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "DATA_DIR - ${DATA_DIR}"
echo "SRC_PATH - ${SRC_PATH}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TYPE - ${TYPE}"

nvidia-smi

mkdir -p $TMPDIR/vismt/test
pushd $TMPDIR/vismt/test
tar xf ${EXP_DIR}/decode_images_test.tar.gz
popd

cd $EXP_DIR

python -u ${SRC_PATH}/fairseq-ocr/generate.py \
${DATA_DIR} \
--path=${EXP_DIR}/${TYPE}/checkpoint_last.pt \
--user-dir=${SRC_PATH} \
--gen-subset=test \
--batch-size=4 \
--raw-text \
--beam=5 \
--source-lang 'zh' \
--target-lang 'en' \
--task 'visaligntranslation' \
--image-pretrain-path ${TMPDIR}/vismt

#--image-font-path ${FONT_PATH} \


echo "COMPLETE"