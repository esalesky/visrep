#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-23
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N dewordext \
#   -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 \
#   /expscratch/detter/src/fairseq/fairseq/visual/extract_embedding.sh
#
# Extract word embeddings
#

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
echo $CUDA_VISIBLE_DEVICES
nvidia-smi


SRC_LANG=de
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq
DICT=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/raw/dict.de.txt
MODEL=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/exp/fairseq/visual_embedding/checkpoints/model.pth
TEST_FONT=/expscratch/detter/fonts/unifont.txt

EXP_DIR=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/exp/fairseq/visual_embedding



echo $EXP_DIR
echo $DICT
echo $TEST_FONT
echo $MODEL
echo $FAIRSEQ_PATH

mkdir -p $EXP_DIR

python $FAIRSEQ_PATH/visual/get_embeddings.py \
--model-path=$MODEL \
--input=$DICT \
--font=$TEST_FONT \
--output=$EXP_DIR

echo "COMPLETE"
