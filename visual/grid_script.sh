#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-17
#
# Train word embeddings
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N wordemb -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 /expscratch/detter/src/fairseq-ocr/visual/grid_script.sh
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

EXP_DIR=/expscratch/detter/mt/visual3
DICT=/exp/mpost/mtocr19/data/mttt/de-en/fairseq_binary/dict.de.txt
TRAIN_FONT=/expscratch/detter/fonts/mt/train_font.txt
VALID_FONT=/expscratch/detter/fonts/mt/valid_font.txt
FAIRSEQ=/expscratch/detter/src/fairseq-ocr

echo $EXP_DIR
echo $DICT
echo $TRAIN_FONT
echo $VALID_FONT
echo $FAIRSEQ

mkdir -p $EXPDIR

python $FAIRSEQ/visual/train.py \
--input=$DICT \
--font=$TRAIN_FONT \
--valid_font=$VALID_FONT \
--output=$EXP_DIR


echo "COMPLETE"
