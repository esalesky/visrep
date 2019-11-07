#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-25
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N zhfconv \
# -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=2 \
# /expscratch/detter/src/fairseq/fairseq/grid_scripts/train_fconv_zh.sh
#
#
# Train zh Fully Conv model
#
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


SRC_LANG=zh
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr
DATA_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k/raw
CKPT_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k/raw/exp/fullyconv
#
echo $DATA_DIR
echo $FAIRSEQ_PATH
echo $SRC_LANG
echo $TGT_LANG
echo $CKPT_DIR

mkdir -p $CKPT_DIR

python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--arch=fconv_iwslt_de_en \
--lr=0.5 \
--lr-scheduler=fixed \
--clip-norm=0.1 \
--dropout=0.2 \
--max-tokens=4000 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--force-anneal=50 \
--max-epoch=250 \
--num-workers=0 \
--save-dir=$CKPT_DIR \
--raw-text \
--no-epoch-checkpoints # only store last and best checkpoints

