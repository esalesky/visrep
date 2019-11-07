#!/bin/bash
#. /etc/profile.d/modules.sh
#$ -v PATH
#$ -S /bin/bash 
#$ -b y 
#$ -q gpu.q@@1080 
#$ -cwd 
#$ -j y 
#$ -N zhtrans
#$ -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=2
#
#
# 2019-10-25
#
# 
# /expscratch/detter/src/fairseq/fairseq-ocr/grid_scripts/train_trans_zh_pre.sh
#
#$ -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=2
#
# Train de Transformer model
#
# Uses pretrained visual embeddings (see fairseq/visual)
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
CKPT_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k/raw/exp/trans_pre
PRE_TRAIN=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k/raw/exp/visual_embedding/word_embeddings.txt

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
--arch=transformer_iwslt_de_en \
--share-decoder-input-output-embed \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--clip-norm=0.0 \
--lr=5e-4 \
--lr-scheduler=inverse_sqrt \
--warmup-updates=4000 \
--dropout=0.3 \
--weight-decay=0.0001 \
--max-tokens=4000 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--max-epoch=250 \
--num-workers=0 \
--save-dir=$CKPT_DIR \
--raw-text \
--encoder-embed-path=$PRE_TRAIN \
--freeze-enocder-embed \
--no-epoch-checkpoints # only store last and best checkpoints

#--encoder-embed-path=$PRE_TRAIN \
#--num-workers=16 \