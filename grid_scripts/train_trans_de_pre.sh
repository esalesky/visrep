#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-16
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N depre \
# -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=2 \
# /expscratch/detter/src/fairseq/fairseq/grid_scripts/train_trans_de_pre.sh
#
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


SRC_LANG=de
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq
DATA_DIR=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/raw
CKPT_DIR=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/exp/fairseq/pretrain
PRE_TRAIN=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/10000/exp/fairseq/visual_embedding/word_embeddings.txt

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
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--max-tokens=4096 \
--save-dir=$CKPT_DIR \
--encoder-embed-dim=512 \
--dataset-impl=raw \
--no-epoch-checkpoints \
--max-tokens=2048 \
--encoder-embed-path=$PRE_TRAIN \
--num-workers=0

#--num-workers=16 \