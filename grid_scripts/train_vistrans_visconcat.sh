#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-11-07
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N koscr \
# -l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
# /expscratch/detter/src/fairseq/fairseq-ocr/grid_scripts/train_trans_pre.sh SRC_LANG [ko zh ja de fr]
#
# Train transformer with pre calc embeddings
#

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH


SRC_LANG=ko #${1} # ko zh ja de fr
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr
SIZE=5k #${2}

DATA_DIR=/exp/esalesky/mtocr19/$SRC_LANG-$TGT_LANG/data/${SIZE} #/dict.$SRC_LANG.txt 

CKPT_DIR=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}/vis_trans/visconcat
FONT_FILE=/expscratch/detter/fonts/mt/test_${SRC_LANG}_font.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "source lang - ${SRC_LANG}"
echo "ckpt dir - ${CKPT_DIR}"
echo "data dir - ${DATA_DIR}"
echo "size - ${SIZE}"
echo "font file - ${FONT_FILE}"
echo "fairseq path - ${FAIRSEQ_PATH}"

nvidia-smi

mkdir -p $CKPT_DIR

python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--task=visualmt \
--arch=visual_transformer_iwslt_de_en \
--image-type=Word \
--image-font-path=$FONT_FILE \
--image-samples-path=$CKPT_DIR \
--image-use-cache \
--image-augment \
--image-height=32 \
--image-width=64 \
--image-layer='avgpool' \
--image-src-loss-scale=1.0 \
--image-tgt-loss-scale=1.0 \
--image-embed-type='concat' \
--image-embed-dim=512 \
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
--criterion=visual_label_smoothed_cross_entropy \
--label-smoothing=0.3 \
--max-epoch=100 \
--update-freq=8 \
--num-workers=0 \
--save-dir=$CKPT_DIR \
--raw-text \
--no-epoch-checkpoints 

# --image-verbose \
# --image-enable \