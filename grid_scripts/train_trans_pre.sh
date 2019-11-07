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

source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH


SRC_LANG=${1} # ko zh ja de fr
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr

# list_include_item "10 11 12" "2"
function list_include_item {
  local list="$1"
  local item="$2"
  if [[ $list =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}

if `list_include_item "ko fr ja" "${SRC_LANG}"` ; then
    SIZE=2.5k
elif `list_include_item "de" "${SRC_LANG}"` ; then
    SIZE=2.5k
elif `list_include_item "zh" "${SRC_LANG}"` ; then
    SIZE=5k
else
    SIZE=2.5k
fi

DATA_DIR=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}/word_embeddings.npz
CKPT_DIR=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}/trans_pre
PRE_TRAIN=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}/norm_word_embeddings.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "source lang - ${SRC_LANG}"
echo "ckpt dir - ${CKPT_DIR}"
echo "pre train - ${PRE_TRAIN}"
echo "data dir - ${DATA_DIR}"
echo "size - ${SIZE}"
echo "fairseq path - ${FAIRSEQ_PATH}"

nvidia-smi

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
--no-epoch-checkpoints 

#--encoder-embed-path=$PRE_TRAIN \
#--num-workers=16 \
#--freeze-enocder-embed \