#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-11-07
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N koscr \
# -l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
# /expscratch/detter/src/fairseq/fairseq-ocr/visual/scripts/score_embedding.sh ko
#
# Score visual embeddings
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
EXP_DIR=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "source lang - ${SRC_LANG}"
echo "exp dir - ${EXP_DIR}"
echo "data dir - ${DATA_DIR}"
echo "size - ${SIZE}"
echo "fairseq path - ${FAIRSEQ_PATH}"

nvidia-smi

mkdir -p $EXP_DIR


python $FAIRSEQ_PATH/visual/score_embeddings.py \
--input=$DATA_DIR \
--output=$EXP_DIR

echo "COMPLETE"
