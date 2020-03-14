#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-03-15
# 
# Train pretrain concat 
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

SRC_LANG=zh 
TGT_LANG=en
SIZE=5k 
DATA_DIR=/exp/esalesky/mtocr19/$SRC_LANG-$TGT_LANG/data/${SIZE}  

FAIRSEQ_PATH=/expscratch/detter/src/Mar2020/fairseq/robust

EXP_DIR=/expscratch/detter/vismt/zh/orig/20200314

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "SRC_LANG - ${SRC_LANG}"
echo "TGT_LANG - ${TGT_LANG}"
echo "SIZE - ${SIZE}"
echo "DATA_DIR - ${DATA_DIR}"
echo "FAIRSEQ_PATH - ${FAIRSEQ_PATH}"
echo "EXP_DIR - ${EXP_DIR}"

nvidia-smi

mkdir -p $EXP_DIR/samples
cd $EXP_DIR

python -u $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--save-dir $EXP_DIR \
--source-lang $SRC_LANG \
--target-lang $TGT_LANG \
--user-dir $FAIRSEQ_PATH \
--task translation \
--arch transformer_iwslt_de_en \
--share-decoder-input-output-embed \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--dropout 0.3 \
--weight-decay 0.0001 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.3 \
--max-epoch 100 \
--num-workers 8 \
--max-sentences 32 \
--raw-text \
--no-epoch-checkpoints
