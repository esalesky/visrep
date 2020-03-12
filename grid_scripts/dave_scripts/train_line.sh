#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-03-04
# 
# Train sentence embeddings 
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

SRC_LANG=zh #${1} # ko zh ja de fr
TGT_LANG=en
SIZE=5k #5k #${2}
DATA_DIR=/exp/esalesky/mtocr19/$SRC_LANG-$TGT_LANG/data/${SIZE} #/dict.$SRC_LANG.txt 
FONT_FILE=/expscratch/detter/fonts/mt/${SRC_LANG}.txt

FAIRSEQ_PATH=/expscratch/detter/src/Mar2020/fairseq/robust
PRETRAIN_PATH=/expscratch/detter/vismt/zh/vista_maxpool/20200310

EXP_DIR=/expscratch/detter/vismt/zh/vista_maxpool/20200310/fairseq

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "SRC_LANG - ${SRC_LANG}"
echo "TGT_LANG - ${TGT_LANG}"
echo "SIZE - ${SIZE}"
echo "DATA_DIR - ${DATA_DIR}"
echo "FONT_FILE - ${FONT_FILE}"
echo "FAIRSEQ_PATH - ${FAIRSEQ_PATH}"
echo "EXP_DIR - ${EXP_DIR}"
echo "PRETRAIN_PATH - ${PRETRAIN_PATH}"

nvidia-smi

mkdir -p $EXP_DIR
cd $EXP_DIR

mkdir -p $TMPDIR/vismt
pushd $TMPDIR/vismt
tar xf ${PRETRAIN_PATH}/decode_embeddings.tar.gz
popd

python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--save-dir $EXP_DIR \
--source-lang $SRC_LANG \
--target-lang $TGT_LANG \
--user-dir $FAIRSEQ_PATH \
--task visualmt \
--arch visual_transformer_iwslt_de_en \
--image-type line \
--image-pretrain-path $TMPDIR/vismt/embeddings \
--image-vista-kernel-size 2 \
--image-vista-width 0.7 \
--image-font-path $FONT_FILE \
--image-font-size 6 \
--image-pad-right 5 \
--image-samples-path $EXP_DIR \
--image-height 32 \
--image-width 32 \
--image-src-loss-scale 1.0 \
--image-tgt-loss-scale 1.0 \
--image-embed-type 'concat' \
--image-embed-dim 512 \
--image-use-bridge \
--image-layer 'avgpool' \
--image-backbone "vista" \
--share-decoder-input-output-embed \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 \
--lr 5e-4 \
--lr-scheduler inverse_sqrt \
--warmup-updates 4000 \
--dropout 0.3 \
--weight-decay 0.0001 \
--criterion visual_label_smoothed_cross_entropy \
--label-smoothing 0.3 \
--max-epoch 100 \
--num-workers 0 \
--max-sentences 4 \
--raw-text \
--no-epoch-checkpoints \
--image-use-cache \
--image-verbose

#--max-sentences 4 \
#--image-use-cache \
#--max-tokens=600 \
#--update-freq=8 \
