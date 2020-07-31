#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
# 
# Train pretrain avg e2e 
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

TYPE=avg_e2e
EXP_DIR=/expscratch/detter/vismt/zh/20200730/aligned/chars

SRC_PATH=/home/hltcoe/detter/src/pytorch

DATA_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/dave/zh_char_en_10k

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "DATA_DIR - ${DATA_DIR}"
echo "SRC_PATH - ${SRC_PATH}"
echo "EXP_DIR - ${EXP_DIR}"
echo "TYPE - ${TYPE}"

nvidia-smi


mkdir -p $TMPDIR/vismt/train
pushd $TMPDIR/vismt/train
tar xf ${EXP_DIR}/decode_images_train.tar.gz
popd

mkdir -p $TMPDIR/vismt/valid
pushd $TMPDIR/vismt/valid
tar xf ${EXP_DIR}/decode_images_valid.tar.gz
popd


mkdir -p $EXP_DIR/${TYPE}
cd $EXP_DIR/${TYPE}

python -u ${SRC_PATH}/fairseq-ocr/train_align.py \
${DATA_DIR} \
--user-dir ${SRC_PATH} \
--save-dir ${EXP_DIR}/${TYPE} \
--arch 'vis_align_transformer_iwslt_de_en' \
--image-pretrain-path $TMPDIR/vismt \
--image-samples-path ${EXP_DIR}/${TYPE}/samples \
--image-embed-type 'avg' \
--image-enable-src-loss \
--image-embedding-normalize \
--source-lang 'zh' \
--target-lang 'en' \
--left-pad-source 0 \
--left-pad-target 0 \
--task 'visaligntranslation' \
--train-subset 'train' \
--valid-subset 'valid' \
--criterion 'vis_align_label_smoothed_cross_entropy' \
--adam-betas '(0.9, 0.98)' \
--adam-eps 1e-08 \
--decoder-attention-heads 4 \
--decoder-embed-dim 512 \
--decoder-ffn-embed-dim 1024 \
--decoder-layers 6 \
--dropout 0.3 \
--encoder-attention-heads 4 \
--encoder-embed-dim 512 \
--encoder-ffn-embed-dim 1024 \
--encoder-layers 6 \
--label-smoothing 0.1 \
--lr 5e-4 \
--lr-scheduler 'inverse_sqrt' \
--max-epoch 100 \
--max-source-positions 1024 \
--max-target-positions 1024 \
--max-tokens 2000 \
--max-tokens-valid 2000 \
--min-loss-scale 0.0001 \
--no-epoch-checkpoints \
--num-workers 8 \
--optimizer 'adam' \
--raw-text \
--share-decoder-input-output-embed \
--warmup-updates 4000 \
--weight-decay 0.0001 \
--update-freq=2

#--layernorm-embedding \
#--image-embedding-normalize \
#--no-token-positional-embeddings \