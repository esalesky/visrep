#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@2080 -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N vismt-pretrain-update
#$ -j y -o logs/
# num_proc=16,mem_free=32G,
#
# 2020-09-15
# 
# Train pretrain update
#

module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.1_cuda10.1
module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate ocr

SRC_LANG=${1}
TYPE=${2}

SEG=5k
TRAIN_TYPE=update
TGT_LANG=en
LANG_PAIR=${SRC_LANG}-${TGT_LANG}

SRC_PATH=/exp/esalesky/mtocr19
DATA_DIR=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}
EXP_DIR=/exp/esalesky/mtocr19/exps/aligned/${SRC_LANG}
TMPDIR=${EXP_DIR}/${SRC_LANG}-${TRAIN_TYPE}-${TYPE}-tmp
IMGTMP=${EXP_DIR}/tmp
IMG_CKPT_PATH=${EXP_DIR}/checkpoints/model_ckpt_best.pth
CKPT_PATH=${EXP_DIR}/checkpoints/pretrain-${TRAIN_TYPE}-${TYPE}

echo "SRC - ${SRC_LANG}"
echo "TYPE - ${TYPE}"
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "DATA_DIR - ${DATA_DIR}"
echo "SRC_PATH - ${SRC_PATH}"
echo "EXP_DIR - ${EXP_DIR}"
echo "IMG_CKPT_PATH - ${IMG_CKPT_PATH}"
echo "CKPT_PATH - ${CKPT_PATH}"

nvidia-smi

mkdir -p ${EXP_DIR}
mkdir -p ${CKPT_PATH}
mkdir -p ${TMPDIR}
cd ${TMPDIR}

mkdir -p ${TMPDIR}/vismt/train
pushd ${TMPDIR}/vismt/train
ln -s ${IMGTMP}/train/embeddings/images .
#tar --extract --skip-old-files --file=${IMGTMP}/decode_images_train.tar.gz
popd

mkdir -p ${TMPDIR}/vismt/valid
pushd ${TMPDIR}/vismt/valid
ln -s ${IMGTMP}/valid/embeddings/images .
#tar --extract --skip-old-files --file=${IMGTMP}/decode_images_valid.tar.gz
popd

mkdir -p ${TMPDIR}/vismt/test
pushd ${TMPDIR}/vismt/test
ln -s ${IMGTMP}/test/embeddings/images .
#tar --extract --skip-old-files --file=${IMGTMP}/decode_images_test.tar.gz
popd


python -u ${SRC_PATH}/fairseq-ocr/train_align.py \
${DATA_DIR} \
--user-dir ${SRC_PATH} \
--arch 'vis_align_transformer_iwslt_de_en' \
--image-pretrain-path ${TMPDIR}/vismt \
--image-checkpoint-path ${IMG_CKPT_PATH} \
--save-dir=${CKPT_PATH} \
--image-embed-type ${TYPE} \
--image-samples-path ${TMPDIR}/samples \
--image-embedding-normalize \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
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
--label-smoothing 0.2 \
--lr 5e-4 \
--lr-scheduler 'inverse_sqrt' \
--max-epoch 100 \
--max-source-positions 1024 \
--max-target-positions 1024 \
--max-tokens 4000 \
--max-tokens-valid 4000 \
--min-loss-scale 0.0001 \
--num-workers 0 \
--optimizer 'adam' \
--raw-text \
--share-decoder-input-output-embed \
--warmup-updates 4000 \
--weight-decay 0.0001 \
--update-freq=8 \
--no-epoch-checkpoints \
--log-format=simple \
--log-interval=10 2>&1 | tee ${CKPT_PATH}/${TRAIN_TYPE}-${TYPE}-train.log

#--layernorm-embedding \
#--image-embedding-normalize \
#--no-token-positional-embeddings \

# -----
# SCORE
# -----

wait
echo "-- SCORE TIME --"

cd $EXP_DIR

# -- TEST --
python -u ${SRC_PATH}/fairseq-ocr/generate.py \
${DATA_DIR} \
--path=${CKPT_PATH}/checkpoint_best.pt \
--user-dir=${SRC_PATH} \
--gen-subset=test \
--batch-size=4 \
--raw-text \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
--task 'visaligntranslation' \
--image-pretrain-path ${TMPDIR}/vismt

#--image-font-path ${FONT_PATH} \

# -- DEV -- 
python -u ${SRC_PATH}/fairseq-ocr/generate.py \
${DATA_DIR} \
--path=${CKPT_PATH}/checkpoint_best.pt \
--user-dir=${SRC_PATH} \
--gen-subset=valid \
--batch-size=4 \
--raw-text \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
--task 'visaligntranslation' \
--image-pretrain-path ${TMPDIR}/vismt

#--image-font-path ${FONT_PATH} \


echo "--COMPLETE--"
