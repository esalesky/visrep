#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@rtx -cwd 
#$ -l h_rt=120:00:00,gpu=1 
#$ -N update
#$ -j y -o logs/update-avg/
# num_proc=16,mem_free=32G,
#
# 2020-09-15
# 
# Train pretrain update no src loss
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
SEG=${2} #5k or chars
TYPE=${3} #add, avg, concat, visonly

TRAIN_TYPE=update
TGT_LANG=en
LANG_PAIR=${SRC_LANG}-${TGT_LANG}

SRC_PATH=/exp/esalesky/mtocr19
DATA_DIR=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}
EXP_DIR=/exp/esalesky/mtocr19/exps/update-${TYPE}/${SRC_LANG}-${SEG}
TMPDIR=${EXP_DIR}/tmp
OCR_CKPT_PATH=/exp/esalesky/mtocr19/exps/ocr/${SRC_LANG}-${SEG}/checkpoints/model_ckpt_best.pth
CKPT_PATH=${EXP_DIR}/checkpoints/

case ${SRC_LANG} in
  de | fr | en )
    FONT_PATH=/exp/ocr/fonts/all/noto/NotoMono-Regular.ttf
    ;;
  zh | ja | ko )
    FONT_PATH=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf
    ;;
  *)
    echo "no font set for src language ${SRC_LANG} -- check and try again!"
    exit 0
    ;;
esac

case ${SEG} in
  5k )
    WIDTH=64
    ;;
  chars )
    WIDTH=32
    ;;
  words )
    WIDTH=160
    ;;
  *)
    echo "unexpected ${SEG} -- check and try again!"
    exit 0
    ;;
esac


echo "SRC - ${SRC_LANG}"
echo "TYPE - ${TYPE}"
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "DATA_DIR - ${DATA_DIR}"
echo "SRC_PATH - ${SRC_PATH}"
echo "EXP_DIR - ${EXP_DIR}"
echo "OCR_CKPT_PATH - ${OCR_CKPT_PATH}"
echo "CKPT_PATH - ${CKPT_PATH}"

nvidia-smi


mkdir -p ${EXP_DIR}
mkdir -p ${CKPT_PATH}
mkdir -p ${TMPDIR}


python -u ${SRC_PATH}/fairseq-ocr/train_align.py \
${DATA_DIR} \
--user-dir ${SRC_PATH} \
--arch 'vis_align_transformer_iwslt_de_en' \
--image-checkpoint-path ${OCR_CKPT_PATH} \
--image-font-path ${FONT_PATH} \
--save-dir=${CKPT_PATH} \
--image-embed-type ${TYPE} \
--image-embedding-normalize \
--image-height 32 \
--image-width ${WIDTH} \
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
--min-loss-scale 0.0001 \
--num-workers 0 \
--optimizer 'adam' \
--raw-text \
--seed 42 \
--share-decoder-input-output-embed \
--update-freq=8 \
--warmup-updates 2000 \
--max-tokens 2000 \
--max-tokens-valid 2000 \
--weight-decay 0.0001 \
--no-epoch-checkpoints \
--log-format=simple \
--log-interval=10 

#--image-enable-src-loss False \
#--layernorm-embedding \
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
--image-font-path ${FONT_PATH} \
--image-width ${WIDTH} \
--image-embedding-normalize \
--gen-subset=test \
--batch-size=4 \
--raw-text \
--seed=42 \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
--task 'visaligntranslation' 


# -- DEV -- 
python -u ${SRC_PATH}/fairseq-ocr/generate.py \
${DATA_DIR} \
--path=${CKPT_PATH}/checkpoint_best.pt \
--user-dir=${SRC_PATH} \
--image-font-path ${FONT_PATH} \
--image-width ${WIDTH} \
--image-embedding-normalize \
--gen-subset=valid \
--batch-size=4 \
--raw-text \
--seed=42 \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
--task 'visaligntranslation' 




echo "--COMPLETE--"