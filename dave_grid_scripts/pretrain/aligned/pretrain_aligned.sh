#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@RTX -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N pretrain
#$ -j y -o /exp/esalesky/mtocr19/exps/ocr/logs/
# num_proc=16,mem_free=32G,
#
# 2020-07-30
# 
# Train aligned embeddings 
# -----------------------

set -eu

# module load cuda10.1/toolkit/10.1.105
# module load cudnn/7.6.1_cuda10.1
# module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

# conda deactivate; conda deactivate
# conda activate fairseq

# SRC_LANG \in de
SRC_LANG=${1}
# SEG \in char 2.5k 5k 10k 15k 20k
SEG=${2}

TGT_LANG=en
LANG_PAIR=${SRC_LANG}-${TGT_LANG}

FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr
EXP_DIR=./${SRC_LANG}-${SEG}.7layers/
TMPDIR=/expscratch/mpost
CKPT_PATH=${EXP_DIR}/checkpoints/model_ckpt_best.pth

TRAIN_DATA=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}/train.${LANG_PAIR}.${SRC_LANG}
VALID_DATA=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}/valid.${LANG_PAIR}.${SRC_LANG}
DICT=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}/dict.${SRC_LANG}.txt

case ${SRC_LANG} in
  de | fr | en )
    TRAIN_FONT=/exp/ocr/fonts/all/noto/NotoMono-Regular.ttf
    ;;
  zh | ja | ko )
    TRAIN_FONT=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf
    ;;
  *)
    echo "no font set for src language ${SRC_LANG} -- check and try again!"
    exit 0
    ;;
esac
VALID_FONT=${TRAIN_FONT}

case ${SEG} in
  5k )
    WIDTH=32
    ;;
  chars )
    WIDTH=16
    ;;
  words )
    WIDTH=80
    ;;
  *)
    echo "unexpected ${SEG} -- check and try again!"
    exit 0
    ;;
esac

echo "FONT - ${TRAIN_FONT}"
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "DICT - ${DICT}"
echo "TRAIN_DATA - ${TRAIN_DATA}"
echo "VALID_DATA - ${VALID_DATA}"
echo "TRAIN_FONT - ${TRAIN_FONT}"
echo "VALID_FONT - ${VALID_FONT}"

hostname
nvidia-smi

mkdir -p $EXP_DIR

PYTHONPATH=$FAIRSEQ python -u $FAIRSEQ/visual/aligned/train.py \
--output ${EXP_DIR} \
--dict ${DICT} \
--train ${TRAIN_DATA} \
--train-font ${TRAIN_FONT} \
--valid ${VALID_DATA} \
--valid-font ${TRAIN_FONT} \
--image-height 16 \
--image-width ${WIDTH} \
--train-max-text-width 35 \
--encoder-embed-dim 512 \
--font-size 8 \
--batch-size 32 \
--num-workers 8 \
--epochs 2 \
--lr 1e-3 

echo "COMPLETE"

#--write-image-samples \
