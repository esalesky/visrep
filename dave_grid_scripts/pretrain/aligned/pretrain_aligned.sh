#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@2080 -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N pretrain
# num_proc=16,mem_free=32G,
#
# 2020-07-30
# 
# Train aligned embeddings 

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
SEG=5k

TGT_LANG=en
LANG_PAIR=${SRC_LANG}-${TGT_LANG}

SRC_PATH=/exp/esalesky/mtocr19
EXP_DIR=/exp/esalesky/mtocr19/exps/aligned/${SRC_LANG}/
TMPDIR=${EXP_DIR}/tmp
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
echo "SRC_PATH - ${SRC_PATH}"

nvidia-smi


mkdir -p $EXP_DIR
cd $EXP_DIR

python -u ${SRC_PATH}/fairseq-ocr/visual/aligned/train.py \
--output ${EXP_DIR} \
--dict ${DICT} \
--train ${TRAIN_DATA} \
--train-font ${TRAIN_FONT} \
--valid ${VALID_DATA} \
--valid-font ${TRAIN_FONT} \
--image-height 32 \
--image-width 32 \
--train-max-text-width 40 \
--font-size 16 \
--batch-size 64 \
--num-workers 8 \
--epochs 2 \
--lr 1e-3 

echo "COMPLETE"
