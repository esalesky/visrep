#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@rtx -cwd 
#$ -l h_rt=2:00:00,gpu=1 
#$ -N generate
#$ -j y -o logs/generate/
# num_proc=16,mem_free=32G,
#
# 2020-09-15
# 
# Score
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


cd $EXP_DIR

# -- TEST --
python -u ${SRC_PATH}/fairseq-ocr/generate.py \
${DATA_DIR} \
--path=${CKPT_PATH}/checkpoint_best.pt \
--user-dir=${SRC_PATH} \
--image-font-path ${FONT_PATH} \
--image-width $WIDTH \
--gen-subset=test \
--batch-size=4 \
--raw-text \
--beam=5 \
--seed=42 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} \
--task 'visaligntranslation' 

echo "--COMPLETE--"
