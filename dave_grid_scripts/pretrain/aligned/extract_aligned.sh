#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@2080 -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N extract
#$ -j y -o /exp/esalesky/mtocr19/exps/ocr/logs/
# num_proc=16,mem_free=32G,
#
# 2020-07-30
# 
# Extract aligned embeddings 
#

set -eu

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

SRC_LANG=${1}
SEG=${2}

TGT_LANG=en
LANG_PAIR=${SRC_LANG}-${TGT_LANG}

FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr
EXP_DIR=./${SRC_LANG}-${SEG}.7layers/
TMPDIR=/expscratch/mpost
CKPT_PATH=${EXP_DIR}/checkpoints/model_ckpt_best.pth

mkdir -p $EXP_DIR
mkdir -p $TMPDIR

DICT=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}/dict.${SRC_LANG}.txt
VOCAB=${TMPDIR}/vocab
cut -d" " -f1 ${DICT} > ${VOCAB}

case ${SRC_LANG} in
  de | fr | en )
    EXTRACT_FONT=/exp/ocr/fonts/all/noto/NotoMono-Regular.ttf
    ;;
  zh | ja | ko )
    EXTRACT_FONT=/exp/ocr/fonts/all/noto_zh/NotoSansCJKjp-hinted/NotoSansCJKjp-Regular.otf
    ;;
  *)
    echo "no font set for src language ${SRC_LANG} -- check and try again!"
    exit 0
    ;;
esac

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

echo "------"
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "TMPDIR - ${TMPDIR}"
echo "EXP_DIR - ${EXP_DIR}"
echo "DICT - ${DICT}"
echo "EXTRACT_FONT - ${EXTRACT_FONT}"
echo "FAIRSEQ - ${FAIRSEQ}"
echo "CKPT_PATH - ${CKPT_PATH}"

hostname
nvidia-smi

PYTHONPATH=$FAIRSEQ python -u $FAIRSEQ/visual/aligned/decode.py \
--output ${EXP_DIR}/checkpoints \
--dict ${DICT} \
--test ${VOCAB} \
--test-font ${EXTRACT_FONT} \
--image-height 16 \
--image-width ${WIDTH} \
--encoder-embed-dim 512 \
--font-size 8 \
--num-workers 0 \
--load-checkpoint-path ${CKPT_PATH}


echo "-- COMPLETE --"
