#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu_short.q -cwd 
#$ -l h_rt=0:59:00,gpu=1 
#$ -N score
#$ -j y -o logs/
# num_proc=16,mem_free=32G,

# Score baseline Transformer model
# -----------------------

module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.1_cuda10.1
module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source activate /home/hltcoe/esalesky/anaconda3/envs/fs

SRC_LANG=$1
SEG=$2
TGT_LANG=en
TRAIN_TYPE=baseline

LANG_PAIR=${SRC_LANG}-${TGT_LANG}
DATA_DIR=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}
EXP_DIR=/exp/esalesky/mtocr19/exps/baseline/${SRC_LANG}
TMPDIR=${EXP_DIR}/${SRC_LANG}-baseline-tmp
CKPT_DIR=${EXP_DIR}/checkpoints/baseline-${SEG}
FAIRSEQ_PATH=/exp/esalesky/mtocr19/fairseq

echo "${LANG_PAIR} - ${SEG}"
echo $CUDA_VISIBLE_DEVICES

nvidia-smi

# -----
# SCORE
# -----

wait
echo "-- SCORE TIME --"

cd $EXP_DIR

# -- TEST --
python -u ${FAIRSEQ_PATH}/generate.py \
${DATA_DIR} \
--path=${CKPT_DIR}/checkpoint_best.pt \
--gen-subset=test \
--batch-size=4 \
--raw-text \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} 

# -- DEV -- 
python -u ${FAIRSEQ_PATH}/generate.py \
${DATA_DIR} \
--path=${CKPT_DIR}/checkpoint_best.pt \
--gen-subset=valid \
--batch-size=4 \
--raw-text \
--beam=5 \
--source-lang ${SRC_LANG} \
--target-lang ${TGT_LANG} 

echo "--COMPLETE--"
