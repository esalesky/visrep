#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@RTX -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N base
#$ -j y -o logs/
# num_proc=16,mem_free=32G,

# Train Transformer model
# -----------------------

module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.1_cuda10.1
module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate ocr

echo $LD_LIBRARY_PATH
echo $PYTHONPATH
echo $CUDA_VISIBLE_DEVICES

nvidia-smi

SRC_LANG=$1
SEG=$2
TGT_LANG=en
TRAIN_TYPE=frozen

LANG_PAIR=${SRC_LANG}-${TGT_LANG}
DATA_DIR=/exp/esalesky/mtocr19/${LANG_PAIR}/data/${SEG}
EXP_DIR=/exp/esalesky/mtocr19/exps/frozen/${SRC_LANG}
CKPT_DIR=${EXP_DIR}/checkpoints/${SEG}
FAIRSEQ_PATH=/exp/esalesky/mtocr19/fairseq-ocr

echo $FAIRSEQ_PATH
echo $DATA_DIR
echo $LANG_PAIR
echo $CKPT_DIR

mkdir -p $EXP_DIR
mkdir -p $CKPT_DIR

python -u $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--arch=transformer_iwslt_de_en \
--save-dir=$CKPT_DIR \
--share-decoder-input-output-embed \
--encoder-embed-path=/exp/esalesky/mtocr19/exps/ocr/$SRC_LANG-$TGT_LANG/checkpoints/embeddings.txt \
--freeze-encoder-embed \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--adam-eps=1e-08 \
--decoder-attention-heads=4 \
--decoder-embed-dim=512 \
--decoder-ffn-embed-dim=1024 \
--decoder-layers=6 \
--dropout=0.3 \
--encoder-attention-heads=4 \
--encoder-embed-dim=512 \
--encoder-ffn-embed-dim=1024 \
--encoder-layers=6 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.2 \
--lr=5e-4 \
--lr-scheduler='inverse_sqrt' \
--max-epoch=100 \
--max-source-positions=1024 \
--max-target-positions=1024 \
--max-tokens=4000 \
--max-tokens-valid=4000 \
--min-loss-scale=0.0001 \
--no-epoch-checkpoints \
--optimizer='adam' \
--raw-text \
--seed 42 \
--share-decoder-input-output-embed \
--warmup-updates=4000 \
--weight-decay=0.0001 \
--update-freq=4 \
--log-format=simple \
--log-interval=10 2>&1 | tee $CKPT_DIR/train.log

# only store last and best checkpoints


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
