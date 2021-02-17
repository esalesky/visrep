#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@RTX -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N frozen.5layers
#$ -j y -o logs/frozen.layers/
# num_proc=16,mem_free=32G,

# Train Transformer model
# -----------------------

set -eu

# module load cuda10.1/toolkit/10.1.105
# module load cudnn/7.6.1_cuda10.1
# module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

# source deactivate
# source activate fairseq

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
EXP_DIR=./${SRC_LANG}-${SEG}.7layers
CKPT_DIR=${EXP_DIR}/checkpoints/${SEG}
FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr

echo $FAIRSEQ
echo $DATA_DIR
echo $LANG_PAIR
echo $CKPT_DIR

mkdir -p $EXP_DIR
mkdir -p $CKPT_DIR

PYTHONPATH=$FAIRSEQ python -u -m fairseq_cli.train \
$DATA_DIR \
--seed 42 \
--validate-interval-updates 1000 \
--patience 10 \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ \
--arch=transformer_iwslt_de_en \
--save-dir $CKPT_DIR \
--share-decoder-input-output-embed \
--encoder-embed-path=${EXP_DIR}/checkpoints/norm_embeddings.txt \
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
--encoder-normalize-before \
--encoder-layers=6 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.2 \
--lr=5e-4 \
--lr-scheduler='inverse_sqrt' \
--max-epoch=100 \
--max-source-positions=1024 \
--max-target-positions=1024 \
--max-tokens 4000 \
--max-tokens-valid=16000 \
--min-loss-scale=0.0001 \
--optimizer='adam' \
--dataset-impl raw \
--share-decoder-input-output-embed \
--warmup-updates=4000 \
--weight-decay=0.0001 \
--log-format json \
--log-interval 10 > $CKPT_DIR/train.log 2>&1

#--encoder-embed-path=/exp/esalesky/mtocr19/exps/ocr/$SRC_LANG-$SEG/checkpoints/embeddings.txt \
#--encoder-embed-path=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/$SEG/norm_word_embeddings.txt \


# -----
# SCORE
# -----

wait
echo "-- SCORE TIME --"

#cd $EXP_DIR

# -- TEST --
PYTHONPATH=$FAIRSEQ python -um fairseq_cli.generate \
  ${DATA_DIR} \
  --path=${CKPT_DIR}/checkpoint_best.pt \
  --gen-subset=test \
  --batch-size=4 \
  --dataset-impl raw \
  --beam=5 \
  -s ${SRC_LANG} \
  -t ${TGT_LANG} \
> $EXP_DIR/test.out

# -- DEV --
PYTHONPATH=$FAIRSEQ python -um fairseq_cli.generate \
  ${DATA_DIR} \
  --path=${CKPT_DIR}/checkpoint_best.pt \
  --gen-subset=valid \
  --batch-size=4 \
  --dataset-impl raw \
  --beam=5 \
  --source-lang ${SRC_LANG} \
  --target-lang ${TGT_LANG} \
> $EXP_DIR/valid.out

echo "--COMPLETE--"

