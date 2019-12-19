#!/bin/bash
#. /etc/profile.d/modules.sh
#
#$ -S /bin/bash -q gpu.q@@2080 -cwd 
#$ -l h_rt=48:00:00,gpu=1 
#$ -N pretrain_update
# num_proc=16,mem_free=32G,

# Train Transformer model
# -----------------------

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source activate /home/hltcoe/esalesky/anaconda3/envs/fs

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
echo $CUDA_VISIBLE_DEVICES
nvidia-smi
hostname


SRC_LANG=$1
TGT_LANG=en
SEGM=$2
CKPT_DIR=$3
DATA_DIR=$SRC_LANG-$TGT_LANG/data/$SEGM
FAIRSEQ_PATH=/exp/esalesky/mtocr19/fairseq

echo "data dir: $DATA_DIR"
echo "fairseq: $FAIRSEQ_PATH"
echo "src lang: $SRC_LANG"
echo "tgt lang: $TGT_LANG"
echo "ckpt dir: $CKPT_DIR"
echo "segments: $SEGM"

mkdir -p $CKPT_DIR

python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--arch=transformer_iwslt_de_en \
--share-decoder-input-output-embed \
--encoder-embed-path=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/$SEGM/norm_word_embeddings.txt \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--clip-norm=0.0 \
--lr=5e-4 \
--lr-scheduler=inverse_sqrt \
--warmup-updates=4000 \
--dropout=0.3 \
--weight-decay=0.0001 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.3 \
--max-epoch=100 \
--num-workers=0 \
--save-dir=$CKPT_DIR \
--raw-text \
--no-epoch-checkpoints \
--log-format=simple \
--max-tokens=4000 \
--update-freq=8 \
--log-interval=10 2>&1 | tee $CKPT_DIR/train.log

# only store last and best checkpoints


#--score!!--
PAIR=$SRC_LANG-$TGT_LANG

#dev
python /exp/esalesky/mtocr19/fairseq-ocr/generate.py /exp/esalesky/mtocr19/$PAIR/data/$SEGM --path $CKPT_DIR/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe=sentencepiece --dataset-impl raw --sacrebleu --quiet --gen-subset valid
echo "dev scored"
#test
python /exp/esalesky/mtocr19/fairseq-ocr/generate.py /exp/esalesky/mtocr19/$PAIR/data/$SEGM --path $CKPT_DIR/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe=sentencepiece --dataset-impl raw --sacrebleu --quiet --gen-subset test
echo "test scored"
