#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-09-03
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N transedge1 -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 /expscratch/detter/src/fairseq-ocr/grid_scripts/train_visualmt_robusttrans.sh
#
#
# Train Robust Transformer model (character-aware model)
#

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH
echo $CUDA_VISIBLE_DEVICES
nvidia-smi


DATA_DIR=/expscratch/detter/mt/robust/iwslt2016/en-fr/char_new/preprocessed_fairseq-robust_n0.0
FAIRSEQ_PATH=/expscratch/detter/src/fairseq-ocr
ROBUST_EMB=/expscratch/detter/mt/robust/iwslt2016/en-fr/char_new/preprocessed_fairseq-robust_n0.0/char_img_emb.pt
SRC_LANG=en
TGT_LANG=fr

CKPT_DIR=/expscratch/detter/mt/exp/robust_trans/ckpt

echo $DATA_DIR
echo $SRC_LANG
echo $TGT_LANG
echo $CKPT_DIR
echo $FAIRSEQ_PATH
echo $ROBUST_EMB

mkdir -p $CKPT_DIR


python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--arch=visual_edge_robust_transformer  \
--num-source-feats=20 \
--num-target-feats=1 \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--lr-scheduler=inverse_sqrt \
--warmup-init-lr='1e-07' \
--warmup-updates=8000 \
--lr=0.0007 \
--min-lr='1e-09' \
--dropout=0.1 \
--weight-decay=0.0001 \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--max-tokens=3000 \
--seed=200 \
--max-epoch=100 \
--no-epoch-checkpoint \
--robust-embedder-resource=$ROBUST_EMB \
--edge-threshold=0.075 \
--save-dir=$CKPT_DIR \
--share-all-embeddings \
--num-workers 0
               
               
