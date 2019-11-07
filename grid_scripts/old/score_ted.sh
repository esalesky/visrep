#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-29
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N score \
# -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 \
# /expscratch/detter/src/fairseq/fairseq-ocr/grid_scripts/score_ted.sh
#
#
#  update 2019-10-29
# 
#  BLEU4 = 17.96, spm.20000.10k/raw/exp/trans 
#  BLEU4 = 13.71, spm.20000.10k/raw/exp/trans_pre
#  BLEU4 = 13.71, spm.20000.10k/raw/exp/trans_pre_freeze
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

SRC_LANG=zh
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr
DATA_DIR=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/matt/spm.20000.10k/raw
CKPT_DIR=/expscratch/detter/mt/multitarget-ted/$TGT_LANG-$SRC_LANG/matt/spm.20000.10k/raw/exp/trans_pre_freeze

echo $DATA_DIR
echo $FAIRSEQ_PATH
echo $SRC_LANG
echo $TGT_LANG
echo $CKPT_DIR

python $FAIRSEQ_PATH/generate.py \
$DATA_DIR \
--path=$CKPT_DIR/checkpoint_best.pt \
--user-dir=$FAIRSEQ_PATH \
--gen-subset=test \
--batch-size=32 \
--raw-text \
--beam=5 


#--remove-bpe




               