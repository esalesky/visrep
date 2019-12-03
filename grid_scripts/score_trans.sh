#!/bin/bash
#. /etc/profile.d/modules.sh
#
#
# 2019-11-19
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@2080 -cwd -j y -N koscr \
# -l num_proc=4,mem_free=16G,h_rt=48:00:00,gpu=1 \
# /expscratch/detter/src/fairseq/fairseq-ocr/grid_scripts/score_visual.sh SRC_LANG [ko zh ja de fr]
#
# Score transformer 
#

module load cuda10.0/toolkit/10.0.130
module load cudnn/7.5.0_cuda10.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH


SRC_LANG=ko #${1} # ko zh ja de fr
TGT_LANG=en
FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq-ocr
SIZE=5k #${2}

DATA_DIR=/exp/esalesky/mtocr19/$SRC_LANG-$TGT_LANG/data/${SIZE} #/dict.$SRC_LANG.txt 

CKPT_DIR=/expscratch/detter/mt/multitarget-ted/visemb/$SRC_LANG-$TGT_LANG/${SIZE}/vis_trans/trans
FONT_FILE=/expscratch/detter/fonts/mt/test_${SRC_LANG}_font.txt

echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "source lang - ${SRC_LANG}"
echo "ckpt dir - ${CKPT_DIR}"
echo "data dir - ${DATA_DIR}"
echo "size - ${SIZE}"
echo "font file - ${FONT_FILE}"
echo "fairseq path - ${FAIRSEQ_PATH}"

nvidia-smi


python $FAIRSEQ_PATH/generate.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--task=translation \
--optimizer=adam \
--adam-betas='(0.9, 0.98)' \
--lr-scheduler=inverse_sqrt \
--warmup-updates=4000 \
--weight-decay=0.0001 \
--max-tokens=4000 \
--criterion=visual_label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--num-workers=0 \
--raw-text \
--beam=5 \
--path=$CKPT_DIR/checkpoint_best.pt

#--arch=visual_transformer_iwslt_de_en \
#--remove-bpe




               