#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-09-03
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N fconv1 -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 /expscratch/detter/src/fairseq-ocr/grid_scripts/train_visualmt_fconv.sh
#
#
# Train Visual Fully Convolutional Model
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


DATA_DIR=/expscratch/detter/mt/robust/iwslt2016/en-fr/baseline_new/preprocessed_fairseq-robust_n0.0
FAIRSEQ_PATH=/expscratch/detter/src/fairseq-ocr
FONT_LIST=/expscratch/detter/fonts/mt.txt
SRC_LANG=en
TGT_LANG=fr

CKPT_DIR=/expscratch/detter/mt/exp/visual_fconv/ckpt
SAMPLES_DIR=/expscratch/detter/mt/exp/visual_fconv/samples

echo $DATA_DIR
echo $SRC_LANG
echo $TGT_LANG
echo $CKPT_DIR
echo $FAIRSEQ_PATH
echo $SAMPLES_DIR
echo $FONT_LIST

mkdir -p $CKPT_DIR
mkdir -p $SAMPLES_DIR


python $FAIRSEQ_PATH/train.py \
$DATA_DIR \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--task=translation \
--arch=visual_fconv_iwslt_de_en \
--lr=0.25 \
--min-lr='1e-09' \
--lr-scheduler=inverse_sqrt \
--clip-norm=0.1 \
--dropout=0.1 \
--max-sentences=15 \
--seed=200 \
--max-epoch=100 \
--no-epoch-checkpoint \
--criterion=label_smoothed_cross_entropy \
--label-smoothing=0.1 \
--save-dir=$CKPT_DIR \
--image-type=word \
--image-font-path=$FONT_LIST \
--image-font-size=24 \
--image-embed-dim=256 \
--image-channels=3 \
--image-height=30 \
--image-width=120 \
--image-stride=1 \
--image-pad=1 \
--image-kernel=3 \
--image-maxpool-height=0.5 \
--image-maxpool-width=0.7 \
--image-verbose \
--image-samples-path=$SAMPLES_DIR \
--image-use-cache \
--image-font-color='black' \
--image-bkg-color='white' \
--num-workers=4

#--image-rand-augment \
#--image-rand-font \
#--image-rand-style \
#--image-edgedetect \
#--image-edgedetect-alpha=1.0 \


      
               