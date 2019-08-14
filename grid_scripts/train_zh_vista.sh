#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-08-14
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N zhvista3 -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1 /expscratch/detter/src/fairseq-ocr/grid_scripts/train_zh_vista.sh


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

EXPDIR=/expscratch/detter/ocr/data/derived/ocrseq/vista/20190814
LMDB_LANG=/exp/ocr/data/yomdle
LANG_CODE=zh
LANG_NAME=chinese

echo $EXPDIR
echo $LANG_CODE
echo $LANG_NAME

mkdir -p $EXPDIR
mkdir -p $EXPDIR/samples
mkdir -p $EXPDIR/scores
mkdir -p $EXPDIR/results
mkdir -p $EXPDIR/ckpt

mkdir -p $TMPDIR/${LANG_CODE}
pushd $TMPDIR/${LANG_CODE}
tar xf ${LMDB_LANG}/yomdle_${LANG_NAME}_traindevtest.tar
popd

python /expscratch/detter/src/fairseq-ocr/train.py \
--user-dir=/expscratch/detter/src/fairseq-ocr \
--save-dir=$EXPDIR/ckpt \
--lmdb=${TMPDIR}/${LANG_CODE}/yomdle_${LANG_NAME}_traindevtest \
--dictionary=${TMPDIR}/${LANG_CODE}/yomdle_${LANG_NAME}_traindevtest/desc.json \
--result_path=$EXPDIR \
--valid-subset=valid \
--train-subset=train \
--lr=1e-3 \
--min-lr=1e-07 \
--optimizer=adam \
--save-interval=1 \
--validate-interval=1 \
--batch-size=32 \
--height=32 \
--max-width=900 \
--num-workers=16 \
--max-epoch=500 \
--task=ocr \
--arch=ocr_vista \
--encoder-arch=vista \
--encoder-dim=512 \
--lstm-layers=2 \
--lstm-hidden-size=640 \
--lstm-bidirectional \
--lstm-dropout=0.50 \
--criterion=ctc_loss \
--augment

##--lr-scheduler=reduce_lr_on_plateau \
#--lr-shrink=0.95 \

echo "COMPLETE"
