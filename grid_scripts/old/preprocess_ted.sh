#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-10-17
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N mtpreproc -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1  /expscratch/detter/src/fairseq-ocr/grid_scripts/transformer/preprocess_ted.sh
#
#
# Preprocess MT src and tgt files (build dictionary, created binary format)
#

source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH

SRC_LANG=zh #de zh
TGT_LANG=en
DATA_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k
RESULT_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/spm.20000.10k/raw
OUT_FORMAT=raw  # binary or raw
NBR_WORKERS=20

FAIRSEQ_PATH=/expscratch/detter/src/fairseq/fairseq

echo $DATA_DIR
echo $FAIRSEQ_PATH
echo $RESULT_DIR
echo $SRC_LANG
echo $TGT_LANG
echo $OUT_FORMAT
echo $NBR_WORKERS

python $FAIRSEQ_PATH/preprocess.py \
--source-lang=$SRC_LANG \
--target-lang=$TGT_LANG \
--user-dir=$FAIRSEQ_PATH \
--trainpref=$DATA_DIR/ted_train_$TGT_LANG-$SRC_LANG.sp \
--validpref=$DATA_DIR/ted_dev_$TGT_LANG-$SRC_LANG.sp \
--testpref=$DATA_DIR/ted_test1_$TGT_LANG-$SRC_LANG.sp \
--destdir=$RESULT_DIR \
--workers=$NBR_WORKERS \
--dataset-impl=$OUT_FORMAT

#--trainpref=$DATA_DIR/ted_train_$TGT_LANG-$SRC_LANG.raw.norm.10000 \
#--validpref=$DATA_DIR/ted_dev_$TGT_LANG-$SRC_LANG.raw.norm.10000 \
#--testpref=$DATA_DIR/ted_test1_$TGT_LANG-$SRC_LANG.raw.norm.10000 \
