#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2019-09-06
#
# qsub -v PATH -S /bin/bash -b y -q gpu.q@@1080 -cwd -j y -N mtpreproc -l num_proc=16,mem_free=32G,h_rt=48:00:00,gpu=1  /expscratch/detter/src/fairseq/grid_scripts/kevin/preprocess_ted.sh
#
#
# Preprocess MT src and tgt files (build dictionary, created binary format)
#

source activate /expscratch/detter/tools/py36

export LD_LIBRARY_PATH=/cm/local/apps/gcc/7.2.0/lib64:$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH
echo $PYTHONPATH

DATA_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/tok_char
FAIRSEQ_PATH=/expscratch/detter/src/fairseq-ocr
RESULT_DIR=/expscratch/detter/mt/multitarget-ted/en-zh/matt/tok_char/binary
SRC_LANG=zh #de
TGT_LANG=en
#OUT_FORMAT=binary  # binary or raw
NBR_WORKERS=20

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
--trainpref=$DATA_DIR/ted_train_en-zh.sp \
--validpref=$DATA_DIR/ted_dev_en-zh.sp \
--testpref=$DATA_DIR/ted_test1_en-zh.sp \
--destdir=$RESULT_DIR \
--workers=$NBR_WORKERS

#--output-format=$OUT_FORMAT