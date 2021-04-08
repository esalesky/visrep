#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-30
#
# End-to-end training with source loss.
#

# module load cuda10.1/toolkit/10.1.105
# module load cudnn/7.6.1_cuda10.1
# module load gcc/7.2.0

set -eu

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

# source deactivate
# source activate /expscratch/detter/tools/anaconda3

## Settings
SRC=de
TRG=en
FAIRSEQ=/home/hltcoe/mpost/code/fairseq-ocr

MODELDIR=$1
shift

export PYTHONPATH=$FAIRSEQ

echo "PATH: ${PATH}"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "FAIRSEQ: $FAIRSEQ"
echo "MODELDIR: $MODELDIR"

PYTHONPATH=$FAIRSEQ python -m fairseq_cli.interactive \
  $MODELDIR \
  --task 'visual_text' \
  --path $MODELDIR/checkpoint_best.pt \
  -s $SRC -t $TRG \
  --target-dict dict.$TRG.txt \
  --beam 5 \
  "$@"
