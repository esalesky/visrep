#!/bin/bash

#$ -cwd -S /bin/bash -V
#$ -j y -o logs/
#$ -l gpu=1 -q gpu.q -l h_rt=48:00:00

source /opt/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate fairseq

trainscript=$1
shift

bash $trainscript "$@"
