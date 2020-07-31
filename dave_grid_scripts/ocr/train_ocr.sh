#!/bin/bash
#. /etc/profile.d/modules.sh
#
# 2020-07-16
# 
# Train OCR
#
module load cuda10.1/toolkit/10.1.105
module load cudnn/7.6.1_cuda10.1
module load gcc/7.2.0

if [ ! -z $SGE_HGR_gpu ]; then
    export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
    sleep 3
fi

source deactivate
source activate /expscratch/detter/tools/anaconda3

EXP_DIR=/expscratch/detter/ocr/ocrseq/20200716
SRC_PATH=/home/hltcoe/detter/src/current/ocr/fairseq-ocr
DATA_DIR=/expscratch/detter/ocr/data/yomdle

mkdir -p $TMPDIR/ocr/zh
pushd $TMPDIR/ocr/zh
tar xf ${DATA_DIR}/zh/lmdb.tar
popd

#mkdir -p $TMPDIR/ocr/en
#pushd $TMPDIR/ocr/en
#tar xf ${DATA_DIR}/en/lmdb.tar
#popd

mkdir -p "${EXP_DIR}/ckpt"
mkdir -p "${EXP_DIR}/samples"
mkdir -p "${EXP_DIR}/tensorboard"
cd ${EXP_DIR}

GPU_NAME=`hostname -i`
LISTEN_PORT=5680

nvidia-smi
echo "PATH - ${PATH}"
echo "LD_LIBRARY_PATH - ${LD_LIBRARY_PATH}"
echo "PYTHONPATH - ${PYTHONPATH}"
echo "CUDA_VISIBLE_DEVICES - ${CUDA_VISIBLE_DEVICES}"
echo "EXP_DIR - ${EXP_DIR}"
echo "DATA_DIR - ${DATA_DIR}"
echo "TMPDIR - ${TMPDIR}"
echo "SRC_PATH - ${SRC_PATH}"
echo "GPU_NAME - ${GPU_NAME}"
echo "LISTEN_PORT - ${LISTEN_PORT}"

python -m debugpy --listen ${GPU_NAME}:${LISTEN_PORT} --wait-for-client \
${SRC_PATH}/train_ocr.py \
$TMPDIR/ocr/zh/lmdb \
--user-dir ${SRC_PATH} \
--save-dir ${EXP_DIR} \
--backbone 'vista' \
--criterion 'ctc_loss' \
--arch 'ocr_crnn_lstm' \
--task 'ocr' \
--tensorboard-path $EXP_DIR/tensorboard \
--ocr-height 32 \
--max-allowed-width 1600 \
--valid-subset 'validation' \
--batch-size 32 \
--image-samples-path ${EXP_DIR}/samples \
--image-verbose

