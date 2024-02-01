#!/bin/bash

set -eu

FAIRSEQ=/exp/esalesky/clean/visrep

MODELDIR=$1
DATADIR=$2
OUTFILE=$3
SRC=$4
CKPT=$5

TGT=en
touch $MODELDIR/dict.${SRC}.txt  # path is required but file unused; can be empty
cp $MODELDIR/tgt.vocab $MODELDIR/dict.${TGT}.txt

echo "HOSTNAME: $(hostname)"
export PYTHONPATH=$FAIRSEQ

lang_pairs="ar-en,az-en,be-en,bg-en,bn-en,bs-en,cs-en,da-en,de-en,el-en,eo-en,es-en,et-en,eu-en,fa-en,fi-en,frca-en,fr-en,gl-en,he-en,hi-en,hr-en,hu-en,hy-en,id-en,it-en,ja-en,ka-en,kk-en,ko-en,ku-en,lt-en,mk-en,mn-en,mr-en,ms-en,my-en,nb-en,nl-en,pl-en,ptbr-en,pt-en,ro-en,ru-en,sk-en,sl-en,sq-en,sr-en,sv-en,ta-en,th-en,tr-en,uk-en,ur-en,vi-en,zhcn-en,zh-en,zhtw-en"
lang_list=$MODELDIR/lang_list


cat $DATADIR/test/test.${SRC}-${TGT}.${SRC} | fairseq-interactive $MODELDIR \
  --path $CKPT \
  --task pixel_translation_multi_simple_epoch \
  --image-font-path fairseq/data/visual/fonts/NotoSans-Regular.ttf \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --beam 5 \
  --source-lang $SRC \
  --target-lang $TGT \
> $OUTFILE 2>&1

cleanfile=$(echo $OUTFILE | perl -pe "s|/out\.|/clean.|")
bleufile=$(echo $OUTFILE | perl -pe "s|/out\.|/bleu.|")
reffile=$DATADIR/test/test.${SRC}-${TGT}.${TGT}


OUTDIR=$(dirname $OUTFILE)
grep ^H- $OUTFILE | sort -V | cut -f3 | /home/hltcoe/esalesky/bin/deseg > ${OUTDIR}/clean.${SRC}.txt
cat ${OUTDIR}/clean.${SRC}.txt | python -m sacrebleu -b $reffile > $bleufile
