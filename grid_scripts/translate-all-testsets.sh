#!/usr/bin/env bash

# Usage: translate-all-testsets MODELDIR SRC > lang-test.sh

set -eu

MODELDIR=$1
SRC=$2
TRG=en

#[[ ! -e $MODELDIR/status ]] && exit

for testset in /exp/esalesky/visrep/fairseq-ocr/visual/test-sets/*.$SRC-$TRG.$SRC; do
    ref=$(echo $testset | perl -pe "s/.$SRC$/.$TRG/")
    outfile=$MODELDIR/out.$(basename $testset)
    bleufile=$MODELDIR/bleu.$(basename $testset)
    if [[ ! -s $bleufile ]]; then
        # echo -ne "Translating $testset with $MODELDIR\t"
        echo qsub /exp/esalesky/visrep/fairseq-ocr/grid_scripts/translate.qsub $MODELDIR $testset $MODELDIR/out.$(basename $testset) $ref
    fi
done
