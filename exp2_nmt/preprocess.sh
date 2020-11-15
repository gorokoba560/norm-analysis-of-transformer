#!/bin/sh

PROGRESS_DIR=./work/data_in_progress
DATA_DIR=./work/processed_data

bash data_preparation.sh

sed '/^[[:space:]]*$/d' < ./data/DeEn/alignmentDeEn.talp > work/gold_alignment/alignment.talp
cat ./data/DeEn/de | sed '/^[[:space:]]*$/d' | iconv -f latin1 -t utf-8 > $PROGRESS_DIR/test.uc.de
cat ./data/DeEn/en | sed '/^[[:space:]]*$/d' | iconv -f latin1 -t utf-8 > $PROGRESS_DIR/test.uc.en

python preprocess.py