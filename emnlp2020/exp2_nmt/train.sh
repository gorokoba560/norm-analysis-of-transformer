#!/bin/sh

export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

DATA_DIR=./work/processed_data
BPE_DIR=./work/bpe_model_and_vocab

# convert BPE vocab to use for fairseq
cut -f1 $BPE_DIR/de.vocab | tail -n +4 | sed "s/$/ 100/g" > $DATA_DIR/de.vocab
cut -f1 $BPE_DIR/en.vocab | tail -n +4 | sed "s/$/ 100/g" > $DATA_DIR/en.vocab

# fairseq preprocess
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $DATA_DIR/train.bpe \
    --validpref $DATA_DIR/valid.bpe \
    --testpref $DATA_DIR/valid.bpe \
    --srcdict $DATA_DIR/de.vocab \
    --tgtdict $DATA_DIR/en.vocab \
    --destdir $DATA_DIR/fairseq_preprocessed_data \
    --workers 16 # adjust here to suit your environment

# train 5 models
seed=(2253 5498 9819 9240 2453) # these seeds were randomly decided
for i in ${seed[@]}
do
    mkdir ./work/checkpoints_seed${i}
    mkdir -p ./work/results_seed${i}/alignments
    CUDA_VISIBLE_DEVICES=0,1 fairseq-train \
        $DATA_DIR/fairseq_preprocessed_data \
        --arch transformer_iwslt_de_en \
        --activation-fn relu \
        --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
        --lr 0.001 --min-lr 9e-8 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
        --weight-decay 0.0 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --update-freq 8 \
        --skip-invalid-size-inputs-valid-test \
        --batch-size 80 --max-tokens 4000 --max-epoch 100 --seed ${i} \
        --warmup-init-lr 1e-07 --warmup-updates 4000 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --eval-bleu-print-samples \
        --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
        --log-format json --save-dir ./work/checkpoints_seed${i} --fp16 \
        --source-lang de --target-lang en \
        --save-interval-updates 10000 --keep-last-epochs 5 \
        --eval-tokenized-bleu | tee ./work/checkpoints_seed${i}/train.log
done
