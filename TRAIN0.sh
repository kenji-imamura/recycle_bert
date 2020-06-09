#! /bin/sh
#
# Tokenize raw sentences
#
trap 'exit 2' 2
DIR=$(cd $(dirname $0); pwd)

CODE=$DIR/user_code
export PYTHONPATH="$CODE:$PYTHONPATH"

BERT_MODEL=$DIR/uncased_L-12_H-768_A-12
CORPUS=$DIR/corpus
SRC=en
TRG=de

#
# Download the corpus
#
mkdir -p $CORPUS
for prefix in train newstest2012 newstest2013 newstest2014 newstest2015; do
    for lang in $SRC $TRG; do
	file=$prefix.$lang
	if [ !  -f $CORPUS/$file ]; then
	    wget -P $CORPUS https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/$file
	fi
    done
done

#
# Download the BERT model
#
if [ ! -d $BERT_MODEL ]; then
    wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
    unzip uncased_L-12_H-768_A-12.zip
    (cd $BERT_MODEL ; \
     ln -s bert_config.json config.json ; \
     transformers-cli convert --model_type bert \
	--tf_checkpoint bert_model.ckpt \
	--config bert_config.json \
	--pytorch_dump_output pytorch_model.bin)
fi

#
# Train sub-word models
#

### sentencepiece
sp_encode () {
    lang=$1
    size=$2
    spm_train --model_prefix=$CORPUS/train.spm.$lang \
	      --input=$CORPUS/train.$lang \
	      --vocab_size=$size \
	      --character_coverage=1.0 \
	      > $CORPUS/train.spm.$lang.log 2>&1
}

sp_encode   $TRG 16384 &
wait

#
# Apply the sub-word models
#

### sentencepiece
sp_decode () {
    lang=$1
    testset=$2
    cat $CORPUS/${testset}.$lang \
	| spm_encode --model=$CORPUS/train.spm.$lang.model \
	> $CORPUS/${testset}.bpe.$lang
}

### BERT tokenizer
bert_decode () {
    lang=$1
    testset=$2
    model=$3
    cat $CORPUS/${testset}.$lang \
	| python3.7 $CODE/bert_tokenize.py \
		  --model=$model \
		  > $CORPUS/${testset}.bpe.$lang
}

for testset in train newstest2012 newstest2013 newstest2014 newstest2015; do
    sp_decode   $TRG $testset &
    bert_decode $SRC $testset $BERT_MODEL &
    wait
done
