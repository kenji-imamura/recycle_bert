# Recycling a Pre-trained BERT Encoder for Neural Machine Translation (Imamura and Sumita, 2019)

This example replaces the Transformer-based fairseq encoder with a
pre-trained BERT encoder.  Based on the fairseq v0.8.0, the paper
[https://www.aclweb.org/anthology/D19-5603/] is implemented.

## Example usage
This example assumes the following data.

- The corpus used here is 
[WMT-2014 En-De Corpus](https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/)
preprocessed by the Stanford NLP Group.
- For the pre-trained BERT model,
[Google's uncased BERT-base model](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)
is used.

### Requirements
This example is based on the [fairseq](https://github.com/pytorch/fairseq)
and uses [transformers](https://github.com/huggingface/transformers)
for applying pre-trained BERT models.
```
pip3 install fairseq
pip3 install transformers
```

### Directories / paths
```
#! /bin/bash
export BERT_MODEL=./uncased_L-12_H-768_A-12
export CODE=./user_code
export CORPUS=./corpus
export DATA=./data
export MODEL_STAGE1=./model.stage1
export MODEL_STAGE2=./model.stage2
export PYTHONPATH="$CODE:$PYTHONPATH"
```

### Conversion
To use pre-trained BERT models for the TensorFlow library in the
fairseq translator, they have to be converted into the models for the
PyTorch library.

- If you have already installed
[transformers](https://github.com/huggingface/transformers), there is
the `transformers` command in your path.
```
cd $BERT_MODEL
ln -s bert_config.json config.json
transformers bert \
	     bert_model.ckpt \
	     config.json \
	     pytorch_model.bin
```

### Tokenization
The source side of corpora
is tokenized and converted into sub-words using the BERT tokenizer.
```
cat $CORPUS/train.en \
    | python3 $CODE/bert_tokenize.py \
          --model=$BERT_MODEL > $CORPUS/train.bpe.en
```

- `TRAIN0.sh` is a sample script.
- You can use any tokenizers for the target side.
The [sentencepiece](https://github.com/google/sentencepiece) is used
in `TRAIN0.sh`.

### Binarization
First, the vocabulary file in the BERT model is converted into that
for the fairseq.
- The IDs 0 to 3 are removed from the vocabulary
because they are reserved in the fairseq.
```
mkdir -p $DATA
cat $BERT_MODEL/vocab.txt \
    | tail -n +5 | sed -e 's/$/ 0/' \
    > $DATA/dict.en.txt
```

Then, the tokenized corpora are converted into binary data for the fairseq.
- You have to use `$CODE/preprocess.py` instead of
`fairseq-preprocess`.  This program is modified for applying BERT
models.  `TRAIN1.sh` is a sample script.
```
python3 $CODE/preprocess.py \
    --source-lang en --target-lang de \
    --srcdict $DATA/dict.en.txt \
    --trainpref $CORPUS/train.bpe \
    --validpref $CORPUS/newstest2013.bpe \
    --testpref $CORPUS/newstest2014.bpe,$CORPUS/newstest2015.bpe \
    --destdir $DATA
```

### Training stage 1 (decoder training)

In the first stage of training,
only the decoder is trained by freezing the BERT model.

- The required arguments are `--user-dir $CODE`,
`--task translation_with_bert`,
`--arch transformer_with_pretrained_bert`(BERT base),
and`--bert-model $BERT_MODEL`.
- You can specify
`--arch transformer_with_pretrained_bert_large`
instead of `--arch transformer_with_pretrained_bert`
if you use a BERT large model.
- `TRAIN2.bash` is a sample script, in which the learning rate is set
to 0.0004, and the mini-batch size is set to around 500 sentences
(i.e., about 9,000 updates/epoch).

```
mkdir -p $MODEL_STAGE1
fairseq-train $DATA -s en -t de \
    --user-dir $CODE --task translation_with_bert \
    --bert-model $BERT_MODEL \
    --arch transformer_with_pretrained_bert \
    --no-progress-bar --log-format simple \
    --log-interval 1800 \
    --max-tokens 5000 --update-freq 4 \
    --max-epoch 20 \
    --optimizer adam --lr 0.0004 --adam-betas '(0.9, 0.99)' \
    --label-smoothing 0.1 --clip-norm 5 \
    --dropout 0.15 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 45000 --warmup-init-lr '1e-07' \
    --save-dir $MODEL_STAGE1
```

### Training stage 2 (fine-tuning)
In the stage 2,
the entire model including the BERT encoder is tuned.

- The required arguments are 
`--fine-tuning` and
`--restore-file $MODEL_STAGE1/checkpoint_best.pt`,
in addition to the arguments of the stage 1.
You can additionally specify 
`--reset-optimizer`, `--reset-lr-scheduler`, and `--reset-meters`.
- `TRAIN3.bash` is a sample script, in which the learning rate is set
to 0.00008.
- The stage 2 consumes large GPU memories.
You may adjust the mini-batch size.

```
mkdir -p $MODEL_STAGE2
fairseq-train $DATA -s en -t de \
    --user-dir $CODE --task translation_with_bert \
    --bert-model $BERT_MODEL \
    --arch transformer_with_pretrained_bert \
    --fine-tuning \
    --restore-file $MODEL_STAGE1/checkpoint_best.pt \
    --reset-lr-scheduler --reset-meters --reset-optimizer \
    --no-progress-bar --log-format simple \
    --log-interval 1800 \
    --max-tokens 5000 --update-freq 4 \
    --max-epoch 60 \
    --optimizer adam --lr 0.00008 --adam-betas '(0.9, 0.99)' \
    --label-smoothing 0.1 --clip-norm 5 \
    --dropout 0.15 \
    --min-lr '1e-09' --lr-scheduler inverse_sqrt \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy \
    --warmup-updates 9000 --warmup-init-lr '1e-07' \
    --save-dir $MODEL_STAGE2
```

### Evaluation
When you run `fairseq-generate` or `fairseq-interactive`,
you must give `--user-dir $CODE`, `--task translation_with_bert`, and
`--bert-model $BERT_MODEL`.

```
fairseq-generate $DATA -s en -t de \
    --user-dir $CODE --task translation_with_bert \
    --bert-model $BERT_MODEL \
    --no-progress-bar \
    --gen-subset valid \
    --path $MODEL_STAGE2/checkpoint_best.pt \
    --lenpen 1.0 \
    --beam 10 --batch-size 32
```

- `TEST.bash` is a sample script.
- When the fairseq loads a checkpoint, it first creates an
uninitialized model and then copies parameters from the checkpoint.
This example reads settings from the BERT model for creating the
uninitialized model.  This is the reason that `--bert-model` is
necessary in the fine-tuning and evaluation stages.

## Citation
```bibtex
@inproceedings{imamura-sumita-2019-recycling,
  title     = "Recycling a Pre-trained {BERT} Encoder for Neural Machine Translation",
  author    = "Imamura, Kenji and Sumita, Eiichiro",
  booktitle = "Proceedings of the 3rd Workshop on Neural Generation and Translation",
  publisher = "Association for Computational Linguistics",
  pages     = "23--31",
  month     = November,
  year      = 2019,
  address   = "Hong Kong",
  url       = "https://www.aclweb.org/anthology/D19-5603/",
}
```

## Acknowledgement

This work was supported by the "Research and Development of Enhanced
Multilingual and Multipurpose Speech Translation Systems" a program of
the Ministry of Internal Affairs and Communications, Japan.
