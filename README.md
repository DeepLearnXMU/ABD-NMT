Asynchronous Bidirectional Decoding for Neural Machine Translation
=====================================================================

This codebase contains all scripts except training corpus to reproduce our results in the paper.

### Installation

The following packages are needed:

- Python >= 2.7
- numpy
- Theano >= 0.7 (and its dependencies).

### Preparation

First, preprocess your training corpus. For English-German translation, use BPE(byte-piar-encoding) to segment text into subword units. Please follow <https://github.com/rsennrich/subword-nmt> for further details.

To obtain vocabulary for training, invoke following commands:

    python scripts/buildvocab.py --corpus /path/to/BPE/train.en --output /path/to/BPE/en.voc5.pkl \
    --limit 50000 --groundhog
    python scripts/buildvocab.py --corpus /path/to/BPE/train.de --output /path/to/BPE/de.voc5.pkl \
    --limit 50000 --groundhog

And also, it's preferred, but not required to initialize encoder-backward decoder component with pretrained parameters in the proposed model of this work.


### Training

To start training, do the following:

    python -u rnnsearch.py train --corpus /path/to/BPE/train.en /path/to/BPE/train.de \
    --vocab /path/to/BPE/en.voc5.pkl /path/to/BPE/de.voc5.pkl \
    --method concat \
    --softk 1 \
    --lambda 0.5 \
    --ext-val-script scripts/validate-deen-bpe.sh \
    --model ende \
    --embdim 620 620 \
    --hidden 1000 1000 1000 \
    --maxhid 500 \
    --deephid 620 \
    --maxpart 2 \
    --alpha 5e-4 \
    --norm 1.0 \
    --batch 80 \
    --maxepoch 5 \
    --seed 1234 \
    --freq 500 \
    --vfreq 2500 \
    --sfreq 500 \
    --sort 32 \
    --validation /path/to/BPE/newstest2013.en \
    --references /path/to/newstest2013.tc.de \
    --optimizer rmsprop \
    --shuffle 1 \
    --keep-prob 0.7 \
    --limit 50 50 \
    --delay-val 1 \
    --initialize /path/to/models/r.ende.best.pkl \
    > log.ende 2>&1 &

where /path/to/models/r.ende.best.pkl is a pretrained model running from right to left. The training procedure continues about 7 days On a single Nvidia Titan x GPU.

### Evaluation

The evaluation metric we use is case-sensitive BLEU on tokenized reference. Translate the test set and restore text to the original segmentation:

    python rnnsearch.py translate --model ende.best.pkl < /path/to/BPE/newstest2015.en \
    | scripts/restore_bpe.sh > newstest2015.de.trans

And evaluation proceeds by running:

    perl scripts/multi-bleu.perl /path/to/newstest2015.tc.de < newstest2015.de.trans