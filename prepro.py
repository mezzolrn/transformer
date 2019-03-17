# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

import os
import errno
from hparams import Hparams
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)


def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")
    train_feature = "data/feature_train.txt"
    train_label = "data/label_train.txt"
    eval_feature = "data/feature_eval.txt"
    eval_label = "data/label_eval.txt"
    test_feature = "data/feature_test.txt"
    test_label = "data/label_test.txt"
    for f in (train_feature, train_label, eval_feature, eval_label, test_feature, test_label):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")

    def _prepro(x):
        pre = []
        post = []
        for line in open(x, 'r').read().strip().split("\n"):
            a = ' '.join(line[2:-2].split("', '")[:10])
            pre.append(a)
            a = ' '.join(line[2:-2].split("', '")[10:])
            post.append(a)
        return pre, post

    # train

    prepro_train_feature_pre, prepro_train_feature_post = _prepro(train_feature)
    print(np.shape(prepro_train_feature_pre), np.shape(prepro_train_feature_post))

    assert len(prepro_train_feature_pre) == len(prepro_train_feature_post), "Check if train source and target files match."

    # eval
    prepro_eval_feature_pre, prepro_eval_feature_post = _prepro(eval_feature)
    assert len(prepro_eval_feature_pre) == len(prepro_eval_feature_post), "Check if eval source and target files match."

    # test
    prepro_test_feature_pre, prepro_test_feature_post = _prepro(test_feature)
    assert len(prepro_test_feature_pre) == len(prepro_test_feature_post), "Check if test source and target files match."

    logging.info("Let's see how preprocessed data look like")
    logging.info("prepro_train_feature_pre:"+str(prepro_train_feature_pre[0]))

    logging.info("# write preprocessed files to disk")
    os.makedirs("data/prepro", exist_ok=True)

    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train_feature_pre, "data/prepro/prepro_train_feature_pre")
    _write(prepro_train_feature_post, "data/prepro/prepro_train_feature_post")
    _write(prepro_eval_feature_pre, "data/prepro/prepro_eval_feature_pre")
    _write(prepro_eval_feature_post, "data/prepro/prepro_eval_feature_post")
    _write(prepro_test_feature_pre, "data/prepro/prepro_test_feature_pre")
    _write(prepro_test_feature_post, "data/prepro/prepro_test_feature_post")


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")