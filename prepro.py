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
    pre = 'data/yucheng/'
    logging.info("# Check if raw files exist")
    train_feature = pre+"feature_train.txt"
    train_label = pre+"label_train.txt"
    eval_feature = pre+"feature_eval.txt"
    eval_label = pre+"label_eval.txt"
    test_feature = pre+"feature_test.txt"
    test_label = pre+"label_test.txt"
    for f in (train_feature, train_label, eval_feature, eval_label, test_feature, test_label):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")

    def _prepro(x):
        pre = []
        post = []
        for line in open(x, 'r', encoding='utf-8').readlines():
#             print(line)
            line = line.strip().split(" ")
#             print(line)
            a = ' '.join(line[:hp.feature_num*10])
            pre.append(a)
            a = ' '.join(line[hp.feature_num*10:])
            post.append(a)
#             print(pre, post)
        return pre, post
    
    def _prepro_3(x):
        pre = []
        post = []
        for line in open(x, 'r', encoding='utf-8').readlines():
            total = line.strip().split(" ")
            a = ''
            for i in range(20):
                if float(total[i]) == 1.:
                    total[i] = '0 1 0'
                elif float(total[i]) == 0.:
                    total[i] = '1 0 0'
                else:
                    total[i] = '0 0 1'
            a = ' '.join(total[:10])
            pre.append(a)
            a = ' '.join(total[10:])
            post.append(a)
        return pre, post

    # train
    prepro_train_feature_pre, prepro_train_feature_post = _prepro(train_feature)
    assert len(prepro_train_feature_pre) == len(prepro_train_feature_post), "Check if train source and target files match."

    # eval
    prepro_eval_feature_pre, prepro_eval_feature_post = _prepro(eval_feature)
    assert len(prepro_eval_feature_pre) == len(prepro_eval_feature_post), "Check if eval source and target files match."

    # test
    prepro_test_feature_pre, prepro_test_feature_post = _prepro(test_feature)
    assert len(prepro_test_feature_pre) == len(prepro_test_feature_post), "Check if test source and target files match."

    
    # test
    _, prepro_train_label_post = _prepro_3(train_label)
#     assert len(prepro_test_feature_pre) == len(prepro_test_feature_post), "Check if test source and target files match."

    # test
    _, prepro_eval_label_post = _prepro_3(eval_label)
#     assert len(prepro_test_feature_pre) == len(prepro_test_feature_post), "Check if test source and target files match."

    # test
    _, prepro_test_label_post = _prepro_3(test_label)
#     assert len(prepro_test_feature_pre) == len(prepro_test_feature_post), "Check if test source and target files match."

    
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
    
    _write(prepro_train_label_post, "data/prepro/prepro_train_label_post")
    _write(prepro_eval_label_post, "data/prepro/prepro_eval_label_post")
    _write(prepro_test_label_post, "data/prepro/prepro_test_label_post")


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")