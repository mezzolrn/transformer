# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu, postprocess, load_hparams
import logging
import numpy as np
from sklearn.metrics import classification_report


logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.ckpt)

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(hp.prepro_test_feature_pre,
                                                              hp.prepro_test_feature_post,
                                                              hp.prepro_test_label_post,
                                                              hp.batch_size,
                                                              shuffle=False)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
logits, pre_3, _ = m.eval(xs, ys)

logging.info("# Session")
with tf.Session() as sess:
    ckpt_ = tf.train.latest_checkpoint(hp.ckpt)
    ckpt = hp.ckpt if ckpt_ is None else ckpt_ # None: ckpt is a file. otherwise dir.
    saver = tf.train.Saver()
    print(ckpt,'ckpt')
    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# get hypotheses")
    hypotheses_feature_str, hypotheses_feature = get_hypotheses(num_test_batches, num_test_samples, sess, logits)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output)
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses_feature_str))
        
    logging.info("# get hypotheses")
    hypotheses_label_str, hypotheses_label = get_hypotheses(num_test_batches, num_test_samples, sess, pre_3)

    logging.info("# write results")
    model_output = ckpt.split("/")[-1]
    if not os.path.exists(hp.testdir): os.makedirs(hp.testdir)
    translation = os.path.join(hp.testdir, model_output+'_label')
    with open(translation, 'w') as fout:
        fout.write("\n".join(hypotheses_label_str))


    
    y_true = open(hp.prepro_test_label_post, 'r', encoding='utf-8').readlines()
    for i in range(len(y_true)):
        tmp = y_true[i].strip().split(' ')
        for j in range(len(tmp)):
            tmp[j] = int(tmp[j])
        y_true[i] = tmp
    y_true = np.argmax(np.reshape(y_true, [-1,10,3]),2)
    y_predict = np.argmax(np.reshape(hypotheses_label, [-1,10,3]), 2)

    for i in range(10):
        one = y_true[:,i:i+1]
        one = np.reshape(one, [-1,])

        two = y_predict[:,i:i+1]
        two = np.reshape(two, [-1,])

        print(one)
        print(two)
        result = classification_report(one, two, digits=8)
        print('time_series: '+str(i))
        print(result)
    
