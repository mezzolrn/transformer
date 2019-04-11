# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses, calc_bleu
import os
from hparams import Hparams
import math
import logging
import numpy as np
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO)


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.logdir)

logging.info("# Prepare train/eval batches")
print(hp.batch_size)
train_batches, num_train_batches, num_train_samples = get_batch(hp.prepro_train_feature_pre,
                                                                hp.prepro_train_feature_post,
                                                                hp.prepro_train_label_post,
                                                                hp.batch_size,
                                                                shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.prepro_eval_feature_pre,
                                                             hp.prepro_eval_feature_post,
                                                             hp.prepro_eval_label_post,
                                                             hp.batch_size,
                                                             shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries, logits, class_3, y_3 = m.train(xs, ys)
logits, class_3, eval_summaries = m.eval(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.num_epochs)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.logdir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.logdir, "specs"))
    else:
        saver.restore(sess, ckpt)

    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)
    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary, _class_3, _y_3 = sess.run([train_op, global_step, train_summaries, class_3, y_3])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            logging.info("# test evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses_feature_str, hypotheses_feature = get_hypotheses(num_eval_batches, num_eval_samples, sess, logits)
            hypotheses_label_str, hypotheses_label = q(num_eval_batches, num_eval_samples, sess, class_3)

            logging.info("# write results")
            
            model_output = "feature_%02dL%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses_feature_str))

#             logging.info("# save models")
#             ckpt_name = os.path.join(hp.logdir, model_output)
#             saver.save(sess, ckpt_name, global_step=_gs)
#             logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))
            
            
            
            model_output = "model_epoch%02d_loss%.2f" % (epoch, _loss)
            if not os.path.exists(hp.evaldir): os.makedirs(hp.evaldir)
            translation = os.path.join(hp.evaldir, model_output)
            with open(translation, 'w') as fout:
                fout.write("\n".join(hypotheses_label_str))
                
            logging.info("# save models")
            if not os.path.exists(hp.logdir): os.makedirs(hp.logdir)
            ckpt_name = os.path.join(hp.logdir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            
            y_true = open(hp.prepro_eval_label_post, 'r', encoding='utf-8').readlines()
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
                             
                print(one, two)
                result = classification_report(one, two, digits=8)
                print('time_series: '+str(i))
                print(result)
            
            
            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
