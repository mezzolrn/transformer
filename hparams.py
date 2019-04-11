import argparse

class Hparams:
    parser = argparse.ArgumentParser()

    # train
    parser.add_argument('--prepro_train_feature_pre', default='data/prepro/prepro_train_feature_pre',
                        help="training feature")
    parser.add_argument('--prepro_train_feature_post', default='data/prepro/prepro_train_feature_post',
                        help="training feature")
    parser.add_argument('--prepro_eval_feature_pre', default='data/prepro/prepro_eval_feature_pre',
                        help="eval feature")
    parser.add_argument('--prepro_eval_feature_post', default='data/prepro/prepro_eval_feature_post',
                        help="eval feature")
    parser.add_argument('--prepro_test_feature_pre', default='data/prepro/prepro_test_feature_pre',
                        help="eval feature")
    parser.add_argument('--prepro_test_feature_post', default='data/prepro/prepro_test_feature_post',
                        help="eval feature")
    
    parser.add_argument('--prepro_train_label_post', default='data/prepro/prepro_train_label_post',
                        help="eval feature")
    parser.add_argument('--prepro_eval_label_post', default='data/prepro/prepro_eval_label_post',
                        help="eval feature")
    parser.add_argument('--prepro_test_label_post', default='data/prepro/prepro_test_label_post',
                        help="eval feature")
    ## files
    parser.add_argument('--train_feature', default='data/feature_train.txt',
                             help="training feature")
    parser.add_argument('--eval_feature', default='data/feature_eval.txt',
                             help="eval feature")
    parser.add_argument('--train_label', default='data/label_train.txt',
                             help="training label")
    parser.add_argument('--eval_label', default='data/label_eval.txt',
                             help="eval label")

    # training scheme
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)

    parser.add_argument('--lr', default=0.00003, type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=4000, type=int)
    parser.add_argument('--logdir', default="log/1", help="log directory")
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--evaldir', default="eval/1", help="evaluation dir")
    parser.add_argument('--feature_num', default=12, help="evaluation dir")

    # model
    parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
    parser.add_argument('--d_ff', default=2048, type=int,
                        help="hidden dimension of feedforward layer")
    parser.add_argument('--num_blocks', default=6, type=int,
                        help="number of encoder/decoder blocks")
    parser.add_argument('--num_heads', default=2, type=int,
                        help="number of attention heads")
    parser.add_argument('--maxlen1', default=10, type=int,
                        help="maximum length of a source sequence")
    parser.add_argument('--maxlen2', default=10, type=int,
                        help="maximum length of a target sequence")
    parser.add_argument('--dropout_rate', default=0.3, type=float)
    parser.add_argument('--smoothing', default=0.1, type=float,
                        help="label smoothing rate")

    # test
    parser.add_argument('--test_feature', default='data/feature_test.txt',
                        help="test_feature")
    parser.add_argument('--test_label', default='data/label_eval.txt',
                        help="test_label")
    parser.add_argument('--ckpt', help="checkpoint file path")
    parser.add_argument('--test_batch_size', default=8, type=int)
    parser.add_argument('--testdir', default="test/1", help="test result dir")