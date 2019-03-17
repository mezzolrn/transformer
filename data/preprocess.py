import pandas as pd
import numpy as np

import random

f = pd.read_excel('./钰诚系_微博_人工_timesort.xlsx')

f = f[['SVM12维特征','人工']]

feature = []
label = []

random.seed(1)
a = list(range(len(f)-20))
random.shuffle(a)
print(a)

train_len = int(len(a)*0.8)
test_len = int(len(a)*0.1)+train_len

f_feature = open('feature_train.txt', 'w', encoding='utf-8')
for i in a[:train_len]:
    feature.append(list(f['SVM12维特征'][i:i+20]))
    f_feature.write(str(list(f['SVM12维特征'][i:i+20]))+'\n')

f_feature = open('feature_eval.txt', 'w', encoding='utf-8')
for i in a[train_len:test_len]:
    feature.append(list(f['SVM12维特征'][i:i+20]))
    f_feature.write(str(list(f['SVM12维特征'][i:i+20]))+'\n')


f_feature = open('feature_test.txt', 'w', encoding='utf-8')
for i in a[test_len:]:
    feature.append(list(f['SVM12维特征'][i:i+20]))
    f_feature.write(str(list(f['SVM12维特征'][i:i+20]))+'\n')

print(np.shape(feature))
print(feature[0])







f_label = open('label_train.txt', 'w', encoding='utf-8')
for i in a[:train_len]:
    label.append(list(f['人工'][i:i+20]))
    f_label.write(str(list(f['人工'][i:i+20]))+'\n')

f_label = open('label_eval.txt', 'w', encoding='utf-8')
for i in a[train_len:test_len]:
    label.append(list(f['人工'][i:i+20]))
    f_label.write(str(list(f['人工'][i:i+20]))+'\n')

f_label = open('label_test.txt', 'w', encoding='utf-8')
for i in a[test_len:]:
    label.append(list(f['人工'][i:i+20]))
    f_label.write(str(list(f['人工'][i:i+20]))+'\n')

print(np.shape(label))
print(label[0])


