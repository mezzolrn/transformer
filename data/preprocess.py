import pandas as pd
import numpy as np

import random

f = pd.read_excel('./钰诚系_微博_人工_timesort.xlsx')

f = f[['SVM12维特征','人工']]

label = []
type = []
f_label = open('label.txt', 'w', encoding='utf-8')

random.seed(1)
a = list(range(len(f)-20))
random.shuffle(a)
print(a)

for i in a:
    label.append(list(f['SVM12维特征'][i:i+20]))
    f_label.write(str(list(f['SVM12维特征'][i:i+20]))+'\n')

print(np.shape(label))
print(label[0])

f_type = open('type.txt', 'w', encoding='utf-8')

for i in a:
    type.append(list(f['人工'][i:i+20]))
    f_type.write(str(list(f['人工'][i:i+20]))+'\n')

print(np.shape(type))
print(type[0])


