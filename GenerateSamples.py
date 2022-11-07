# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:14:14 2022

@author: MOOM
"""
from DataPreprocessing import *
import numpy as np
### 查看数据
print(data.shape)

'''
数据可视化
'''
# 数据可视化
import matplotlib.pyplot as plt
plt.figure(1)
for i in range(4):
  plt.plot(range(data.shape[0])[:512], data.iloc[:,i].values[:512])
  plt.show()

# 按标签拆分数据
SIGNALS = []
for i in range(4):
  signal = data.iloc[:,i].values
  nan_array = np.isnan(signal)
  not_nan_array = ~nan_array
  new_signal = signal[not_nan_array]
  SIGNALS.append(new_signal)
print(SIGNALS[0].shape)

lengths = [SIGNALS[i].shape[0] for i in range(4)]

# 基于train.csv文件生成的样本
import pandas as pd
import random
X_test = pd.read_csv('D:/biancheng/datasets/Sub_healthData/test.csv')
columns_sample = list(X_test.columns)[1:] + ['label']

def sample_generater(SIGNALS, size, columns_sample):
  data_reset = []
  for i in range(4):
    signal_i = SIGNALS[i]
    m = random.choice(range(size))
    print(m)
    indexs_i = range(m, m + size*(int(len(signal_i)/size)-2), int(size/10))  # 此处的5可以控制样本量
    for j in indexs_i:
      sample_ = list(signal_i[j:j+size]) + [i]
      data_reset.append(sample_)
  data_reset = pd.DataFrame(data_reset, columns=columns_sample)
  print(data_reset.shape)
  return data_reset
data_reset = sample_generater(SIGNALS, size=512, columns_sample=columns_sample)


# 打乱顺序并保存
from sklearn.utils import shuffle
data_reset = shuffle(data_reset)
data_reset.to_csv('D:/biancheng/datasets/Sub_healthData/train_samples.csv', index=False)

print(data_reset.head())

temp = pd.read_csv('D:/biancheng/datasets/Sub_healthData/train_samples.csv')
print(temp.isnull().any())  # 判断有没有空值

'''
# 探索性数据分析
import pandas_profiling
pfr = pandas_profiling.ProfileReport(data_reset)
pfr.to_file('EDA.html')
'''