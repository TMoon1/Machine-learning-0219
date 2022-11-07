# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:29:55 2022

@author: MOOM
"""

import pandas as pd
data_features_labels = pd.read_csv('D:/biancheng/datasets/Sub_healthData/Data_features.csv')

data_features = data_features_labels.iloc[:,:-1].values
data_labels = data_features_labels.iloc[:,-1].values

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_features, data_labels, random_state=501) # 0501

import lightgbm as lgb
import numpy as np


# 数据准备
train_matrix = lgb.Dataset(X_train, label=y_train)
valid_matrix = lgb.Dataset(X_val, label=y_val)


# 构建lightgbm
params = {
    'learning_rate': 0.1,
    'boosting': 'gbdt',
    'lambda_l2': 0.1,
    'max_depth': -1,
    'num_leaves': 512,
    'bagging_fraction': 0.8,
    'feature_fraction':0.8,
    'metric': None,
    'objective': 'multiclass',
    'num_class': 4,
    'nthread': 10,
    'verbose': -1,
}

# 使用lightgbm训练
model = lgb.train(params,
          train_set=train_matrix,
          valid_sets=valid_matrix,
          num_boost_round=2000,   # 决策树提升循环次数
          verbose_eval=50,
          early_stopping_rounds=200,
          # feval=f1_score
          )

# 对验证集进行预测
'''对验证集进行预测'''
from sklearn.metrics import f1_score
val_pre_lgb = model.predict(X_val, num_iteration=model.best_iteration)
preds = np.argmax(val_pre_lgb, axis=1) 
score = f1_score(y_true=y_val, y_pred=preds, average='macro')
print('未调参前lightgbm单模型在验证集上的f1：{}'.format(score))

# 对模型进行保存
model.save_model('D:/biancheng/datasets/Sub_healthData/Model/lightGBM_model_0501.txt')
# 对测试集进行预测
X_test = pd.read_csv('D:/biancheng/datasets/Sub_healthData/test.csv')

# 特征工程
import tsfel
import pandas as pd

X_test_features = pd.DataFrame()
for i in range(X_test.shape[0]):
  signal_i = X_test.iloc[i,1:]
  cfg_file = tsfel.get_features_by_domain()
  features_i = tsfel.time_series_features_extractor(cfg_file, signal_i, fs=1, window_size=512)  #非常耗时
  X_test_features = pd.concat([X_test_features, features_i])
  # if i==10:
  #   break

print(X_test_features)
X_test_features.to_csv('D:/biancheng/datasets/Sub_healthData/X_test_features.csv')