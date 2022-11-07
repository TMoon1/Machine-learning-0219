# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:31:19 2022

@author: MOOM
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
import joblib
# 加载模型
#from sklearn.externals import joblib
model = lgb.Booster(model_file='D:/biancheng/datasets/Sub_healthData/Model/lightGBM_model_0501.txt')

X_test_features = pd.read_csv('D:/biancheng/datasets/Sub_healthData/X_test_features.csv')

test_pre_lgb = model.predict(X_test_features, num_iteration=model.best_iteration)
preds = np.argmax(test_pre_lgb, axis=1) 

# 生成submint.csv文件
submit = pd.read_csv('D:/biancheng/datasets/Sub_healthData/submit_sample.csv')
submit['label'] = preds
submit.to_csv('D:/biancheng/datasets/Sub_healthData/submit.csv', index=False)