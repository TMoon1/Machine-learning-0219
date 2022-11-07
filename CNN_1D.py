# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:26:21 2022

@author: MOOM
"""

import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt 
import glob
import numpy as np
import pandas as pd
import math
import os
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import optimizers
#[from tensorflow.keras.objectives import *
from sklearn.preprocessing import OneHotEncoder
import os
import tensorflow.compat.v1 as tf
# from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1.keras.backend import set_session

# 加载数据
path_file = 'D:/biancheng/datasets/Sub_healthData/train_samples.csv'
data_ = pd.read_csv(path_file)

# 划分train和val
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data_.iloc[:,1:-1].values, data_.iloc[:,-1].values, random_state=501) #501

# 变换为CNN_1D的适配形式
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_val = y_val.reshape(y_val.shape[0], 1)

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

# print(data_.columns[1:-1])
print(X_train.shape)

# 构建CNN_1D模型
'''
单个样本可看到明显周期性规律，设置卷积核尺寸与“波长”相近
模式类别共计4种，网络模型预先设计三套卷积池化层
'''


"""GPU设置为按需增长"""


os.environ["CUDA_VISIBLE_DEVICES"] = "0" #有多个GPU时可以指定只使用第几号GPU
config = tf.ConfigProto()
config.allow_soft_placement=True #允许动态放置张量和操作符
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #最多使用40%GPU内存
config.gpu_options.allow_growth=True   #初始化时不全部占满GPU显存, 按需分配 
sess = tf.Session(config = config)

set_session(sess)

from keras import backend as K


TIME_PERIODS = 511
num_sensors = 1
def build_model(input_shape=(TIME_PERIODS,num_sensors),num_classes=4):
    model = Sequential()
    #model.add(Reshape((TIME_PERIODS, num_sensors), input_shape=input_shape))
    model.add(Conv1D(64, 15, strides=1,input_shape=input_shape, kernel_regularizer=keras.regularizers.l2(0.01))) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv1D(64, 15, strides=1,padding="same"))   
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(2))                            
    model.add(Conv1D(128, 15,strides=1,padding="same"))    
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv1D(128, 15,strides=1,padding="same"))    
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(2))                           
    model.add(Conv1D(256, 15,strides=1,padding="same")) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Conv1D(256, 15,strides=1,padding="same")) 
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(MaxPooling1D(2))  
    model.add(GlobalAveragePooling1D())                       
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))               
    return(model)

K.clear_session()
model_CNN = build_model(input_shape=(TIME_PERIODS,num_sensors),num_classes=4)
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30)
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=50, decay_rate=0.9)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
model_CNN.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model_CNN.fit(X_train, y_train, epochs=300, batch_size=100, validation_split =0.4, callbacks= [callback] )
model_CNN.save('D:/biancheng/datasets/Sub_healthData/Model/CNN_ECG_0501.h5')