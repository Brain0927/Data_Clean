#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "Brain"


import tensorflow as tf
import numpy as np

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import random
import math
import time
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error
import datetime
import operator
import pandas as pd
import myfun
from sklearn.preprocessing import MinMaxScaler
plt.style.use('seaborn-poster')

df = pd.read_excel("NASA(清洗後).xlsx",header=0)
print(df.head(5))
#   Pandas  x 轉 numpy
x=df.to_numpy()
print(x)

#  標準化 x
scaler = MinMaxScaler(feature_range=(0,1))      # 初始化 # 設定縮放的區間上下限
scaler.fit(x)                                   # 找標準化範圍
x= scaler.transform(x)                          # 把資料轉換
print("標準化:",x[:2])

# numpy 轉 Pandas
df = pd.DataFrame(x, columns=df.columns)
from pandas import ExcelWriter
writer = ExcelWriter('NASA標準化.xlsx', engine='xlsxwriter')      # 另存為資料清洗後
df.to_excel(writer, sheet_name='NASA標準化',index=False,header=1)   # 分頁欄位的名稱 header=1 要印表頭
writer.save()
print("====讀取資料==標準化============")
colx=['est_diameter_min','est_diameter_max','relative_velocity','miss_distance','orbiting_body','sentry_object','absolute_magnitude']
train_x, test_x, train_y, test_y=myfun.ML_read_dataframe(df, colx, ['hazardous'],0.5)
print("外型大小",train_x.shape,test_x.shape,train_y.shape,test_y.shape)
print("前面幾筆:",train_x)

category=2
dim=7


train_y2=tf.keras.utils.to_categorical(train_y, num_classes=(category))
test_y2=tf.keras.utils.to_categorical(test_y, num_classes=(category))

print("train_x[:4]",train_x[:8])
print("train_y[:4]",train_y[:8])
print("train_y2[:4]",train_y2[:8])


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu,
    input_dim=dim))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=1000,
    activation=tf.nn.relu ))
model.add(tf.keras.layers.Dense(units=category,
    activation=tf.nn.softmax ))
model.compile(optimizer='adam',
    loss=tf.keras.losses.categorical_crossentropy,
    metrics=['accuracy'])
model.fit(train_x, train_y2,
          epochs=300,
          batch_size=100)

#測試
score = model.evaluate(test_x, test_y2, batch_size=128)
print("score:",score)

predict = model.predict(test_x)
print("Ans:",np.argmax(predict,axis=-1))






