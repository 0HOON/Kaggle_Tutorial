# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Convolutional neural networks (CNN)
# =====================
#
#
# >  resnetV1 모델을 베이스로 이미지 분류기를 학습시켜보자!

import os, warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_hub as hub

# 이것저것 import

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='inferno')

# matplotlib 기본 설정

train_dataset = image_dataset_from_directory(
    '../car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True
)
valid_dataset = image_dataset_from_directory(
    '../car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False
)


# keras의 image_dataset_from_directory로 데이터를 불러온다
#
# * labels : 'inferred', None, list/tuple
# * label_mode : 분류 방식
# * image_size : 이미지 크기
# * interpolation : 빈칸 채우기
# * batch_size : 배치 크기
# * shuffle : 순서 무작위로

# +
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_dataset = (
    train_dataset
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

valid_dataset = (
    valid_dataset
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

# tf.data.Dataset 객체의 map, cache, prefetch 메소드로 학습에 맞게 preprocessing

base_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/resnet_v1_152/feature_vector/5", 
                            trainable=False)

# tensorflow_hub에서 resnet_v1_152 모델을 불러온다. 
#
# 미리 학습된 모델을 가져오므로 이 부분은 학습시키지 않는다.

# +
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    base_model,
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
# -

# 모델 구성. resnet_v1 모델에 아주 간단한 Dense layer 두 개를 쌓았다.

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# 모델 컴파일. ADAM optimizer를 사용하고 이진 분류에 맞는 loss function과 metrics 설정

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=30
)

# 모델 훈련. 

import pandas as pd
history_frame = pd.DataFrame(history.history)
history_frame.head()

history_frame.loc[:,['loss', 'val_loss']].plot()
history_frame.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()

# 그래프를 보니 확실히 overfitting되었다. 
#
# 이미 학습된 모델을 베이스로 해서 그런지 5번째 epoch근처에서부터 validation loss가 올라가는 모습이 보인다.
#
# dropout layer, EarlyStopping을 넣어 수정해보자.

model_dropout = keras.Sequential([
    base_model,
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')    
])

early_stopping=keras.callbacks.EarlyStopping(min_delta=0.001, patience=5, restore_best_weights=True)
model_dropout.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history_dropout = model_dropout.fit(
    train_dataset,
    validation_data=valid_dataset,
    callbacks=[early_stopping],
    epochs=30
)

history_dropout_df = pd.DataFrame(history_dropout.history)
history_dropout_df.loc[:, ['loss', 'val_loss']].plot()
history_dropout_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()

# overfitting이 일어나기 전에 멈춘 모습을 확인할 수 있다. 약 87%의 정확도를 달성한 모습.
