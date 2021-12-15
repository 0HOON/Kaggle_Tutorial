# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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

# Data Augmentation
# =========
#
# `05.Custom Convnets` 에서 마지막에 잠깐 다뤘던 `Data Augmentation`을 살펴보자. 
#
# 모델의 성능에 직결되는 요소 중 하나는 당연하게도 학습에 사용되는 데이터의 질과 양이다. 그 중 양을 늘리기 위한 간단한 방법 중 하나가 Data Augmentation이다. 기존에 갖고 있는 데이터를 뒤집거나 살짝 기울이거나, 색, 대비를 조절하여 새로운 데이터를 추가하는 것이다. 
#
# 이 때, 데이터의 성질을 고려하여 변형의 정도를 정해야 한다. 예를 들어 손글씨 분류를 위한 학습데이터에 뒤집거나 기울이는 변형을 적용한다면 6과 9는 구별하기 어려워지는 사태가 발생할 것이다.

# +
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing import image_dataset_from_directory

import matplotlib.pyplot as plt

import numpy as np

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='cividis')

ds_train_ = image_dataset_from_directory(
    '../car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True
)

ds_val_ = image_dataset_from_directory(
    '../car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False
)

def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_val = (
    ds_val_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
# -

# 이것저것 `import` 및 데이터셋을 불러온다.

# +
ex = next(iter(ds_train.unbatch().map(lambda x, y: x).batch(1)))

augment = [
    keras.Sequential([
        preprocessing.RandomContrast(factor=0.5)
    ]),
    keras.Sequential([
        preprocessing.RandomFlip(mode='horizontal_and_vertical')
    ]),
    keras.Sequential([
        preprocessing.RandomWidth(factor=0.15)
    ]),
    keras.Sequential([
        preprocessing.RandomRotation(factor=0.2)
    ]),
    keras.Sequential([
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)
    ])
]

plt.figure(figsize=(20,20))
for i in range(5):
    for j in range(5):
        image = augment[i](ex, training=True)
        plt.subplot(5, 5, i*5 + j + 1)
        plt.imshow(tf.squeeze(image))
        plt.axis('off')
plt.show()
# -

# `RandomContrast`,`RandomFlip`,`RandomWidth`,`RandomRotation`,`RandomTranslation`의 효과는 위와 같다.

# 차, 트럭 데이터에 대해서는 RandomContrast, RandomFlip, RandomRotation, RandomTranslation이 모두 효과 있을 것 같다. 저번에는 데이터셋에 먼저 처리를 한 후 그 데이터를 학습시켰는데, 데이터 처리 과정을 모델에 포함하면 각 배치마다 새로 처리된 데이터로 학습될 것이므로 더 많은 데이터로 학습시키는 효과가 있을 것이다. 
#
# 또한 저번 학습 과정에서 loss = 0.6808 근처에서 학습이 전혀 진행이 되지 않았던 경우가 있었는데, initialization시의 값이나 learning rate로 인해 local optimum에 빠져서 그랬던 것이 아닐까 싶다. 이런 문제를 완화할 수 있도록 Batch noermalization도 적용시켜보자.

# +
my_conv_model = keras.Sequential([
    
    layers.InputLayer(input_shape=[128, 128, 3]),
    
    preprocessing.RandomContrast(factor=0.2),
    preprocessing.RandomFlip(mode='horizontal'),
    preprocessing.RandomRotation(factor=0.1),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),

    layers.BatchNormalization(),
    layers.Conv2D(filters=16, kernel_size=7, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.BatchNormalization(),
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.BatchNormalization(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    
])
# -

my_conv_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# +
early = keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

history = my_conv_model.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=50
)
# -

import pandas as pd
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
history_df.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
print("Highest validation accuracy: {}".format(history_df.val_binary_accuracy.max()))

# 학습 과정에서 학습이 멈추는 문제도 없었고, 정확도는 91.9%를 달성하며 만족스러운 결과를 얻었다. 다양한 데이터셋 및 batch normalization의 효과를 체감할 수 있는 결과이다. 
