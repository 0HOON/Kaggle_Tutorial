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

# Custom Convnets
# =======
#
# 앞서 알아본 feature extraction을 적용해서 resnet 같은 base model을 직접 쌓아보자. `keras`를 이용해 간단하게 할 수 있다.

# +
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='plasma')

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

# 이것저것 import 및 데이터셋 불러와 전처리.

# +
from tensorflow import keras
from tensorflow.keras import layers

my_conv_model = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=62, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
    
])
# -

# 모델을 정의한다. keras로 정의하니 아주 간단하게 끝난다. 4개의 `Conv2D`, `MaxPool2D` 레이어 다음 두 개의 `Dense`층을 두었다. 01.The Convolutional Classifier에서 만들었던 모델과 유사한 모양.
#
#

my_conv_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

# +
early = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

history = my_conv_model.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=50
)

# +
import pandas as pd

df_history = pd.DataFrame(history.history)
df_history.loc[:,['loss', 'val_loss']].plot()
df_history.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
# -

# 17 epoch의 학습 결과 약 84%의 정확도를 달성했다. 미리 학습된 resnet 모델을 이용해 학습했을 때에는 18번의 epoch 끝에 87%의 정확도를 달성했었다. resnet 모델은 학습시킨 것이 아니므로 그에 비해 이번에 학습시켜야할 parameter들이 많았던 것인데, 학습 시간에 큰 차이 없이 비슷한 정확도를 달성했다는 것이 만족스럽다.
#
# 조금 더 정확도를 끌어올려보자. train loss는 계속 줄어드는 반면 validation loss는 치솟는 부분이 보인다. 과적합이 있었던 것은 아닐까? `global average pooling` 레이어를 넣어서 모델을 단순화시켜보자.

# +
my_conv_model_gap = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=62, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAvgPool2D(),
    
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
    
])
# -

my_conv_model_gap.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history_gap = my_conv_model_gap.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=50
)

df_history_gap = pd.DataFrame(history_gap.history)
df_history_gap.loc[:,['loss', 'val_loss']].plot()
df_history_gap.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()

# 학습을 더 진행하여 조금더 높은 정확도를 얻었지만 (86%) validation loss와 accuracy가 튀는 현상은 더 심해졌다. 
#
# 이번엔 복잡성이 부족해져서 정확한 feature를 잡아내지 못하기 때문에 생긴 현상으로 보인다. kernel_size를 좀 건드려보고 filter가 더 많은 레이어를 하나 추가해보자.

# +
my_conv_model_mf = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=5, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=62, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAvgPool2D(),
    
    layers.Dense(6, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
    
])
# -

my_conv_model_mf.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history_mf = my_conv_model_mf.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early],
    epochs=50
)

df_history_mf = pd.DataFrame(history_mf.history)
df_history_mf.loc[:,['loss', 'val_loss']].plot()
df_history_mf.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()

# 크게 달라지지 않았다. 정확도는 83%대로 조금 떨어졌다. 오히려 복잡성이 부족해 특성을 잘 잡아내지 못하기 때문에 생긴 현상이었던 것은 아닐까? 모델을 더 복잡하게 만들어보자.

# +
my_conv_model_mc = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=7, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    
])
# -

my_conv_model_mc.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

early_2 = tf.keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)
history_mc = my_conv_model_mc.fit(
    ds_train,
    validation_data=ds_val,
    callbacks=[early_2],
    epochs=50
)

df_history_mc = pd.DataFrame(history_mc.history)
df_history_mc.loc[:,['loss', 'val_loss']].plot()
df_history_mc.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
print(df_history_mc.val_binary_accuracy.max())

# 정확도 87%로 조금 나은 결과이지만 validation 관련 값들이 튀는 문제가 해결되어 보이진 않는다. `val_loss`와 `val_binary_accuracy`는 여전히 요동친다. 모델의 복잡성을 늘려도 특성을 잘 못잡아낸다면 학습할 데이터셋을 늘려보면 어떨까? 

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2)
])

ds_train_a = (
    ds_train_
    .map(convert_to_float)
    .map(lambda x, y: (data_augmentation(x, training=True), y))
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)

ds_train_a = ds_train_a.concatenate(ds_train)

# +
my_conv_model_mc_a = keras.Sequential([
    layers.Conv2D(filters=16, kernel_size=7, activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.Conv2D(filters=32, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=64, kernel_size=5, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    
    layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
    layers.GlobalAveragePooling2D(),
    
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    
])
# -

my_conv_model_mc_a.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
)

history_mc_a = my_conv_model_mc_a.fit(
    ds_train_a,
    validation_data=ds_val,
    callbacks=[early_2],
    epochs=50
)

df_history_mc_a = pd.DataFrame(history_mc_a.history)
df_history_mc_a.loc[:,['loss', 'val_loss']].plot()
df_history_mc_a.loc[:,['binary_accuracy', 'val_binary_accuracy']].plot()
print(df_history_mc_a.val_binary_accuracy.max())

# 그다지 도움이 되진 않았다. 오히려 마지막 부분에서 과적합이 발생했다. 최종적으로 달성한 정확도 역시 85%정도이다. 데이터셋을 늘려서 해결할 수 있는 문제가 아닌 것으로 보인다. 유용한 feature를 뽑아내기 위해서는 layer의 개수와 구조 변경이 필요하다고 생각된다. 이 챕터는 이정도로 마무리하고 넘어가자.
