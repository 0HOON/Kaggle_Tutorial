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

# Convolution and ReLU
# ====================
#
# 이전에 학습시켜봤던 모델에서 **resnetV1**과 같은 base model은 이미지의 *특성을 추출*해내는 역할을 한다. 
#
# 1. 특성을 추출하기 위한 *필터* 적용 (convolution)
# 2. 특성을 *감지* (ReLU 이용)
# 3. 특성 *강화* (maximum pooling)
#
# 의 과정을 거치는데, 이번에는 1, 2번 과정을 해보며 **feature extraction**에 대한 감을 잡아보자. 

# +
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large', titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')

# -

# 이것저것 import 및 matplotlib 기본 설정해주면서 시작.

# +
image_path = './Nekko.jpg'
image = tf.io.read_file(image_path)
image = tf.io.decode_jpeg(image, channels=1)
image = tf.image.resize(image, size=[400, 400])

img = tf.squeeze(image).numpy()
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.axis('off')
plt.show()
# -

# 이미지를 불러온다.
# 텐서를 따라 이미지를 불러와서 grayscale(channels=1)로 디코딩. 이미지 크기도 조정해준다.
# `tf.squeeze()`로 2차원 array로 변경하여 이미지 확인도 해본다.

# ##  Filter

# +
names = ["Edge Detect", "Bottom Sobel", "Emboss", "Sharpen"]
kernels = [
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]],
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]],
    [[-2, -1, 0],
     [-1, 1, 1],
     [0, 1, 2]],
    [[0, -1, 0],
     [-1, 5, -1],
     [0, -1, 0]]
]

for i, (name, kernel) in enumerate(zip(names, kernels)):
    print('{} : '.format(name))
    for r in kernel:
        print(r)
    
# -

# 필터들의 예이다. 가장자리, 경계선, 양각, 선명함 등의 필터들이다.
# 이 필터를 각 픽셀에 적용해 모두 더한 값으로 그 필터에 해당하는 **특성을 추출**한다.

# +
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

for i,k in enumerate(kernels):
    kernels[i] = tf.squeeze(tf.constant(k))

plt.figure(figsize=(12,12))
for i, (name,kernel) in enumerate(zip(names, kernels)):
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.cast(kernel, dtype=tf.float32)
    image_filter = tf.nn.conv2d(
        input=image,
        filters=kernel,
        strides=1,
        padding='SAME'
    )
    plt.subplot(1, 4, i+1)
    plt.imshow(tf.squeeze(image_filter))
    plt.title(name)
plt.tight_layout()
# -

# 각 필터들을 이미지에 적용한 예시들이다. 필터에 따라서 강조되는 부분이 다른 것이 보인다.

# ## ReLU

plt.figure(figsize=(12,12))
for i, (name,kernel) in enumerate(zip(names, kernels)):
    kernel = tf.reshape(kernel, [*kernel.shape, 1, 1])
    kernel = tf.cast(kernel, dtype=tf.float32)
    image_filter = tf.nn.conv2d(
        input=image,
        filters=kernel,
        strides=1,
        padding='SAME'
    )
    image_detect = tf.nn.relu(image_filter)
    plt.subplot(1, 4, i+1)
    plt.imshow(tf.squeeze(image_detect))
    plt.title(name)
plt.tight_layout()

# 앞 단계에서 추출해본 이미지에 ReLU 함수를 적용하여 **특성을 감지**한다.
# ReLU 함수는 0 이하의 값은 모두 똑같이 0으로 취급하는 함수이므로 특성 값이 *0보다 작은 부분은 그 특성을 가지고 있지 않다*고 보는 것. 
#
# Edge Detect는 거의 보이지 않는다..
