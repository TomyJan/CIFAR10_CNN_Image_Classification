#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import os

# 创建test_images目录（如果不存在）
os.makedirs('test_images', exist_ok=True)

# 加载CIFAR10数据集
(_, _), (x_test, y_test) = cifar10.load_data()

# 类别标签
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 为每个类别保存一张示例图片
for class_id in range(10):
    # 找到该类别的第一张图片
    idx = (y_test == class_id).flatten().nonzero()[0][0]
    img = x_test[idx]
    
    # 保存图片
    plt.imsave(f'test_images/{labels[class_id]}_sample.png', img)
