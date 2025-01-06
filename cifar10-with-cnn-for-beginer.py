#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10 #CIFAR-10 数据集
from tensorflow.keras.preprocessing.image import ImageDataGenerator #图像数据生成器，增强数据
from tensorflow.keras.models import Sequential #，搭建神经网络模型
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten #，构建神经网络模型
from tensorflow.keras.layers import Conv2D, MaxPooling2D #，构建卷积神经网络模型

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt #设置参数
plt.rcParams['font.sans-serif'] = ['MiSans', 'SimHei', 'Microsoft YaHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号

from sklearn.metrics import confusion_matrix, classification_report  #导入混淆矩阵和分类报告模块，评估模型的性能
import pandas as pd

# 设置Keras的批处理大小
batch_size = 32
# 数据集的类别数
num_classes = 10
# 训练轮次
epochs = 60
# 是否使用数据增强
data_augmentation = False

# 加载CIFAR10数据集，并划分为训练集和测试集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 绘制训练集和测试集的类别分布
fig, axs = plt.subplots(1, 2, figsize=(15, 5))
sns.countplot(data=pd.DataFrame(y_train, columns=['label']), x='label', ax=axs[0])
axs[0].set_title('训练集数据分布')
axs[0].set_xlabel('类别')
sns.countplot(data=pd.DataFrame(y_test, columns=['label']), x='label', ax=axs[1])
axs[1].set_title('测试集数据分布')
axs[1].set_xlabel('类别')
plt.savefig('数据分布.png')
plt.show()

# 数据标准化处理
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 将类别向量转换为二进制（one-hot）类别矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 定义卷积神经网络
model = Sequential([
    Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]),
    Activation('relu'),
    Conv2D(32, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    Conv2D(64, (3, 3)),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512),
    Activation('relu'),
    Dropout(0.5),
    Dense(num_classes),
    Activation('softmax')
])

# 显示模型概要
model.summary()

# 编译模型
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 添加早停回调
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 添加模型保存回调
model_checkpoint = keras.callbacks.ModelCheckpoint(
    filepath='saved_models/best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=False,
    mode='max',
    verbose=1
)

# 训练模型
if not data_augmentation:
    print('未使用数据增强。')
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=[early_stopping, model_checkpoint]
    )
else:
    print('使用实时数据增强。')
    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=False
    )
    datagen.fit(x_train)
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=(x_test, y_test),
        workers=4,
        callbacks=[early_stopping, model_checkpoint]
    )

# 绘制训练历史
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('模型精度')
plt.ylabel('精度')
plt.xlabel('周期')
plt.legend(['训练', '验证'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('周期')
plt.legend(['训练', '验证'], loc='upper left')
plt.savefig('acc_loss.png')
plt.show()

# 评估模型
scores = model.evaluate(x_test, y_test, verbose=1)
print('测试损失:', scores[0])
print('测试精度:', scores[1])

# 预测
pred = model.predict(x_test)

# 计算混淆矩阵
Y_pred = np.argmax(pred, axis=1)
Y_true = np.argmax(y_test, axis=1)
cm = confusion_matrix(Y_true, Y_pred)

# 绘制混淆矩阵
plt.figure(figsize=(12, 12))
labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels,
            yticklabels=labels)
plt.title('混淆矩阵')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.savefig('混淆矩阵.png')
plt.show()

# 打印分类报告
print('\n分类报告:')
print(classification_report(Y_true, Y_pred, target_names=labels))
