#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function
import keras
from keras.datasets import cifar10 #CIFAR-10 数据集
from keras.preprocessing.image import ImageDataGenerator #图像数据生成器，增强数据
from keras.models import Sequential #，搭建神经网络模型
from keras.layers import Dense, Dropout, Activation, Flatten #，构建神经网络模型
from keras.layers import Conv2D, MaxPooling2D #，构建卷积神经网络模型
import os

import numpy as np

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt #设置参数
plt.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文
plt.rcParams['axes.unicode_minus'] = False # 显示负号

from sklearn.metrics import confusion_matrix, classification_report  #导入混淆矩阵和分类报告模块，评估模型的性能
import itertools

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
sns.countplot(y_train.ravel(), ax=axs[0])
axs[0].set_title('训练集数据分布')
axs[0].set_xlabel('类别')
sns.countplot(y_test.ravel(), ax=axs[1])
axs[1].set_title('测试集数据分布')
axs[1].set_xlabel('类别')
plt.savefig('数据分布.png')
plt.show()

# 数据标准化处理，转换数据类型为float32，并归一化，可以提高模型的训练效果，并使得不同特征对模型的影响更加平衡
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# 将类别向量转换为二进制（one-hot）类别矩阵
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 定义卷积神经网络
model = Sequential()
# 卷积层 => 激活层 => 卷积层 => 激活层 => 池化层 => Dropout层
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 卷积层 => 激活层 => 卷积层 => 激活层 => 池化层 => Dropout层
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 展平层 => 全连接层 => 激活层 => Dropout层
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
# Softmax分类器
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# 显示模型概要
model.summary()

# 初始化RMSprop优化器
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# 使用RMSprop训练模型
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = None  # 用于记录训练过程的历史信息
if not data_augmentation:
    print('未使用数据增强。')
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        shuffle=True)
else:
    print('使用实时数据增强。')
    # 进行预处理和实时数据增强：
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 在数据集上将输入均值设为0
        samplewise_center=False,  # 将每个样本的均值设为0
        featurewise_std_normalization=False,  # 将输入除以数据集的标准差以进行标准化
        samplewise_std_normalization=False,  # 将每个输入除以其标准差
        zca_whitening=False,  # 应用ZCA白化
        zca_epsilon=1e-06,  # ZCA白化的epsilon值
        rotation_range=0,  # 随机旋转图片的角度范围（0到180度）
        width_shift_range=0.1,  # 随机水平移动图片的范围（总宽度的比例）
        height_shift_range=0.1,  # 随机垂直移动图片的范围（总高度的比例）
        shear_range=0.,  # 随机剪切变换的范围
        zoom_range=0.,  # 随机缩放图片的范围
        channel_shift_range=0.,  # 随机通道移动的范围
        fill_mode='nearest',  # 输入边界之外的点的填充模式
        cval=0.,  # fill_mode为"constant"时的值
        horizontal_flip=True,  # 随机水平翻转图片
        vertical_flip=False,  # 随机垂直翻转图片
        rescale=None,  # 应用于每个输入的重缩放因子
        preprocessing_function=None,  # 应用于每个输入的预处理函数
        data_format=None,  # 图像数据格式，"channels_first"或"channels_last"
        validation_split=0.0)  # 保留用于验证的图像的比例（0到1之间）

    # 计算特征归一化所需的数量
    datagen.fit(x_train)

    # 使用datagen.flow()生成的批次来拟合模型。
    history = model.fit_generator(datagen.flow(x_train, y_train,
                                               batch_size=batch_size),
                                  epochs=epochs,
                                  validation_data=(x_test, y_test),
                                  workers=4)

# 定义绘制训练历史的函数
def plotmodelhistory(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    # 总结精度的历史
    axs[0].plot(history.history['accuracy'])
    axs[0].plot(history.history['val_accuracy'])
    axs[0].set_title('模型精度')
    axs[0].set_ylabel('精度')
    axs[0].set_xlabel('周期')
    axs[0].legend(['训练', '验证'], loc='upper left')
    # 总结损失的历史
    axs[1].plot(history.history['loss'])
    axs[1].plot(history.history['val_loss'])
    axs[1].set_title('模型损失')
    axs[1].set_ylabel('损失')
    axs[1].set_xlabel('周期')
    axs[1].legend(['训练', '验证'], loc='upper left')
    plt.show()

# 列出历史数据中的所有数据
print(history.history.keys())

# 绘制训练历史
plotmodelhistory(history)

# 评估训练好的模型
scores = model.evaluate(x_test, y_test, verbose=1)
print('测试损失:', scores[0])
print('测试精度:', scores[1])

# 进行预测
pred = model.predict(x_test)



def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    根据numpy数组和两个标签列表创建热图。
    """
    if not ax:
        ax = plt.gca()

    # 绘制热图
    im = ax.imshow(data, **kwargs)

    # 创建颜色条
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # 设置水平轴标签显示在顶部
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    # 显示所有刻度...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ...并用相应的列表项标记它们
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel('预测标签')
    ax.set_ylabel('真实标签')

    return im, cbar


def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    """
    注释热图的函数。
    """
    # 根据数据更改文本的颜色
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt),
                                horizontalalignment="center",
                                color="white" if data[i, j] > threshold else "black")
            texts.append(text)

    return texts

# 定义标签
labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

# 将预测的类别转换为one-hot向量
Y_pred_classes = np.argmax(pred, axis=1)
# 将测试集的观察结果转换为one-hot向量
Y_true = np.argmax(y_test, axis=1)
# 错误是预测标签与真实标签之间的差异
errors = (Y_pred_classes - Y_true != 0)

Y_pred_classes_errors = Y_pred_classes[errors]
Y_pred_errors = pred[errors]
Y_true_errors = Y_true[errors]
X_test_errors = x_test[errors]

# 计算混淆矩阵
cm = confusion_matrix(Y_true, Y_pred_classes)
thresh = cm.max() / 2.

# 绘制混淆矩阵热图
fig, ax = plt.subplots(figsize=(12, 12))
im, cbar = heatmap(cm, labels, labels, ax=ax, cmap=plt.cm.Blues, cbarlabel="预测次数")
texts = annotate_heatmap(im, data=cm, threshold=thresh)

fig.tight_layout()
plt.savefig("混淆矩阵.png")
plt.show()

# 打印分类报告
print(classification_report(Y_true, Y_pred_classes))

# 显示正确分类的图片
R = 5
C = 5
fig, axes = plt.subplots(R, C, figsize=(12, 12))
axes = axes.ravel()

for i in np.arange(0, R * C):
    axes[i].imshow(x_test[i])
    axes[i].set_title("真实: %s \n预测: %s" % (labels[Y_true[i]], labels[Y_pred_classes[i]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)

# 显示被错误分类的图片
R = 3
C = 5
fig, axes = plt.subplots(R, C, figsize=(12, 8))
axes = axes.ravel()

misclassified_idx = np.where(Y_pred_classes != Y_true)[0]
for i in np.arange(0, R * C):
    axes[i].imshow(x_test[misclassified_idx[i]])
    axes[i].set_title("真实: %s \n预测: %s" % (labels[Y_true[misclassified_idx[i]]],
                                             labels[Y_pred_classes[misclassified_idx[i]]]))
    axes[i].axis('off')
    plt.subplots_adjust(wspace=1)

def display_errors(errors_index, img_errors, pred_errors, obs_errors):
    """ 显示10张带有预测和真实标签的图片 """
    n = 0
    nrows = 2
    ncols = 5
    fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True, figsize=(12, 6))
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row, col].imshow((img_errors[error]).reshape((32, 32, 3)))
            ax[row, col].set_title("预测: {}\n真实: {}".
                                   format(labels[pred_errors[error]], labels[obs_errors[error]]))
            n += 1
            ax[row, col].axis('off')
            plt.subplots_adjust(wspace=1)

# 计算错误预测的概率
Y_pred_errors_prob = np.max(Y_pred_errors, axis=1)

# 计算错误集中真实值的预测概率
true_prob_errors = np.diagonal(np.take(Y_pred_errors, Y_true_errors, axis=1))

# 计算预测标签和真实标签的概率差异
delta_pred_true_errors = Y_pred_errors_prob - true_prob_errors

# 对概率差异进行排序
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# 最重要的10个错误
most_important_errors = sorted_dela_errors[-10:]

# 显示最重要的10个错误
display_errors(most_important_errors, X_test_errors, Y_pred_classes_errors, Y_true_errors)

# 测试集中的图片测试模型
def show_test(number):
    fig = plt.figure(figsize=(3, 3))
    test_image = np.expand_dims(x_test[number], axis=0)
    test_result = model.predict(test_image)
    predicted_class = np.argmax(test_result, axis=1)
    plt.imshow(x_test[number])
    dict_key = predicted_class[0]
    plt.title("预测: {} \n真实标签: {}".format(labels[dict_key],
                                             labels[Y_true[number]]))

show_test(20)

# 保存模型和权重
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('已在 %s 保存训练好的模型' % model_path)

# 评估训练好的模型
scores = model.evaluate(x_test, y_test, verbose=1)
print('测试损失:', scores[0])
print('测试精度:', scores[1])
