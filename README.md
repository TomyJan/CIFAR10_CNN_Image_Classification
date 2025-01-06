# CIFAR10 CNN 图像分类

基于CNN（卷积神经网络）的CIFAR10数据集图像分类项目，适合深度学习初学者学习和实践。

## 项目概述

本项目使用卷积神经网络（CNN）对CIFAR10数据集进行图像分类。CIFAR10是一个包含60000张32x32彩色图像的数据集，共有10个类别，每个类别6000张图像。这些类别包括：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船和卡车。

## 主要功能

- 数据加载和预处理
- CNN模型构建和训练
- 模型评估和性能可视化
- 预测结果分析和混淆矩阵生成

## 网络架构

项目使用了一个经典的CNN架构，具体包括：

1. 第一个卷积块：
   - 两个卷积层（32个3x3卷积核）
   - ReLU激活函数
   - 最大池化层
   - Dropout(0.25)防止过拟合

2. 第二个卷积块：
   - 两个卷积层（64个3x3卷积核）
   - ReLU激活函数
   - 最大池化层
   - Dropout(0.25)防止过拟合

3. 全连接层：
   - Flatten层
   - Dense层（512个神经元）
   - Dropout(0.5)
   - 输出层（10个类别）

## 训练参数

- 批次大小：32
- 训练轮次：60（使用早停机制，当验证集损失在5个epoch内没有改善时停止训练）
- 优化器：RMSprop
- 学习率：0.0001
- 支持数据增强（可选）
- 模型保存：自动保存验证集准确率最高的模型

## 项目结构

```
├── train_cifar10_cnn.py              # 主要Python代码文件
├── requirements.txt                    # 项目依赖文件
├── README.md                          # 项目文档
├── LICENSE                            # 许可证文件
├── saved_models/                      # 保存训练模型的目录
│   └── best_model.keras               # 训练过程中的最佳模型
├── 数据分布.png                        # 数据集分布可视化
├── 混淆矩阵.png                        # 模型预测结果混淆矩阵
├── acc_loss.png                       # 训练过程准确率和损失曲线
└── classification_report.txt          # 模型评估报告（包含详细的分类指标）
```

## 可视化结果

项目提供了三种可视化结果和一个详细的评估报告：
1. 数据分布图：显示训练集和测试集中各类别的数据分布情况
2. 训练过程图：展示模型训练过程中的准确率和损失变化
3. 混淆矩阵：直观显示模型在各个类别上的预测效果
4. 分类报告：包含每个类别的精确率、召回率、F1分数等详细指标

## 环境要求

项目依赖以下主要包：
- TensorFlow >= 2.16.1
- NumPy >= 1.26.0
- Pandas >= 2.2.0
- Matplotlib >= 3.9.0
- Seaborn >= 0.13.0
- Scikit-learn >= 1.5.0

详细的依赖要求请参见 `requirements.txt`。

## 使用说明

1. 克隆项目到本地
2. 安装所需依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 运行 `train_cifar10_cnn.py`

## 学习要点

通过本项目，您可以学习：
- CNN的基本架构和实现方法
- 图像分类任务的完整处理流程
- 模型评估和可视化技术
- 深度学习项目的最佳实践

## 注意事项

- 首次运行时会自动下载CIFAR10数据集
- 训练过程可能需要较长时间
- 使用了早停机制来防止过拟合
- 可以通过调整参数来优化模型性能

## 最近更新

- 优化了项目结构，删除了冗余文件
- 更新了依赖包版本要求
- 改进了数据分布可视化方法
- 简化了代码结构，提高了可读性
- 添加了自动保存最佳模型功能
- 优化了文件命名，使其更加规范和专业
