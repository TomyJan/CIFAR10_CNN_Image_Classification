#!/usr/bin/env python
# coding: utf-8
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['MiSans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 类别标签
labels = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

def load_and_preprocess_image(image_path):
    """加载并预处理单张图片"""
    # 读取图片
    img = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(32, 32)
    )
    # 转换为数组
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # 添加batch维度
    img_array = tf.expand_dims(img_array, 0)
    # 标准化
    img_array = tf.cast(img_array, tf.float32) / 255.0
    
    return img_array

def predict_single_image(model, image_path):
    """预测单张图片的类别"""
    # 预处理图片
    img_array = load_and_preprocess_image(image_path)
    
    # 进行预测
    predictions = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # 显示图片和预测结果
    img = plt.imread(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f'预测类别: {labels[predicted_class]}\n置信度: {confidence:.2%}')
    plt.axis('off')
    plt.show()
    
    # 返回预测结果
    return {
        'predicted_class': labels[predicted_class],
        'confidence': confidence,
        'all_probabilities': dict(zip(labels, predictions[0]))
    }

def main():
    # 检查命令行参数
    import sys
    if len(sys.argv) < 2:
        print("使用方法: python predict.py <图片路径1> [图片路径2 ...]")
        print("示例: python predict.py test_images/airplane.jpg test_images/car.jpg")
        return

    try:
        # 加载模型
        print("正在加载模型...")
        model = keras.models.load_model('saved_models/best_model.keras')
        print("模型加载成功！")

        # 对每张输入图片进行预测
        for image_path in sys.argv[1:]:
            try:
                print(f"\n预测图片: {image_path}")
                result = predict_single_image(model, image_path)
                print(f"预测类别: {result['predicted_class']}")
                print(f"置信度: {result['confidence']:.2%}")
                print("\n各类别概率:")
                for label, prob in result['all_probabilities'].items():
                    print(f"{label}: {prob:.2%}")
            except Exception as e:
                print(f"处理图片 {image_path} 时出错: {str(e)}")

    except Exception as e:
        print(f"错误: {str(e)}")

if __name__ == '__main__':
    main()
