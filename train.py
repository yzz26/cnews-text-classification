# -*- coding: utf-8 -*-
"""
基于 TensorFlow/Keras 的中文新闻文本分类（CNN + BiLSTM 混合模型）
使用 CNews 数据集：体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以保证可复现性
tf.random.set_seed(42)
np.random.seed(42)

# ================================ 1. 数据加载与预处理 ================================

def read_vocab(vocab_path):
    """读取词汇表，构建 字->id 和 id->字 的映射"""
    with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as f:
        words = [line.strip() for line in f.readlines()]
    word_to_id = {word: idx for idx, word in enumerate(words)}
    id_to_word = words
    return word_to_id, id_to_word

def read_category():
    """类别列表及 类别->id 映射"""
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cat_to_id = {cat: idx for idx, cat in enumerate(categories)}
    return categories, cat_to_id

def process_file(file_path, word_to_id, cat_to_id, max_length=600):
    """
    将原始数据文件转换为 id 序列和标签
    每行格式：标签\t内容
    """
    data_ids = []
    label_ids = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label, content = line.split('\t')
            except ValueError:
                continue
            # 将每个字符转换为对应id（OOV字符忽略）
            seq = [word_to_id.get(char, 0) for char in content]  # 0 通常对应 <UNK>
            # 截断或保留
            seq = seq[:max_length]
            data_ids.append(seq)
            label_ids.append(cat_to_id[label])
    # 填充至统一长度
    data_ids = pad_sequences(data_ids, maxlen=max_length, padding='post', truncating='post')
    label_ids = np.array(label_ids)
    return data_ids, label_ids

def load_data(data_dir, max_length=600):
    """加载全部数据集"""
    vocab_path = os.path.join(data_dir, 'cnews.vocab.txt')
    train_path = os.path.join(data_dir, 'cnews.train.txt')
    val_path = os.path.join(data_dir, 'cnews.val.txt')
    test_path = os.path.join(data_dir, 'cnews.test.txt')

    word_to_id, id_to_word = read_vocab(vocab_path)
    categories, cat_to_id = read_category()
    vocab_size = len(id_to_word)

    x_train, y_train = process_file(train_path, word_to_id, cat_to_id, max_length)
    x_val, y_val = process_file(val_path, word_to_id, cat_to_id, max_length)
    x_test, y_test = process_file(test_path, word_to_id, cat_to_id, max_length)

    print(f"训练集: {x_train.shape}, {y_train.shape}")
    print(f"验证集: {x_val.shape}, {y_val.shape}")
    print(f"测试集: {x_test.shape}, {y_test.shape}")
    return (x_train, y_train), (x_val, y_val), (x_test, y_test), word_to_id, id_to_word, categories, vocab_size


# ================================ 2. 模型构建 ================================

def build_cnn_bilstm_model(vocab_size, embedding_dim=128, max_length=600, num_classes=10):
    """
    构建 CNN + BiLSTM 混合文本分类模型
    """
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length, name='embedding'),
        Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1'),
        MaxPooling1D(pool_size=5, name='pool1'),
        Conv1D(filters=128, kernel_size=5, activation='relu', name='conv2'),
        MaxPooling1D(pool_size=5, name='pool2'),
        Bidirectional(LSTM(units=64, return_sequences=False), name='bilstm'),
        Dropout(0.5, name='dropout1'),
        Dense(64, activation='relu', name='dense1'),
        Dropout(0.5, name='dropout2'),
        Dense(num_classes, activation='softmax', name='output')
    ])
    return model


# ================================ 3. 训练与评估 ================================

def train_model(model, x_train, y_train, x_val, y_val, batch_size=128, epochs=20):
    """编译并训练模型"""
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        ModelCheckpoint('best_cnews_model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, x_test, y_test, categories):
    """在测试集上评估并打印报告、绘制混淆矩阵"""
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 分类报告
    print("\n========== 分类报告 ==========")
    print(classification_report(y_test, y_pred, target_names=categories, digits=4))

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()

    return y_pred

def plot_training_history(history):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Val Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Val Acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()


# ================================ 主程序 ================================

def main():
    # 配置路径（请根据实际存放位置修改）
    data_dir = './cnews/'    # 确保该目录下有 cnews.train.txt, cnews.val.txt, cnews.test.txt, cnews.vocab.txt
    max_length = 600
    embedding_dim = 128
    batch_size = 128
    epochs = 20

    # 1. 加载数据
    print("========== 加载数据 ==========")
    (x_train, y_train), (x_val, y_val), (x_test, y_test), word_to_id, id_to_word, categories, vocab_size = load_data(data_dir, max_length)
    num_classes = len(categories)
    print(f"词汇表大小: {vocab_size}, 类别数: {num_classes}")

    # 2. 构建模型
    print("\n========== 构建模型 ==========")
    model = build_cnn_bilstm_model(vocab_size, embedding_dim, max_length, num_classes)
    model.summary()

    # 3. 训练模型
    print("\n========== 开始训练 ==========")
    history = train_model(model, x_train, y_train, x_val, y_val, batch_size, epochs)

    # 4. 评估
    print("\n========== 模型评估 ==========")
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集 Loss: {test_loss:.4f}, 测试集 Accuracy: {test_acc:.4f}")

    # 5. 详细分类报告与混淆矩阵
    y_pred = evaluate_model(model, x_test, y_test, categories)

    # 6. 绘制训练过程曲线
    plot_training_history(history)

    # 7. 保存最终模型
    model.save('cnews_final_model.h5')
    print("\n模型已保存为 cnews_final_model.h5")

if __name__ == '__main__':
    main()