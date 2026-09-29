# 中文新闻文本分类（CNN + BiLSTM）

基于 TensorFlow/Keras 的中文新闻文本分类项目，使用 CNews 数据集，涵盖 10 个新闻类别。模型采用 **CNN + BiLSTM** 混合结构，先通过卷积层提取局部 n-gram 特征，再通过双向 LSTM 捕捉长距离语义依赖，最后经全连接层输出分类结果。

## 类别

体育、财经、房产、家居、教育、科技、时尚、时政、游戏、娱乐

## 项目结构

```text
cnews-text-classification/
├── train.py                # 主程序（原 1.py 重命名）
├── requirements.txt
├── README.md
├── .gitignore
├── cnews/                  # 数据集目录（不上传）
│   ├── cnews.train.txt
│   ├── cnews.val.txt
│   ├── cnews.test.txt
│   └── cnews.vocab.txt
└── outputs/                # 训练输出（自动生成，不上传）
    ├── confusion_matrix.png
    └── training_history.png
```

## 环境要求

- Python 3.8+
- TensorFlow 2.11+
- NumPy、scikit-learn、Matplotlib、Seaborn

安装依赖：

```bash
pip install -r requirements.txt
```

> **注意**：Windows 下 TensorFlow 2.11+ 原生不支持 GPU，程序会自动使用 CPU 训练。如需 GPU，请使用 WSL2 或 TensorFlow-DirectML。

## 数据集准备

请自行获取 CNews 数据集，并将以下文件放入 `cnews/` 目录：

- `cnews.train.txt`
- `cnews.val.txt`
- `cnews.test.txt`
- `cnews.vocab.txt`

数据格式为每行 `标签\t内容`，例如：

```text
体育	姚明正式入选奈史密斯篮球名人堂
财经	央行今日开展逆回购操作
```

## 快速开始

1. 将主程序重命名为 `train.py`（或直接修改 `README` 中的命令）。
2. 确保数据集已放入 `cnews/`。
3. 运行训练：

```bash
python train.py
```

训练过程中会：

- 自动加载并预处理数据
- 构建 CNN + BiLSTM 模型
- 使用 EarlyStopping、ReduceLROnPlateau、ModelCheckpoint 进行训练
- 在测试集上评估，输出分类报告与混淆矩阵
- 绘制训练损失与准确率曲线
- 保存最终模型 `cnews_final_model.h5`

## 模型结构

| 层 | 配置 |
|---|---|
| Embedding | `vocab_size × 128` |
| Conv1D | 128 个滤波器，卷积核 5，ReLU |
| MaxPooling1D | pool_size=5 |
| Conv1D | 128 个滤波器，卷积核 5，ReLU |
| MaxPooling1D | pool_size=5 |
| Bidirectional(LSTM) | 64 个单元 |
| Dropout | 0.5 |
| Dense | 64，ReLU |
| Dropout | 0.5 |
| Dense | 10，Softmax |

## 训练参数

- 最大序列长度：600
- 词向量维度：128
- 批大小：128
- 训练轮数：20
- 优化器：Adam，学习率 0.001
- 损失函数：稀疏类别交叉熵
- 回调：EarlyStopping（patience=3）、ReduceLROnPlateau（factor=0.5, patience=2）、ModelCheckpoint

## 评估与输出

运行结束后，会在测试集上输出：

- 测试集 Loss 与 Accuracy
- 分类报告（Precision / Recall / F1-score）
- 混淆矩阵图 `confusion_matrix.png`
- 训练曲线图 `training_history.png`

## 常见问题

**1. 报错 `indices[x,y] = N is not in [0, vocab_size)`**

说明词汇表去重后 `len(word_to_id)` 小于实际最大索引。请将 `load_data()` 中的 `vocab_size` 改为：

```python
vocab_size = len(id_to_word)
# 或者
vocab_size = max(word_to_id.values()) + 1
```

**2. `input_length` 弃用警告**

Keras 3 中 `Embedding` 的 `input_length` 已弃用，可直接删除该参数。

**3. Windows GPU 警告**

TensorFlow 2.11+ 在原生 Windows 上不支持 GPU，可忽略该警告，或使用 WSL2。

## License

本项目仅供学习与研究使用。CNews 数据集版权归其原作者所有，请勿用于商业用途。
