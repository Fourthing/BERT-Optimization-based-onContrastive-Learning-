""""
# 1
import torch

# 这是使用csv文件进行测试的代码
from data_Loader import load_corpus
from sentence_embedder import SentenceEmbedder
from training import train_contrastive_learning
from metrics import compute_alignment, compute_uniformity
from visualization import plot_metrics


# 设置模型路径和参数
data_path = "./data/sentence_pairs.csv"
model_name = "bert-large-uncased"
epochs = 10
learning_rate = 2e-4

# 1. 加载数据
corpus = load_corpus(data_path)

# 2. 初始化句子表示模型
embedder = SentenceEmbedder(model_name)

    # 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到GPU
embedder.model.to(device)

# 3. 提取句子嵌入
sentence_embeddings = embedder.get_sentence_embeddings(corpus)

# 4. 计算初始对齐性和均匀性
alignment = compute_alignment(sentence_embeddings)
uniformity = compute_uniformity(sentence_embeddings)
print(f"Initial Alignment: {alignment}, Initial Uniformity: {uniformity}")

# 5. 进行对比学习训练
trained_embeddings, training_metrics = train_contrastive_learning(
    corpus, embedder, epochs, learning_rate
)

# 6. 训练后的对齐性和均匀性
final_alignment = compute_alignment(trained_embeddings)
final_uniformity = compute_uniformity(trained_embeddings)
print(f"Final Alignment: {final_alignment}, Final Uniformity: {final_uniformity}")

# 7. 可视化训练过程中的损失和对齐性、均匀性
plot_metrics(training_metrics)
"""


"""
# 使用MRPC数据集

# from datasets import load_dataset
# from transformers import BertModel, BertTokenizer
# from sentence_embedder import SentenceEmbedder
# from training import train_contrastive_learning
# from visualization import plot_metrics
#
# # 1. 加载 MRPC 数据集
# dataset = load_dataset("glue", "mrpc", split="train")
#
# # 2. 加载预训练的 BERT 模型和分词器
# model = BertModel.from_pretrained("bert-base-uncased")  # 修改为 BertModel
# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#
#
# # 3. 编写数据预处理函数，对句子进行标记化
# def encode(examples):
#     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
#
#
# dataset = dataset.map(encode, batched=True)
#
# # 4. 将 label 列重命名为 labels
# dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
#
# # 5. 提取句子对以便送入嵌入器
# sentences = [(example['sentence1'], example['sentence2']) for example in dataset]
#
# # 6. 实例化句子嵌入器
# embedder = SentenceEmbedder(model, tokenizer)
#
# # 7. 开始对比学习训练
# trained_embeddings, training_metrics = train_contrastive_learning(
#     sentences=sentences,
#     embedder=embedder,
#     epochs=10,  # 可以根据需要调整
#     learning_rate=1e-5
# )
#
# # 8. 可视化训练结果
# plot_metrics(training_metrics)
"""



# import torch
# from batch_data_loader import load_corpus_in_batches  # 批量加载的函数
# from sentence_embedder import SentenceEmbedder
# from training import train_contrastive_learning
# from metrics import compute_alignment, compute_uniformity
# from visualization import plot_metrics
#
# # 设置模型路径和参数
# data_path = "./data/sentence_pairs.csv"
# model_name = "bert-large-uncased"
# epochs = 10
# learning_rate = 2e-4
# batch_size = 32  # 批量大小
#
# # 1. 加载数据（批量处理），指定batch_size
# corpus_batches = load_corpus_in_batches(data_path, batch_size)
#
# # 2. 初始化句子表示模型
# embedder = SentenceEmbedder(model_name)
#
# # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # 将模型移动到GPU
# embedder.model.to(device)
#
# # 3. 提取句子嵌入并计算初始对齐性和均匀性
# initial_embeddings = []
# for batch in corpus_batches:
#     # 提取批量的句子嵌入
#     sentence_embeddings = embedder.get_sentence_embeddings(batch)
#     initial_embeddings.append(sentence_embeddings)  # 累积批次嵌入
#
# # 合并所有批次的嵌入为一个Tensor
# initial_embeddings = torch.cat(initial_embeddings, dim=0)
#
# # 计算初始对齐性和均匀性
# alignment = compute_alignment(initial_embeddings)
# uniformity = compute_uniformity(initial_embeddings)
# print(f"Initial Alignment: {alignment}, Initial Uniformity: {uniformity}")
#
# # 4. 进行对比学习训练（批量处理）
# trained_embeddings, training_metrics = train_contrastive_learning(
#     corpus_batches, embedder, epochs, learning_rate
# )
#
# # 5. 训练后的对齐性和均匀性
# final_alignment = compute_alignment(trained_embeddings)
# final_uniformity = compute_uniformity(trained_embeddings)
# print(f"Final Alignment: {final_alignment}, Final Uniformity: {final_uniformity}")
#
# # 6. 可视化训练过程中的损失和对齐性、均匀性
# plot_metrics(training_metrics)

# 这是使用TensorFlow的运行代码
# import os
# import pandas as pd
# from transformers import BertTokenizer, TFBertModel
# import tensorflow as tf
# from tensorflow.keras import backend as K
#
#
# # 设置代理（如果需要）
# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'
#
# # 检测GPU
# print(tf.config.list_physical_devices('GPU'))
# gpu_devices = tf.config.list_physical_devices('GPU')
# if gpu_devices:
#     for device in gpu_devices:
#         tf.config.experimental.set_memory_growth(device, True)
#
# # 读取CSV文件
# data = pd.read_csv('./data/sentence_pairs.csv')
#
# # 分别获取句子和标签
# sentences1 = data['sentence1'].tolist()
# sentences2 = data['sentence2'].tolist()
# labels = data['label'].tolist()
#
# # 加载BERT模型和分词器
# tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
# bert_model = TFBertModel.from_pretrained('bert-large-uncased')
#
# # 定义处理句子对的函数
# def tokenize_sentences(sentences1, sentences2):
#     return tokenizer(sentences1, sentences2, padding=True,truncation=True, return_tensors='tf')
#
# # 对句子对进行BERT编码
# tokenized_data = tokenize_sentences(sentences1, sentences2)
#
# # 获取BERT模型的输出
# bert_output = bert_model([tokenized_data['input_ids'],
#                           tokenized_data['attention_mask'],
#                           tokenized_data['token_type_ids']])
#
# # 取出句子嵌入表示（我们使用[CLS] token的输出作为句子表示）
# sentence_embeddings = bert_output.last_hidden_state[:, 0, :]
#
# # 定义对比学习的损失函数（例如：对比损失）
# def contrastive_loss(y_true, y_pred, margin=1.0):
#     """y_true：1 表示相似，0 表示不相似，y_pred 是句子对的余弦相似度"""
#     square_pred = K.square(y_pred)
#     margin_square = K.square(K.maximum(margin - y_pred, 0))
#     return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
#
# # 编写简单的全连接层模型，用于计算句子对的相似性
# input_layer = tf.keras.layers.Input(shape=(10241,))
# dense_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
# output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
# similarity_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
#
# # 编译模型
# similarity_model.compile(optimizer='adam',
#                          loss=contrastive_loss,
#                          metrics=['accuracy'])
#
# # 训练模型
# similarity_model.fit(x=sentence_embeddings,
#                      y=tf.convert_to_tensor(labels),
#                      epochs=5,
#                      batch_size=16)
#
# # 评估模型
# results = similarity_model.evaluate(x=sentence_embeddings,
#                                     y=tf.convert_to_tensor(labels),
#                                     verbose=2)
#
# print(f"模型评估结果：{results}")

import os
import pandas as pd
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from metrics import compute_alignment
from metrics import compute_uniformity

# 设置代理（如果需要）
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# 检测GPU
print(tf.config.list_physical_devices('GPU'))
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

# 读取CSV文件
data = pd.read_csv('./data/sentence_pairs.csv')

# 分别获取句子和标签
sentences1 = data['sentence1'].tolist()
sentences2 = data['sentence2'].tolist()
labels = data['label'].tolist()

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
bert_model = TFBertModel.from_pretrained('bert-large-uncased')

# 定义处理句子对的函数
def tokenize_sentences(sentences1, sentences2):
    return tokenizer(sentences1, sentences2, padding=True, truncation=True, return_tensors='tf')

# 对句子对进行BERT编码
tokenized_data = tokenize_sentences(sentences1, sentences2)

# 获取BERT模型的输出
bert_output = bert_model([tokenized_data['input_ids'],
                          tokenized_data['attention_mask'],
                          tokenized_data['token_type_ids']])

# 取出句子嵌入表示（我们使用[CLS] token的输出作为句子表示）
sentence_embeddings = bert_output.last_hidden_state[:, 0, :]

# 定义对比学习的损失函数（例如：对比损失）
def contrastive_loss(y_true, y_pred, margin=1.0):
    """y_true：1 表示相似，0 表示不相似，y_pred 是句子对的余弦相似度"""
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# 编写简单的全连接层模型，用于计算句子对的相似性
input_layer = tf.keras.layers.Input(shape=(1024,))
dense_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(dense_layer)
similarity_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

# 编译模型
similarity_model.compile(optimizer='adam',
                         loss=contrastive_loss,
                         metrics=['accuracy'])

# 定义对齐度和均匀度的计算函数

# 回调函数用于每个 epoch 后计算对齐度和均匀度
class AlignmentUniformityCallback(tf.keras.callbacks.Callback):
    def __init__(self, sentence_embeddings):
        super().__init__()
        self.sentence_embeddings = sentence_embeddings
        self.alignments = []
        self.uniformities = []

    def on_epoch_end(self, epoch, logs=None):
        # 获取句子嵌入
        current_embeddings = self.model.predict(self.sentence_embeddings)

        # 计算对齐度和均匀度
        alignment = compute_alignment(current_embeddings)
        uniformity = compute_uniformity(current_embeddings)

        # 保存度量值
        self.alignments.append(alignment)
        self.uniformities.append(uniformity)

        print(f"Epoch {epoch + 1} - Alignment: {alignment}, Uniformity: {uniformity}")


# 开始训练并计算对齐度和均匀度
callback = AlignmentUniformityCallback(sentence_embeddings)
history = similarity_model.fit(x=sentence_embeddings,
                               y=tf.convert_to_tensor(labels),
                               epochs=20,
                               batch_size=16,
                               callbacks=[callback])

# 可视化对齐度和均匀度
epochs = range(1, 21)
plt.figure(figsize=(10, 4))

# 对齐度图
plt.subplot(1, 2, 1)
plt.plot(epochs, callback.alignments, 'bo-', label='Alignment')
plt.title('Alignment over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Alignment')
plt.legend()

# 均匀度图
plt.subplot(1, 2, 2)
plt.plot(epochs, callback.uniformities, 'ro-', label='Uniformity')
plt.title('Uniformity over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Uniformity')
plt.legend()

plt.tight_layout()
plt.show()
