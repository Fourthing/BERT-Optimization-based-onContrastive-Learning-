# import torch
#
# # 这是使用csv文件进行测试的代码
# from data_Loader import load_corpus
# from sentence_embedder import SentenceEmbedder
# from training import train_contrastive_learning
# from metrics import compute_alignment, compute_uniformity
# from visualization import plot_metrics
#
#
# # 设置模型路径和参数
# data_path = "./data/sentence_pairs.csv"
# model_name = "bert-large-uncased"
# epochs = 10
# learning_rate = 2e-4
#
# # 1. 加载数据
# corpus = load_corpus(data_path)
#
# # 2. 初始化句子表示模型
# embedder = SentenceEmbedder(model_name)
#
#     # 检查是否有可用的GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # 将模型移动到GPU
# embedder.model.to(device)
#
# # 3. 提取句子嵌入
# sentence_embeddings = embedder.get_sentence_embeddings(corpus)
#
# # 4. 计算初始对齐性和均匀性
# alignment = compute_alignment(sentence_embeddings)
# uniformity = compute_uniformity(sentence_embeddings)
# print(f"Initial Alignment: {alignment}, Initial Uniformity: {uniformity}")
#
# # 5. 进行对比学习训练
# trained_embeddings, training_metrics = train_contrastive_learning(
#     corpus, embedder, epochs, learning_rate
# )
#
# # 6. 训练后的对齐性和均匀性
# final_alignment = compute_alignment(trained_embeddings)
# final_uniformity = compute_uniformity(trained_embeddings)
# print(f"Final Alignment: {final_alignment}, Final Uniformity: {final_uniformity}")
#
# # 7. 可视化训练过程中的损失和对齐性、均匀性
# plot_metrics(training_metrics)
#
# """
# # 使用MRPC数据集
#
# # from datasets import load_dataset
# # from transformers import BertModel, BertTokenizer
# # from sentence_embedder import SentenceEmbedder
# # from training import train_contrastive_learning
# # from visualization import plot_metrics
# #
# # # 1. 加载 MRPC 数据集
# # dataset = load_dataset("glue", "mrpc", split="train")
# #
# # # 2. 加载预训练的 BERT 模型和分词器
# # model = BertModel.from_pretrained("bert-base-uncased")  # 修改为 BertModel
# # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# #
# #
# # # 3. 编写数据预处理函数，对句子进行标记化
# # def encode(examples):
# #     return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, padding="max_length")
# #
# #
# # dataset = dataset.map(encode, batched=True)
# #
# # # 4. 将 label 列重命名为 labels
# # dataset = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)
# #
# # # 5. 提取句子对以便送入嵌入器
# # sentences = [(example['sentence1'], example['sentence2']) for example in dataset]
# #
# # # 6. 实例化句子嵌入器
# # embedder = SentenceEmbedder(model, tokenizer)
# #
# # # 7. 开始对比学习训练
# # trained_embeddings, training_metrics = train_contrastive_learning(
# #     sentences=sentences,
# #     embedder=embedder,
# #     epochs=10,  # 可以根据需要调整
# #     learning_rate=1e-5
# # )
# #
# # # 8. 可视化训练结果
# # plot_metrics(training_metrics)
# """
import torch
from batch_data_loader import load_corpus_in_batches  # 批量加载的函数
from sentence_embedder import SentenceEmbedder
from training import train_contrastive_learning
from metrics import compute_alignment, compute_uniformity
from visualization import plot_metrics

# 设置模型路径和参数
data_path = "./data/sentence_pairs.csv"
model_name = "bert-large-uncased"
epochs = 10
learning_rate = 2e-4
batch_size = 32  # 批量大小

# 1. 加载数据（批量处理），指定batch_size
corpus_batches = load_corpus_in_batches(data_path, batch_size)

# 2. 初始化句子表示模型
embedder = SentenceEmbedder(model_name)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型移动到GPU
embedder.model.to(device)

# 3. 提取句子嵌入并计算初始对齐性和均匀性
initial_embeddings = []
for batch in corpus_batches:
    sentence_embeddings = embedder.get_sentence_embeddings(batch)
    initial_embeddings.extend(sentence_embeddings)  # 累积批量的嵌入

alignment = compute_alignment(initial_embeddings)
uniformity = compute_uniformity(initial_embeddings)
print(f"Initial Alignment: {alignment}, Initial Uniformity: {uniformity}")

# 4. 进行对比学习训练（批量处理）
trained_embeddings, training_metrics = train_contrastive_learning(
    corpus_batches, embedder, epochs, learning_rate
)

# 5. 训练后的对齐性和均匀性
final_alignment = compute_alignment(trained_embeddings)
final_uniformity = compute_uniformity(trained_embeddings)
print(f"Final Alignment: {final_alignment}, Final Uniformity: {final_uniformity}")

# 6. 可视化训练过程中的损失和对齐性、均匀性
plot_metrics(training_metrics)
