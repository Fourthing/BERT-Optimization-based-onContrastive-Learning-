# 基于对比学习的句子表示方法
## 2024年10月13日19:14:09

内容包括：
### 5个测试功能的.py文件
**预训练任务**：

- **遮蔽语言模型（Masked Language Model, MLM）**：BERT在预训练过程中，会随机遮蔽输入中的部分单词（如15%），并通过深度双向Transformer网络来预测这些被遮蔽的单词。例如，输入句子“the man went to the [MASK]”中，BERT需要预测出[MASK]为“store”。
- **下一句预测（Next Sentence Prediction, NSP）**：BERT会接收两句话A和B，判断B是否为A的下一句话，还是随机的一句话。通过这种方式来学习句子之间的关系。例如，句子A是“the man went to the store”，句子B是“he bought a gallon of milk”，BERT需要判断B是否是A的下一句。

**下游任务**（Fine-tuning）：

- **句子分类任务**：如情感分析、句子对比任务（例如MultiNLI），通过微调预训练的BERT模型来进行分类。
- **命名实体识别（NER）**：识别文本中的实体（如人名、地点、组织等）。
- **问答任务（如SQuAD）**：BERT被用于回答问题，给定一个问题和一段文本，模型需要从文本中找到问题的答案。

### 一个用于实验研究的项目：
BERT-ContrastiveLearning

### 几个关于项目思路的md文档：
怎么选择语料.md
手动实现BERT尝试.md
关于怎么量化BERT句子表示的对齐性和均匀性.md
