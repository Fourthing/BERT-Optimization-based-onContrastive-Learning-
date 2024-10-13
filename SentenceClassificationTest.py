import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # 二分类任务

# 输入句子
sentence = "I hate this movie. It was boring!"

# 对输入句子进行编码
inputs = tokenizer(sentence, return_tensors="pt")

# 使用模型进行预测
outputs = model(**inputs)
logits = outputs.logits

# 将logits转换为概率
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)

# 得到分类标签
predicted_class = torch.argmax(probs).item()

# 输出结果
if predicted_class == 0:
    print(f"'{sentence}' 表达的是积极的情感。")
else:
    print(f"'{sentence}' 表达的是消极的情感。")
