import torch
from transformers import BertTokenizer, BertForNextSentencePrediction

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

# 输入两个句子
sentence_a = "The man went to the store."
sentence_b = "He bought a gallon of milk."

# 对两个句子进行编码
encoding = tokenizer(sentence_a, sentence_b, return_tensors='pt')

# 使用模型预测
outputs = model(**encoding)
logits = outputs.logits

# 将logits转换为概率
softmax = torch.nn.Softmax(dim=1)
probs = softmax(logits)

# 判断是否为下一句
next_sentence_label = torch.argmax(probs).item()

# 输出结果
if next_sentence_label == 0:
    print(f"'{sentence_b}' 是 '{sentence_a}' 的下一句。")
else:
    print(f"'{sentence_b}' 不是 '{sentence_a}' 的下一句。")
