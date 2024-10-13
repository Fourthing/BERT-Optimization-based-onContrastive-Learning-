import torch
from transformers import BertTokenizer, BertForTokenClassification

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained('bert-base-cased', num_labels=9)  # 假设我们有9个NER标签

# NER 标签（BIO标记）
label_list = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]

# 输入句子
sentence = "Hawking was a theoretical physicist at Cambridge University."

# 对输入句子进行编码
inputs = tokenizer(sentence, return_tensors="pt", is_split_into_words=False)

# 模型推理
outputs = model(**inputs)
logits = outputs.logits

# 获取每个token的预测标签
predictions = torch.argmax(logits, dim=2)

# 将预测的标签转换为NER标签
predicted_labels = [label_list[pred.item()] for pred in predictions[0]]

# 对应输入的token
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# 输出每个token及其对应的NER标签
for token, label in zip(tokens, predicted_labels):
    print(f"{token}: {label}")
