import torch
from transformers import BertTokenizer, BertForMaskedLM

# 加载预训练的BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 需要预测的句子，使用[MASK]标记需要预测的单词
text = "The man went to the [MASK]."

# 将文本编码为模型输入所需的格式
input_ids = tokenizer.encode(text, return_tensors='pt')

# 找到MASK的位置
mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

# 得到预测结果（logits）
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

# 找到MASK标记处的预测结果
mask_token_logits = logits[0, mask_token_index, :]

# 获取可能性最高的前5个单词
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

# 打印预测的单词
for token in top_5_tokens:
    predicted_token = tokenizer.decode([token])
    print(f"Predicted token: {predicted_token}")
