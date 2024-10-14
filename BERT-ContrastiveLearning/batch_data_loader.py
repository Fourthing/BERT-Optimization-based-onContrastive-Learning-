import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer

# 初始化tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_corpus_in_batches(filepath, batch_size=32):
    """加载句子对数据并返回一个DataLoader，指定批量大小"""

    # 加载CSV文件中的数据
    data = pd.read_csv(filepath)

    # 假设CSV文件包含三列：sentence1, sentence2, label
    sentences = list(zip(data['sentence1'], data['sentence2']))
    labels = data['label'].tolist()

    # 将句子对转化为BERT模型的输入格式
    tokenized_data = tokenizer(
        [s1 for s1, s2 in sentences],
        [s2 for s1, s2 in sentences],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    # 创建 TensorDataset，将句子对的 input_ids, attention_mask 和 标签保存
    dataset = TensorDataset(
        tokenized_data["input_ids"],
        tokenized_data["attention_mask"],
        tokenized_data["token_type_ids"],
        torch.tensor(labels)
    )

    # 返回 DataLoader，指定 batch_size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader
