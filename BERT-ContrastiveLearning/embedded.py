import torch


class SentenceEmbedder:
    def __init__(self, model, tokenizer):
        """初始化 BERT 模型和分词器"""
        self.model = model
        self.tokenizer = tokenizer

    def get_sentence_embeddings(self, sentence):
        """从 BERT 模型中提取句子嵌入"""
        # 将句子转换为 BERT 的输入格式
        inputs = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

        # 获取模型的最后一层隐藏状态
        with torch.no_grad():  # 关闭梯度计算以节省内存
            outputs = self.model(**inputs)

        # 获取每个句子的平均嵌入
        sentence_embedding = outputs.last_hidden_state.mean(dim=1)
        return sentence_embedding
