from transformers import BertTokenizer, BertModel
import torch

class SentenceEmbedder:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_sentence_embeddings(self, sentences):
        # 获取一组句子的BERT嵌入表示
        embeddings = []
        self.model.eval()
        with torch.no_grad():
            for sentence1, sentence2 in sentences:
                inputs1 = self.tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
                inputs2 = self.tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)

                outputs1 = self.model(**inputs1)
                outputs2 = self.model(**inputs2)

                # 获取 [CLS] token 的嵌入作为句子的表示
                sentence_embedding1 = outputs1.last_hidden_state[:, 0, :]
                sentence_embedding2 = outputs2.last_hidden_state[:, 0, :]

                embeddings.append((sentence_embedding1, sentence_embedding2))

        return embeddings

"""

# class SentenceEmbedder:
#     def __init__(self, model, tokenizer):
#         self.tokenizer = tokenizer
#         self.model = model  # 使用传入的 BERT 模型
#
#     def get_sentence_embeddings(self, sentences):
#         # 获取一组句子的BERT嵌入表示
#         embeddings = []
#         self.model.eval()  # 进入评估模式
#
#         for sentence_pair in sentences:
#             # 确保 sentence_pair 是一个元组或列表
#             if isinstance(sentence_pair, (tuple, list)) and len(sentence_pair) == 2:
#                 sentence1, sentence2 = sentence_pair
#             else:
#                 print(f"Skipping invalid sentence pair: {sentence_pair}")
#                 continue  # 跳过无效的句子对
#
#             try:
#                 # 对句子进行分词
#                 inputs1 = self.tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True)
#                 inputs2 = self.tokenizer(sentence2, return_tensors="pt", padding=True, truncation=True)
#
#                 # 获取模型输出
#                 outputs1 = self.model(**inputs1)
#                 outputs2 = self.model(**inputs2)
#
#                 # 获取 [CLS] token 的嵌入作为句子的表示
#                 sentence_embedding1 = outputs1.last_hidden_state[:, 0, :]
#                 sentence_embedding2 = outputs2.last_hidden_state[:, 0, :]
#
#                 # 将句子对的嵌入添加到列表中
#                 embeddings.append((sentence_embedding1, sentence_embedding2))
#             except Exception as e:
#                 print(f"Error processing sentence pair: {sentence_pair}, Error: {e}")
#                 continue  # 如果处理失败，继续处理下一个句子对
#
#         return embeddings
"""

# from transformers import BertTokenizer, BertModel
# import torch
#
#
# class SentenceEmbedder:
#     def __init__(self, model_name):
#         # 初始化BERT模型和tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(model_name)
#         self.model = BertModel.from_pretrained(model_name)
#
#     def get_sentence_embeddings(self, batch):
#         """
#         处理批量句子对，获取BERT嵌入表示
#         batch 包含 input_ids, attention_mask, token_type_ids 和 label
#         """
#         # 从batch中获取input_ids, attention_mask和token_type_ids
#         input_ids, attention_mask, token_type_ids, _ = batch  # 忽略label
#
#         # 设置模型为评估模式，关闭dropout等
#         self.model.eval()
#
#         # 禁用梯度计算
#         with torch.no_grad():
#             # 获取模型的输出，包含所有层的hidden states
#             outputs = self.model(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 token_type_ids=token_type_ids
#             )
#
#         # 获取[CLS] token的嵌入（第0个位置）
#         sentence_embeddings = outputs.last_hidden_state[:, 0, :]
#
#         return sentence_embeddings
