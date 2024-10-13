from transformers import BertTokenizer, BertModel
import torch

class SentenceEmbedder:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def get_sentence_embeddings(self, sentences):
        """获取一组句子的BERT嵌入表示"""
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
