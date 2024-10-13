import torch
import torch.nn.functional as F


def compute_uniformity(embeddings):
    """
    计算句子表示空间的均匀性（Uniformity）。

    参数:
    embeddings: Tensor, 句子表示的向量 (batch_size, embedding_dim)

    返回:
    uniformity_score: float, 均匀性得分
    """
    batch_size = embeddings.size(0)

    # 计算余弦相似度 (Cosine Similarity)
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)  # 归一化到单位球面上
    similarity_matrix = torch.matmul(embeddings_normalized, embeddings_normalized.T)

    # 获取非对角线元素（排除与自己的相似度）
    mask = torch.eye(batch_size).to(embeddings.device)  # 构建对角线掩码
    similarity_matrix = similarity_matrix * (1 - mask)  # 去掉对角线元素

    # 计算均匀性损失
    distance_matrix = torch.pow(similarity_matrix, 2)  # 二次方计算距离
    uniformity_loss = torch.log(torch.exp(-2 * distance_matrix).mean())

    return uniformity_loss.item()

# 实际使用
# 假设 embeddings 是通过 BERT 或 Sentence-BERT 提取的句子表示
# embeddings = model.encode(sentences)
# uniformity_score = compute_uniformity(torch.tensor(embeddings))
# print("Uniformity score:", uniformity_score)
