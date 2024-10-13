import torch
import torch.nn.functional as F

from CheckAlignment import compute_alignment
from CheckUniformity import compute_uniformity


def contrastive_loss(embeddings1, embeddings2, temperature=0.5):
    """
    实现对比学习中的损失函数。

    参数:
    embeddings1: Tensor, 第一个句子集合的表示 (batch_size, embedding_dim)
    embeddings2: Tensor, 第二个句子集合的表示 (batch_size, embedding_dim)
    temperature: float, 对比学习中的温度参数

    返回:
    loss: float, 对比学习损失
    """
    batch_size = embeddings1.size(0)

    # 归一化句子向量
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(embeddings1, embeddings2.T) / temperature

    # 计算对比学习损失
    labels = torch.arange(batch_size).to(embeddings1.device)
    loss = F.cross_entropy(similarity_matrix, labels)

    return loss


def combined_loss(embeddings1, embeddings2, temperature=0.5):
    # 对比学习损失
    contrastive_loss_value = contrastive_loss(embeddings1, embeddings2, temperature)

    # 均匀性损失
    uniformity_loss_value = compute_uniformity(torch.cat([embeddings1, embeddings2], dim=0))

    # 对齐性损失
    alignment_loss_value = compute_alignment(embeddings1, embeddings2)

    # 综合损失，均匀性和对齐性可以通过超参数控制其影响权重
    total_loss = contrastive_loss_value + 0.1 * uniformity_loss_value + 0.1 * alignment_loss_value

    return total_loss
