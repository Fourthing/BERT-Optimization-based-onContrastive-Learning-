"""
import torch
import torch.nn.functional as F


def compute_alignment(embeddings):
    # 计算对齐性 (Alignment)，衡量正例对的句子表示相似性
    total_similarity = 0
    for embedding1, embedding2 in embeddings:
        total_similarity += F.cosine_similarity(embedding1, embedding2).mean().item()

    return total_similarity / len(embeddings)


def compute_uniformity(embeddings, t=2):
    # 计算均匀性 (Uniformity)，衡量所有嵌入分布的均匀程度
    embeddings_concat = torch.cat([e1 for e1, _ in embeddings] + [e2 for _, e2 in embeddings], dim=0)
    distance_matrix = torch.cdist(embeddings_concat, embeddings_concat)
    uniformity_score = torch.exp(-t * distance_matrix).mean().item()

    return uniformity_score
"""
# import torch
# import torch.nn.functional as F
#
# def compute_alignment(embeddings):
#     """计算对齐性 (Alignment)，衡量正例对的句子表示相似性"""
#     total_similarity = 0
#     num_pairs = embeddings.size(0) // 2  # 假设每对句子嵌入有两个张量
#
#     # 遍历所有的句子对
#     for i in range(num_pairs):
#         embedding1 = embeddings[i]
#         embedding2 = embeddings[i + num_pairs]  # 获取相应的第二个句子嵌入
#         total_similarity += F.cosine_similarity(embedding1, embedding2).mean().item()
#
#     return total_similarity / num_pairs
#
# def compute_uniformity(embeddings, t=2):
#     """计算均匀性 (Uniformity)，衡量所有嵌入分布的均匀程度"""
#     # 直接对所有句子的嵌入进行均匀性计算
#     distance_matrix = torch.cdist(embeddings, embeddings)
#     uniformity_score = torch.exp(-t * distance_matrix).mean().item()
#
#     return uniformity_score
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F


def compute_alignment(embeddings):
    # 计算对齐性 (Alignment)，衡量正例对的句子表示相似性
    total_similarity = 0

    # 假设 embeddings 是按句子对的顺序排列
    for i in range(0, len(embeddings), 2):
        embedding1 = embeddings[i]  # 句子对的第一个句子嵌入
        embedding2 = embeddings[i + 1]  # 句子对的第二个句子嵌入# 将 numpy 数组转换为 torch.Tensor
        embedding1_tensor = torch.from_numpy(embedding1)
        embedding2_tensor = torch.from_numpy(embedding2)

        # 计算余弦相似度
        total_similarity += F.cosine_similarity(embedding1_tensor, embedding2_tensor).mean().item()

    # 返回平均对齐性
    return total_similarity / (len(embeddings) / 2)




def compute_uniformity(embeddings, t=2):
    # 计算均匀性 (Uniformity)，衡量所有嵌入分布的均匀程度

    # 首先，我们将 embeddings 中的所有句子嵌入提取出来
    all_embeddings = []
    for i in range(0, len(embeddings), 2):
        embedding1 = embeddings[i]  # 句子对的第一个句子嵌入
        embedding2 = embeddings[i + 1]  # 句子对的第二个句子嵌入

        # 确保 embedding 是 torch.Tensor 类型，如果是 numpy.ndarray 则转换
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1)
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2)

        all_embeddings.append(embedding1)
        all_embeddings.append(embedding2)

    # 将所有嵌入堆叠成一个大张量，用于计算距离
    embeddings_concat = torch.cat(all_embeddings, dim=0)

    # 计算嵌入之间的欧氏距离矩阵
    distance_matrix = torch.cdist(embeddings_concat, embeddings_concat)

    # 计算均匀性得分，使用负指数函数来计算分布的均匀程度
    uniformity_score = torch.exp(-t * distance_matrix).mean().item()

    return uniformity_score


