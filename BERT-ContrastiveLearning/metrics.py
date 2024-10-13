import torch
import torch.nn.functional as F


def compute_alignment(embeddings):
    """计算对齐性 (Alignment)，衡量正例对的句子表示相似性"""
    total_similarity = 0
    for embedding1, embedding2 in embeddings:
        total_similarity += F.cosine_similarity(embedding1, embedding2).mean().item()

    return total_similarity / len(embeddings)


def compute_uniformity(embeddings, t=2):
    """计算均匀性 (Uniformity)，衡量所有嵌入分布的均匀程度"""
    embeddings_concat = torch.cat([e1 for e1, _ in embeddings] + [e2 for _, e2 in embeddings], dim=0)
    distance_matrix = torch.cdist(embeddings_concat, embeddings_concat)
    uniformity_score = torch.exp(-t * distance_matrix).mean().item()

    return uniformity_score
