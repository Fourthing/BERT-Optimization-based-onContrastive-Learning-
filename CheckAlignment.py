import torch

def compute_alignment(embeddings1, embeddings2):
    """
    计算句子对的对齐性（Alignment）。

    参数:
    embeddings1: Tensor, 第一个句子集合的表示 (batch_size, embedding_dim)
    embeddings2: Tensor, 与第一个集合语义相似的第二个句子集合的表示 (batch_size, embedding_dim)

    返回:
    alignment_score: float, 对齐性得分
    """
    # 计算L2范数（欧氏距离）
    alignment_loss = torch.norm(embeddings1 - embeddings2, p=2, dim=1).mean()

    return alignment_loss.item()

# 实际使用
    # 假设 embeddings1 和 embeddings2 是通过 BERT 提取的两个相似句子的表示

    # embeddings1 = model.encode(similar_sentences1)
    # embeddings2 = model.encode(similar_sentences2)
    #
    # alignment_score = compute_alignment(torch.tensor(embeddings1), torch.tensor(embeddings2))
    # print("Alignment score:", alignment_score)
