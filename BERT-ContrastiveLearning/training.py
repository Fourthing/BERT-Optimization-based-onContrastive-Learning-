# import torch
# import torch.optim as optim
# import torch.nn as nn
#
# from metrics import compute_uniformity
# from metrics import compute_alignment
#
#
# def contrastive_loss(embedding1, embedding2, temperature=0.5):
#     """计算对比学习中的损失函数，使用Cosine Similarity"""
#     cos = nn.CosineSimilarity(dim=-1)
#     similarity = cos(embedding1, embedding2)
#     loss = -torch.log(similarity / temperature)
#     return loss.mean()
#
#
# def train_contrastive_learning(sentences, embedder, epochs, learning_rate):
#     """对比学习的训练函数，返回训练后的句子嵌入和训练过程中的损失、对齐性、均匀性"""
#     optimizer = optim.Adam(embedder.model.parameters(), lr=learning_rate)
#     training_metrics = {"loss": [], "alignment": [], "uniformity": []}
#
#     for epoch in range(epochs):
#         epoch_loss = 0
#
#         embeddings = embedder.get_sentence_embeddings(sentences)
#
#         # 确保 embeddings 是一个张量列表
#         embeddings = [torch.tensor(embedding).requires_grad_() for embedding in embeddings]
#
#         # 然后再使用 torch.stack
#         embeddings = torch.stack(embeddings)
#
#         for embedding1, embedding2 in embeddings:
#             optimizer.zero_grad()
#             loss = contrastive_loss(embedding1, embedding2)
#             loss.backward()
#             optimizer.step()
#
#             epoch_loss += loss.item()
#
#         avg_loss = epoch_loss / len(embeddings)
#         alignment = compute_alignment(embeddings)
#         uniformity = compute_uniformity(embeddings)
#
#         training_metrics["loss"].append(avg_loss)
#         training_metrics["alignment"].append(alignment)
#         training_metrics["uniformity"].append(uniformity)
#
#         print(
#             f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Alignment: {alignment:.4f} | Uniformity: {uniformity:.4f}")
#
#     final_embeddings = embeddings
#     return final_embeddings, training_metrics

import torch
import torch.optim as optim
import torch.nn as nn

from metrics import compute_uniformity
from metrics import compute_alignment


def contrastive_loss(embedding1, embedding2, temperature=0.5):
    """计算对比学习中的损失函数，使用Cosine Similarity"""
    cos = nn.CosineSimilarity(dim=-1)
    similarity = cos(embedding1, embedding2)
    loss = -torch.log(similarity / temperature)
    return loss.mean()

def train_contrastive_learning(sentences, embedder, epochs, learning_rate):
    """对比学习的训练函数，返回训练后的句子嵌入和训练过程中的损失、对齐性、均匀性"""
    optimizer = optim.Adam(embedder.model.parameters(), lr=learning_rate)
    training_metrics = {"loss": [], "alignment": [], "uniformity": []}

    for epoch in range(epochs):
        epoch_loss = 0
        embeddings = embedder.get_sentence_embeddings(sentences)

        # 假设 embeddings 是一个包含元组的列表
        for embedding1, embedding2 in embeddings:
            optimizer.zero_grad()

            # 确保 embedding1 和 embedding2 是合适的格式
            embedding1_tensor = torch.tensor(embedding1).float().requires_grad_()
            embedding2_tensor = torch.tensor(embedding2).float().requires_grad_()

            loss = contrastive_loss(embedding1_tensor, embedding2_tensor)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(embeddings)
        alignment = compute_alignment(embeddings)
        uniformity = compute_uniformity(embeddings)

        training_metrics["loss"].append(avg_loss)
        training_metrics["alignment"].append(alignment)
        training_metrics["uniformity"].append(uniformity)

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Alignment: {alignment:.4f} | Uniformity: {uniformity:.4f}")

    final_embeddings = embeddings
    return final_embeddings, training_metrics
