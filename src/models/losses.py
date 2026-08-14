import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NormRegLoss(nn.Module):
    """
    Normality Regularization (NormReg) Loss:
    Calibrates normal click interactions into a compact hypersphere of radius r centered at c,
    while pushing anomaly / botnet nodes outside the boundary.
    """
    def __init__(self, embedding_dim: int, beta: float = 0.5, eps: float = 1e-3):
        super(NormRegLoss, self).__init__()
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.eps = eps
        self.register_buffer("c", torch.zeros(embedding_dim))
        self.register_buffer("r", torch.tensor(0.0))
        self.warmup_steps = 5

    def forward(self, emb: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        device = emb.device
        c = self.c.to(device)
        r = self.r.to(device)

        # Squared Euclidean distance to normality center
        dist = torch.sum(torch.pow(emb - c, 2), dim=-1)
        score = dist - (r ** 2)
        loss = (r ** 2) + (1.0 / self.beta) * torch.mean(torch.relu(score))

        # Dynamic center and radius update during initial warmup epochs
        if is_training and self.warmup_steps > 0:
            with torch.no_grad():
                self.warmup_steps -= 1
                new_r = torch.quantile(torch.sqrt(torch.clamp(dist, min=0.0)), 1.0 - self.beta)
                new_c = torch.mean(emb, dim=0)
                new_c[(torch.abs(new_c) < self.eps) & (new_c < 0)] = -self.eps
                new_c[(torch.abs(new_c) < self.eps) & (new_c > 0)] = self.eps
                self.c.copy_(new_c)
                self.r.copy_(new_r)

        return loss, score


class ScoreDALoss(nn.Module):
    """
    Score Distribution Alignment (ScoreDA) Loss:
    Distills and aligns the anomaly score distribution between the Teacher GNN and Student GNN
    using KL-Divergence and MSE to stabilize extreme class imbalance training.
    """
    def __init__(self, eps: float = 1e-8, kl_weight: float = 0.5, mse_weight: float = 0.5):
        super(ScoreDALoss, self).__init__()
        self.eps = eps
        self.kl_weight = kl_weight
        self.mse_weight = mse_weight

    def forward(self, student_score: torch.Tensor, teacher_score: torch.Tensor) -> torch.Tensor:
        # 1. Normalize scores to valid probability distributions for KL divergence
        s_prob = torch.clamp(student_score, min=self.eps, max=1.0)
        t_prob = torch.clamp(teacher_score, min=self.eps, max=1.0)
        s_prob = s_prob / (s_prob.sum() + self.eps)
        t_prob = t_prob / (t_prob.sum() + self.eps)

        kl_loss = F.kl_div(s_prob.log(), t_prob, reduction="batchmean")

        # 2. Score MSE alignment
        mse_loss = F.mse_loss(student_score, teacher_score, reduction="mean")

        return self.kl_weight * kl_loss + self.mse_weight * mse_loss


class EmbeddingDistillationLoss(nn.Module):
    """Computes Mean Squared Error distillation between Teacher and Student embeddings."""
    def __init__(self):
        super(EmbeddingDistillationLoss, self).__init__()

    def forward(self, emb_s: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(emb_s, emb_t, reduction="mean")
