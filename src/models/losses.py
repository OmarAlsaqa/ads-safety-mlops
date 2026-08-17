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
    using stable MSE and Cosine Ranking Alignment to stabilize extreme class imbalance training.
    """
    def __init__(self, eps: float = 1e-6, mse_weight: float = 1.0):
        super(ScoreDALoss, self).__init__()
        self.eps = eps
        self.mse_weight = mse_weight

    def forward(self, student_score: torch.Tensor, teacher_score: torch.Tensor) -> torch.Tensor:
        # 1. Map unbounded hypersphere distance scores to probabilities in (0, 1)
        s_prob = torch.sigmoid(student_score).clamp(min=1e-5, max=1.0 - 1e-5)
        t_prob = teacher_score.clamp(min=1e-5, max=1.0 - 1e-5)

        # 2. Score MSE alignment
        mse_loss = F.mse_loss(s_prob, t_prob, reduction="mean")

        # 3. Cosine Ranking Alignment (100% immune to log(0) NaN singularities)
        cos_sim = F.cosine_similarity(s_prob.unsqueeze(0), t_prob.unsqueeze(0), dim=-1)
        ranking_loss = torch.mean(1.0 - cos_sim)

        return self.mse_weight * mse_loss + 0.1 * ranking_loss


class EmbeddingDistillationLoss(nn.Module):
    """Computes Mean Squared Error distillation between Teacher and Student embeddings."""
    def __init__(self):
        super(EmbeddingDistillationLoss, self).__init__()

    def forward(self, emb_s: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(emb_s, emb_t, reduction="mean")


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) for handling extreme class imbalance.
    Down-weights well-classified examples and focuses learning on hard negatives.
    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw un-sigmoided predictions (batch_size,)
            targets: Binary labels 0/1 (batch_size,)
        """
        # Numerically stable BCE per-element (no reduction)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Probability of correct class
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        p_t = p_t.clamp(min=1e-6, max=1.0 - 1e-6)

        # Focal modulation factor: (1 - p_t)^gamma
        focal_weight = (1.0 - p_t) ** self.gamma

        # Alpha balancing: alpha for positives, (1-alpha) for negatives
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        loss = alpha_t * focal_weight * bce
        return loss.mean()
