import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .layers import GCNLayer, AvgReadout, MaxReadout, MinReadout, WSReadout
from .losses import NormRegLoss, ScoreDALoss, EmbeddingDistillationLoss


class TeacherGGAD(nn.Module):
    """
    Teacher GNN Model:
    Generates high-capacity structural embeddings and adversarial perturbation-resilient
    anomaly representations across the graph.
    """
    def __init__(self, in_dim: int, hidden_dim: int, readout: str = "avg", noise_var: float = 0.01):
        super(TeacherGGAD, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, act=nn.PReLU())
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, act=nn.PReLU())
        self.gcn3 = GCNLayer(hidden_dim, hidden_dim, act=nn.PReLU())

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2, bias=False)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4, bias=False)
        self.fc3 = nn.Linear(hidden_dim // 4, 1, bias=False)
        self.fc_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.act = nn.ReLU()
        self.noise_var = noise_var

        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout(hidden_dim)
        else:
            self.read = AvgReadout()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, is_training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.gcn1(x, edge_index)
        emb = self.gcn2(h1, edge_index)

        # Inject controlled feature variance during training to simulate evasion attacks
        if is_training and self.noise_var > 0:
            noise = torch.randn_like(emb) * self.noise_var
            emb_perturbed = emb + noise
        else:
            emb_perturbed = emb

        # Score projection
        f1 = self.act(self.fc1(emb_perturbed))
        f2 = self.act(self.fc2(f1))
        scores = torch.sigmoid(self.fc3(f2)).squeeze(-1)

        return emb, scores


class StudentOCGNN(nn.Module):
    """
    Student GNN Model:
    Learns calibrated normality boundaries and distills representations from the Teacher.
    """
    def __init__(self, in_dim: int, hidden_dim: int, readout: str = "avg"):
        super(StudentOCGNN, self).__init__()
        self.gcn1 = GCNLayer(in_dim, hidden_dim, act=nn.PReLU())
        self.gcn2 = GCNLayer(hidden_dim, hidden_dim, act=nn.PReLU())

        if readout == "max":
            self.read = MaxReadout()
        elif readout == "min":
            self.read = MinReadout()
        elif readout == "weighted_sum":
            self.read = WSReadout(hidden_dim)
        else:
            self.read = AvgReadout()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = self.gcn1(x, edge_index)
        h2 = self.gcn2(h1, edge_index)
        return h1, h2


class GraphNC(nn.Module):
    """
    GraphNC (Normality Calibration in Graph Anomaly Detection - ICML 2026):
    Unifies Teacher GNN, Student GNN, NormReg loss, and ScoreDA distribution alignment.
    """
    def __init__(
        self,
        in_dim: int = 7,
        hidden_dim: int = 64,
        readout: str = "avg",
        beta: float = 0.5,
        score_da_weight: float = 1.0,
        distill_weight: float = 1.0,
        norm_reg_weight: float = 1.0,
        noise_var: float = 0.01,
    ):
        super(GraphNC, self).__init__()
        self.teacher = TeacherGGAD(in_dim, hidden_dim, readout=readout, noise_var=noise_var)
        self.student = StudentOCGNN(in_dim, hidden_dim, readout=readout)

        self.norm_reg_loss = NormRegLoss(embedding_dim=hidden_dim, beta=beta)
        self.score_da_loss = ScoreDALoss()
        self.distill_loss = EmbeddingDistillationLoss()

        self.score_da_weight = score_da_weight
        self.distill_weight = distill_weight
        self.norm_reg_weight = norm_reg_weight
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inference mode: returns calibrated anomaly probabilities combining Teacher and Student scores."""
        _, teacher_scores = self.teacher(x, edge_index, is_training=False)
        _, student_emb = self.student(x, edge_index)
        _, norm_scores = self.norm_reg_loss(student_emb, is_training=False)

        # Min-max normalize student normality scores
        min_s = torch.min(norm_scores)
        max_s = torch.max(norm_scores)
        norm_s_scaled = (norm_scores - min_s) / (max_s - min_s + 1e-8)

        # Calibrated ensemble anomaly score (GraphNC specification)
        calibrated_score = 0.5 * teacher_scores + 0.5 * norm_s_scaled
        return calibrated_score

    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete GraphNC multi-objective calibration loss:
        L_total = L_supervised + alpha * L_NormReg + beta * L_Distill + gamma * L_ScoreDA
        """
        # 1. Forward passes
        teacher_emb, teacher_scores = self.teacher(x, edge_index, is_training=is_training)
        _, student_emb = self.student(x, edge_index)

        # 2. Normality Regularization (NormReg) on Student embeddings
        norm_loss, student_scores = self.norm_reg_loss(student_emb, is_training=is_training)

        # 3. Embedding Distillation
        distill_loss = self.distill_loss(student_emb, teacher_emb)

        # 4. Score Distribution Alignment (ScoreDA)
        score_da = self.score_da_loss(student_scores, teacher_scores)

        # 5. Supervised Cross-Entropy Loss on labeled nodes (weighted for extreme imbalance)
        masked_y = y[mask].float()
        masked_pred = teacher_scores[mask]
        
        # Calculate positive class weight to balance 0.23% positive conversions
        pos_weight = (1.0 - masked_y.mean()) / (masked_y.mean() + 1e-6)
        weight_vec = torch.where(masked_y == 1, pos_weight, 1.0)
        bce_loss = F.binary_cross_entropy(masked_pred, masked_y, weight=weight_vec)

        # 6. Total Combined Loss
        total_loss = (
            bce_loss
            + self.norm_reg_weight * norm_loss
            + self.distill_weight * distill_loss
            + self.score_da_weight * score_da
        )

        loss_metrics = {
            "loss_total": total_loss.item(),
            "loss_bce": bce_loss.item(),
            "loss_norm_reg": norm_loss.item(),
            "loss_distill": distill_loss.item(),
            "loss_score_da": score_da.item(),
        }

        return total_loss, loss_metrics
