"""
ContextSurg: Context-Aware Surgical Risk Prediction Model

This module implements the ContextSurg architecture for context-aware healthcare risk prediction.
The model uses hospital-aware routing to dynamically assign patients to specialized expert networks
while maintaining equitable performance across different hospital settings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for feature recalibration.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        reduction_ratio: Reduction ratio for the squeeze operation (default: 16)
    """
    def __init__(self, input_dim: int, output_dim: int, reduction_ratio: int = 16):
        super(SEBlock, self).__init__()

        hidden_dim = max(64, input_dim // reduction_ratio)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply squeeze-and-excitation to input features."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class HospitalRouter(nn.Module):
    """
    Router network that uses hospital features to determine expert assignment.

    Args:
        hospital_feature_dim: Dimension of hospital features
        hidden_dims: List of hidden layer dimensions
        hospital_embedding_dim: Dimension of hospital embeddings
        num_experts: Number of expert networks
    """
    def __init__(
        self,
        hospital_feature_dim: int,
        hidden_dims: List[int] = [128],
        hospital_embedding_dim: int = 64,
        num_experts: int = 3
    ):
        super(HospitalRouter, self).__init__()

        # Build the router network
        layers = []
        prev_dim = hospital_feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        self.he_block = SEBlock(hospital_feature_dim, hospital_feature_dim)
        self.W_dist = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(hospital_feature_dim, hospital_feature_dim).type(torch.FloatTensor),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )
        self.a_dist = nn.Parameter(
            nn.init.xavier_normal_(
                torch.Tensor(2 * hospital_feature_dim, 1).type(torch.FloatTensor),
                gain=np.sqrt(2.0)
            ),
            requires_grad=True
        )
        self.dropout = nn.Dropout(0.5)

        layers.append(nn.Linear(prev_dim, hospital_embedding_dim))
        self.router_network = nn.Sequential(*layers)
        self.dist = nn.Linear(hospital_embedding_dim, num_experts)

    def forward(
        self,
        hospital_features: torch.Tensor,
        all_hospital_features: Optional[torch.Tensor] = None
    ) -> tuple:
        """
        Forward pass through the router network.

        Args:
            hospital_features: Tensor of shape [batch_size, hospital_feature_dim]
            all_hospital_features: Tensor of shape [num_global_hospitals, hospital_feature_dim]

        Returns:
            Tuple containing:
                - final_embd: Final hospital embeddings
                - h_embd: Hospital embeddings before attention
                - expert_weights: Softmax weights for expert routing
                - hospital_se: SE block outputs
        """
        hospital_se = self.he_block(hospital_features)
        global_se = self.he_block(all_hospital_features)

        embd = self.router_network(hospital_features * hospital_se)
        h_embd = torch.mm(hospital_features, self.W_dist)

        N = hospital_features.shape[0]
        M = all_hospital_features.shape[0]

        global_embd = self.router_network(all_hospital_features * global_se)

        # Compute attention between batch and global hospitals
        h_embd_batch_expanded = hospital_features.unsqueeze(1).expand(N, M, -1)
        h_embd_global_expanded = all_hospital_features.unsqueeze(0).expand(N, M, -1)

        h_embd_cat = torch.cat([h_embd_batch_expanded, h_embd_global_expanded], dim=2)

        h_dist = torch.sigmoid(h_embd_cat @ self.a_dist).squeeze(2)
        attention_weights = F.softmax(h_dist, dim=1)

        context_embd = torch.matmul(attention_weights, global_embd)

        final_embd = embd + context_embd

        expert_weights = self.dist(final_embd)
        expert_weights = F.softmax(expert_weights, dim=1)

        return final_embd, h_embd, expert_weights, hospital_se


class SharedEmbedding(nn.Module):
    """
    Shared embedding layer for patient features.

    Args:
        patient_feature_dim: Dimension of patient features
        hidden_dims: List of hidden layer dimensions
        embedding_dim: Dimension of output embeddings
    """
    def __init__(
        self,
        patient_feature_dim: int,
        hidden_dims: List[int] = [128],
        embedding_dim: int = 64
    ):
        super(SharedEmbedding, self).__init__()

        layers = []
        prev_dim = patient_feature_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.embedding = nn.Sequential(*layers)

    def forward(self, patient_features: torch.Tensor) -> torch.Tensor:
        """Generate embeddings from patient features."""
        return self.embedding(patient_features)


class Expert(nn.Module):
    """
    Individual expert network that processes patient data.

    Args:
        embedding_dim: Dimension of input embeddings
        hidden_dims: List of hidden layer dimensions
        expert_dim: Dimension of expert outputs
    """
    def __init__(
        self,
        embedding_dim: int,
        hidden_dims: List[int] = [128],
        expert_dim: int = 64
    ):
        super(Expert, self).__init__()

        layers = []
        prev_dim = embedding_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, expert_dim))
        self.expert_network = nn.Sequential(*layers)

    def forward(self, patient_features: torch.Tensor) -> torch.Tensor:
        """Process patient embeddings through expert network."""
        return self.expert_network(patient_features)


class FinalPredictor(nn.Module):
    """
    Final predictor network that generates risk predictions.

    Args:
        expert_dim: Dimension of expert outputs
        hidden_dim: Hidden layer dimension (default: 32)
    """
    def __init__(self, expert_dim: int, hidden_dim: int = 32):
        super(FinalPredictor, self).__init__()

        self.final_predictor = nn.Sequential(
            nn.Linear(expert_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, expert_embeddings: torch.Tensor) -> torch.Tensor:
        """Generate final risk prediction from expert embeddings."""
        return self.final_predictor(expert_embeddings)


class FairRiskPredictor(nn.Module):
    """
    Network that predicts patient risk based only on patient features.
    Used for fairness constraints.

    Args:
        embedding_dim: Dimension of input embeddings
    """
    def __init__(self, embedding_dim: int):
        super(FairRiskPredictor, self).__init__()

        self.fair_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, patient_embedding: torch.Tensor) -> torch.Tensor:
        """Generate fair risk prediction from patient embeddings."""
        return self.fair_predictor(patient_embedding)


class AdversarialDiscriminator(nn.Module):
    """
    Adversarial network that tries to predict location from embeddings.
    Used to ensure fairness by preventing location information leakage.

    Args:
        embedding_dim: Dimension of input embeddings
        num_locations: Number of location classes
    """
    def __init__(self, embedding_dim: int, num_locations: int = 3):
        super(AdversarialDiscriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Linear(embedding_dim, num_locations),
            nn.Softmax(dim=1)
        )

    def forward(self, fair_embedding: torch.Tensor) -> torch.Tensor:
        """Predict location from fair embeddings."""
        return self.discriminator(fair_embedding)


class ContextSurg(nn.Module):
    """
    ContextSurg: Context-Aware Surgical Risk Prediction Model.

    This model combines hospital-aware routing with mixture-of-experts architecture to provide
    accurate and equitable risk predictions across diverse hospital settings.

    Args:
        patient_feature_dim: Dimension of patient features
        hospital_feature_dim: Dimension of hospital features
        num_experts: Number of expert networks (default: 5)
        num_locations: Number of location classes for fairness (default: 3)
        embedding_dim: Dimension of patient embeddings (default: 64)
        hospital_embedding_dim: Dimension of hospital embeddings (default: 128)
        expert_dim: Dimension of expert outputs (default: 128)
        shared_hidden_dims: Hidden dimensions for shared embedding (default: [128, 64])
        router_hidden_dims: Hidden dimensions for router (default: [64])
        expert_hidden_dims: Hidden dimensions for experts (default: [256, 128])
        lambda_fair: Weight for fairness loss (default: 0.1)
        lambda_adv: Weight for adversarial loss (default: 0.1)
        lambda_pred: Weight for prediction loss (default: 1.0)
        all_train_hospital_features: Global hospital features for routing
    """
    def __init__(
        self,
        patient_feature_dim: int,
        hospital_feature_dim: int,
        num_experts: int = 5,
        num_locations: int = 3,
        embedding_dim: int = 64,
        hospital_embedding_dim: int = 128,
        expert_dim: int = 128,
        shared_hidden_dims: List[int] = [128, 64],
        router_hidden_dims: List[int] = [64],
        expert_hidden_dims: List[int] = [256, 128],
        lambda_fair: float = 0.1,
        lambda_adv: float = 0.1,
        lambda_pred: float = 1.0,
        all_train_hospital_features: Optional[torch.Tensor] = None
    ):
        super(ContextSurg, self).__init__()

        self.num_experts = num_experts
        self.lambda_fair = lambda_fair
        self.lambda_adv = lambda_adv
        self.lambda_pred = lambda_pred

        # Store all hospital features from the training set
        if all_train_hospital_features is not None:
            self.register_buffer('all_hospital_features', all_train_hospital_features)
        else:
            self.all_hospital_features = None

        # Model components
        self.shared_embedding = SharedEmbedding(patient_feature_dim, shared_hidden_dims, embedding_dim)
        self.fair_embedding = SharedEmbedding(patient_feature_dim, shared_hidden_dims, embedding_dim)
        self.router = HospitalRouter(hospital_feature_dim, router_hidden_dims, hospital_embedding_dim, num_experts)
        self.fair_predictor = FairRiskPredictor(embedding_dim)
        self.final_predictor = FinalPredictor(expert_dim)
        self.experts = nn.ModuleList([
            Expert(embedding_dim, expert_hidden_dims, expert_dim) for _ in range(num_experts)
        ])
        self.se_block = SEBlock(hospital_embedding_dim, patient_feature_dim)
        self.patient_se_block = SEBlock(patient_feature_dim, patient_feature_dim)
        self.discriminator = AdversarialDiscriminator(embedding_dim, num_locations)

    def forward(
        self,
        patient_features: torch.Tensor,
        hospital_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            patient_features: Tensor of shape [batch_size, patient_feature_dim]
            hospital_features: Tensor of shape [batch_size, hospital_feature_dim]

        Returns:
            Dictionary with model outputs including:
                - final_prediction: Final risk predictions
                - fair_risk: Fair risk predictions
                - expert_weights: Expert routing weights
                - adv_prediction: Adversarial predictions
                - hospital_embd: Hospital embeddings
                - feature_se: Feature importance weights
                - hospital_se: Hospital importance weights
                - expert_predictions: Individual expert predictions
        """
        # Get expert weights from router
        hospital_embd, h_embd, expert_weights, hospital_se = self.router(
            hospital_features, self.all_hospital_features
        )

        # Apply feature importance weighting
        feature_se = self.se_block(hospital_embd)
        patient_se = self.patient_se_block(patient_features)
        se_features = patient_features * feature_se

        # Get patient embeddings
        patient_embedding = self.shared_embedding(se_features)
        fair_embedding = self.shared_embedding(se_features)

        # Get fair risk prediction
        fair_risk = self.final_predictor(fair_embedding)

        # Get predictions from each expert
        expert_predictions = []
        for expert in self.experts:
            expert_pred = expert(patient_embedding)
            expert_predictions.append(expert_pred)

        # Stack and weight expert predictions
        expert_predictions = torch.stack(expert_predictions, dim=1)
        weighted_expert_pred = torch.sum(
            expert_predictions * expert_weights.unsqueeze(-1),
            dim=1
        )

        # Generate final prediction
        final_prediction = self.final_predictor(weighted_expert_pred)

        # Get adversarial prediction
        adv_prediction = self.discriminator(fair_embedding)

        outputs = {
            "final_prediction": final_prediction,
            "fair_risk": fair_risk,
            "expert_weights": expert_weights,
            "adv_prediction": adv_prediction,
            "hospital_embd": h_embd,
            "feature_se": feature_se,
            "hospital_se": hospital_se,
            "expert_predictions": expert_predictions
        }

        return outputs

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        location: torch.Tensor,
        lambda_pred: Optional[float] = None,
        lambda_fair: Optional[float] = None,
        lambda_adv: Optional[float] = None,
        diversity_weight: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the loss for training.

        Args:
            outputs: Dictionary with model outputs
            targets: Tensor of shape [batch_size, 1] with true labels
            location: Tensor of shape [batch_size] with location indices
            lambda_pred: Override for prediction loss weight
            lambda_fair: Override for fairness loss weight
            lambda_adv: Override for adversarial loss weight
            diversity_weight: Weight for diversity loss (default: 0.01)

        Returns:
            Dictionary with loss components
        """
        lambda_fair = lambda_fair if lambda_fair is not None else self.lambda_fair
        lambda_adv = lambda_adv if lambda_adv is not None else self.lambda_adv
        lambda_pred = lambda_pred if lambda_pred is not None else self.lambda_pred

        # Main prediction loss
        pred_loss = F.binary_cross_entropy(
            outputs["final_prediction"],
            targets
        )

        # Fair risk loss
        fair_loss = F.binary_cross_entropy(
            outputs["fair_risk"],
            targets
        )

        # Adversarial loss
        adv_loss = F.cross_entropy(
            outputs["adv_prediction"],
            location
        )

        # Expert diversity loss (encourage specialization)
        expert_weights = outputs["expert_weights"]
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-10), dim=1)
        diversity_loss = torch.mean(entropy)

        # Total loss
        total_loss = (
            lambda_pred * pred_loss +
            lambda_fair * fair_loss -
            diversity_weight * diversity_loss
        )

        return {
            "total_loss": total_loss,
            "pred_loss": pred_loss,
            "fair_loss": fair_loss,
            "adv_loss": adv_loss,
            "diversity_loss": diversity_loss
        }

    def get_expert_assignments(self, hospital_features: torch.Tensor) -> Dict[int, List[int]]:
        """
        Get the expert assignments for a batch of hospital features.

        Args:
            hospital_features: Tensor of shape [batch_size, hospital_feature_dim]

        Returns:
            Dictionary mapping expert indices to lists of sample indices
        """
        with torch.no_grad():
            _, _, expert_weights, _ = self.router(hospital_features, self.all_hospital_features)
            expert_assignments = torch.argmax(expert_weights, dim=1)

            # Group samples by expert
            assignments = {}
            for i in range(self.num_experts):
                assignments[i] = (expert_assignments == i).nonzero().squeeze().tolist()
                if not isinstance(assignments[i], list):
                    assignments[i] = [assignments[i]]

            return assignments
