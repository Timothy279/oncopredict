import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from config import Config

class DrugResponseModel(nn.Module):
    """
    Multi-Modal Deep Learning Architecture for Drug Response Prediction.
    
    This model fuses two distinct data modalities:
    1. Chemical Structure (Graph Data) processed via Graph Attention Networks (GAT).
    2. Genomic Profile (Vector Data) processed via a Dense MLP.
    
    The goal is to predict the IC50 (drug potency) value.
    """
    
    def __init__(self) -> None:
        super(DrugResponseModel, self).__init__()
        
        # Hyperparameters extraction from centralized Config
        node_features: int = Config.NUM_NODE_FEATURES
        genes: int = Config.NUM_GENES
        hidden: int = Config.HIDDEN_DIM
        heads: int = Config.HEADS
        dropout: float = Config.DROPOUT

        # --- Branch 1: Chemical Structure (GAT) ---
        # GATv2 allows the model to learn dynamic attention weights for each neighbor
        self.conv1 = GATConv(node_features, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, hidden, heads=heads, dropout=dropout)
        
        gnn_out_dim: int = hidden * heads

        # --- Branch 2: Genomics (MLP) ---
        # Encodes the gene expression profile into a latent representation
        self.geno_encoder = nn.Sequential(
            nn.Linear(genes, hidden * 2),
            nn.BatchNorm1d(hidden * 2), # Stabilize learning
            nn.ReLU(),
            nn.Dropout(dropout),        # Prevent overfitting
            nn.Linear(hidden * 2, gnn_out_dim),
            nn.ReLU()
        )

        # --- Fusion Layer ---
        # Concatenates both latent spaces and regresses to the final IC50 value
        self.fusion_layer = nn.Sequential(
            nn.Linear(gnn_out_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, x_gen: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Atom feature matrix [Num_Atoms, Num_Features].
            edge_index (torch.Tensor): Graph connectivity [2, Num_Edges].
            batch (torch.Tensor): Batch vector mapping atoms to molecules [Num_Atoms].
            x_gen (torch.Tensor): Gene expression vector [Batch_Size, Num_Genes].

        Returns:
            torch.Tensor: Predicted IC50 value [Batch_Size, 1].
        """
        # 1. Graph Neural Network Branch
        x = self.conv1(x, edge_index)
        x = F.elu(x) # ELU works well with GAT
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # Global Pooling: Aggregates atom-level embeddings into molecule-level embeddings
        x = global_mean_pool(x, batch)

        # 2. Genomic MLP Branch
        gen = self.geno_encoder(x_gen)

        # 3. Late Fusion & Regression
        combined = torch.cat((x, gen), dim=1)
        out = self.fusion_layer(combined)

        return out
