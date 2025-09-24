import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class DisentangledEncoder(nn.Module):
    """Encodes a single modality input into a disentangled latent vector."""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, latent_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        return self.fc2(x)

class InterpretableFusion(nn.Module):
    """Interpretable attention-based fusion module."""
    def __init__(self, latent_dim, num_modalities):
        super().__init__()
        self.query_net = nn.Sequential(nn.Linear(latent_dim, latent_dim))
        self.key_nets = nn.ModuleList([
            nn.Linear(latent_dim, latent_dim) for _ in range(num_modalities)
        ])
        
    def forward(self, gnn_output, modality_latents):
        N, M, D = modality_latents.shape
        query = self.query_net(gnn_output)
        
        attention_scores = torch.zeros(N, M, device=gnn_output.device)
        for i in range(M):
            key = self.key_nets[i](modality_latents[:, i, :])
            attention_scores[:, i] = (query * key).sum(dim=-1) / (D ** 0.5)
            
        attention_weights = F.softmax(attention_scores, dim=1)
        fused_latent = (attention_weights.unsqueeze(-1) * modality_latents).sum(dim=1)
        return fused_latent, attention_weights

class GDIFNet(nn.Module):
    """Graph-based Disentangled and Interpretable Fusion Network"""
    def __init__(self, num_modalities, latent_dim, gnn_layers, gat_heads):
        super().__init__()
        self.num_modalities = num_modalities
        self.latent_dim = latent_dim

        self.encoders = nn.ModuleList([
            DisentangledEncoder(1, latent_dim) for _ in range(num_modalities)
        ])

        self.initial_fusion_mlp = nn.Linear(latent_dim, latent_dim)
        
        self.gnn_layers = nn.ModuleList()
        in_channels = latent_dim
        for i in range(gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(in_channels, latent_dim, heads=gat_heads, concat=False, dropout=0.1)
            )
            in_channels = latent_dim

        self.fusion_module = InterpretableFusion(latent_dim, num_modalities)
        
        self.risk_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, multi_modal_data, edge_index):
        # 1. Disentangled Encoding
        modality_latents = [
            self.encoders[i](multi_modal_data[:, i].unsqueeze(-1))
            for i in range(self.num_modalities)
        ]
        modality_latents_tensor = torch.stack(modality_latents, dim=1)

        # 2. Initial Fusion for GNN input
        initial_fused_latent = self.initial_fusion_mlp(modality_latents_tensor.mean(dim=1))

        # 3. GNN for Spatial Information Propagation
        gnn_output = initial_fused_latent
        for layer in self.gnn_layers:
            gnn_output = F.gelu(layer(gnn_output, edge_index))

        # 4. Interpretable Fusion
        final_fused_latent, attention_weights = self.fusion_module(gnn_output, modality_latents_tensor)
        
        # 5. Prediction Heads
        risk_value = self.risk_head(final_fused_latent)
        epistemic_uncertainty = self.uncertainty_head(final_fused_latent)
        
        return risk_value, epistemic_uncertainty, attention_weights, modality_latents_tensor
