import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data
import numpy as np
import os
from tqdm import tqdm
import json

from core.graph_constructor import create_3d_grid_graph_pyg
from core.model import GDIFNet

# --- Config ---
with open('configs/default_config.json', 'r') as f:
    config = json.load(f)

DATA_DIR = "training_data"
DIMS = config['grid']['dimensions']
MODEL_CONFIG = config['model']
TRAIN_CONFIG = config['training']
EPOCHS = TRAIN_CONFIG['epochs']
BATCH_SIZE = TRAIN_CONFIG['batch_size']
LEARNING_RATE = TRAIN_CONFIG['learning_rate']

# --- Dataset ---
class RiskDataset(Dataset):
    def __init__(self, root, graph):
        self.graph = graph
        self.filenames = [os.path.join(root, f) for f in os.listdir(root) if f.endswith('.npz')]
        super().__init__(root)

    def len(self):
        return len(self.filenames)

    def get(self, idx):
        filepath = self.filenames[idx]
        with np.load(filepath) as file:
            data = torch.from_numpy(file['data'].astype(np.float32))
            label = torch.from_numpy(file['label'].astype(np.float32))
        
        # Each sample is a graph with new node features (x) and labels (y)
        sample_graph = self.graph.clone()
        sample_graph.x = data
        sample_graph.y = label
        return sample_graph

# --- Init ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

base_graph = create_3d_grid_graph_pyg(DIMS)
model = GDIFNet(**MODEL_CONFIG).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
dataset = RiskDataset(DATA_DIR, base_graph)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        risk_pred, _, _, _ = model(batch.x, batch.edge_index)
        
        loss = criterion(risk_pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.6f}")

    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"gdif_net_epoch_{epoch+1}.pth")

print("Training complete.")
torch.save(model.state_dict(), "gdif_net_final.pth")
