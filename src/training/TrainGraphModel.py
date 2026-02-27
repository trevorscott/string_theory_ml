import torch
import torch.nn as nn
import numpy as np
import os
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 1. Setup Device (M2 Max)
# Note: If PyTorch Geometric throws a fit about MPS with certain graph operations, 
# we can switch this to "cpu", but let's try pushing the GPU first!
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🚀 Igniting Graph Training on: {device}")

# 2. Load the 3D Web Dataset
data = torch.load("smart_graph_multiverse.pt", weights_only=False)
graph_dataset = data["graphs"]
y_dim = data["y_dim"]

# Dynamic 80/20 Split
train_size = int(0.8 * len(graph_dataset))
val_size = len(graph_dataset) - train_size

# PyTorch Geometric has its own special data loaders
train_dataset = graph_dataset[:train_size]
val_dataset = graph_dataset[train_size:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# 3. The Calabi-Yau Graph Neural Network
class UniverseGNN(nn.Module):
    def __init__(self, node_features, output_dim):
        super(UniverseGNN, self).__init__()
        
        # Message Passing Layers (Walking the web)
        self.conv1 = GCNConv(node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        
        # Final prediction layers
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        # 2. Global Pooling (Squish the web into one fingerprint)
        # 'batch' tells the pooling layer which nodes belong to which universe
        x = global_mean_pool(x, batch)  
        
        # 3. Predict the Physics
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

# Each node has 5 features (the 5 vertex indices of the simplex)
model = UniverseGNN(node_features=6, output_dim=y_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 150
print(f"🏋️  Training Graph Model on {train_size} universes, validating on {val_size}...")

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # GNNs need x, edge_index, and the batch mapping
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    if (epoch + 1) % 15 == 0:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_batch in val_loader:
                v_batch = v_batch.to(device)
                v_out = model(v_batch.x, v_batch.edge_index, v_batch.batch)
                val_loss += criterion(v_out, v_batch.y).item()
                
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

torch.save(model.state_dict(), "gnn_universe_model.pth")
print("\n✅ GNN Model Saved! The AI now truly sees in higher dimensions.")