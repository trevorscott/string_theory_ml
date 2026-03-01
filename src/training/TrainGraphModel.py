import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# 1. The NumPy 2.0 Fix (Compatible with NumPy 1.x and 2.x)
try:
    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
except AttributeError:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])

# 2. The Calabi-Yau Graph Neural Network (BottNet)
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
        x = global_mean_pool(x, batch)  
        
        # 3. Predict the Physics
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        
        return out

if __name__ == "__main__":
    # 3. Command-Line Arguments
    parser = argparse.ArgumentParser(description="Train the Predictive Physics Oracle (BottNet).")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the processed graph dataset (e.g., data/smart_graph_50.pt)")
    parser.add_argument('-o', '--output', type=str, default="checkpoints/gnn_universe_model.pth", 
                        help="Path to save the trained Oracle weights.")
    parser.add_argument('-e', '--epochs', type=int, default=150, 
                        help="Number of training epochs (default: 150)")
    parser.add_argument('-b', '--batch_size', type=int, default=32, 
                        help="Training batch size (default: 32)")
    
    args = parser.parse_args()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Igniting Graph Training on: {device}")

    # 4. Dynamically Load Data
    if not os.path.exists(args.input):
        print(f"❌ Error: Dataset '{args.input}' not found.")
        exit(1)

    print(f"📥 Loading graph dataset from: {args.input}")
    data = torch.load(args.input, weights_only=False)
    graph_dataset = data["graphs"]
    y_dim = data["y_dim"]

    # Dynamic 80/20 Split
    train_size = int(0.8 * len(graph_dataset))
    val_size = len(graph_dataset) - train_size

    train_dataset = graph_dataset[:train_size]
    val_dataset = graph_dataset[train_size:]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Note: Each node has 6 features (1 degree + 5 vertex popularities)
    model = UniverseGNN(node_features=6, output_dim=y_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    # 5. Training Loop
    print(f"🏋️  Training Graph Model on {train_size} universes, validating on {val_size}...")
    print(f"⚙️  Settings: {args.epochs} Epochs | Batch Size: {args.batch_size}")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Dynamic print interval based on total epochs
        print_interval = max(1, args.epochs // 10)
        if (epoch + 1) % print_interval == 0 or epoch == args.epochs - 1:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for v_batch in val_loader:
                    v_batch = v_batch.to(device)
                    v_out = model(v_batch.x, v_batch.edge_index, v_batch.batch)
                    val_loss += criterion(v_out, v_batch.y).item()
                    
            avg_train = train_loss / len(train_loader)
            avg_val = val_loss / len(val_loader)
            print(f"   > Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # 6. Save the Oracle
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print("\n" + "="*50)
    print(f"✅ GNN MODEL SAVED")
    print(f"Weights written to: {args.output}")
    print("The Oracle now truly sees in higher dimensions.")
    print("="*50)