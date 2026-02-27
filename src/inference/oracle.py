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

# 2. The GNN Architecture (Must match the training script exactly)
class UniverseGNN(nn.Module):
    def __init__(self, node_features, output_dim):
        super(UniverseGNN, self).__init__()
        self.conv1 = GCNConv(node_features, 128)
        self.conv2 = GCNConv(128, 256)
        self.conv3 = GCNConv(256, 256)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x = global_mean_pool(x, batch)  
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        out = self.fc2(x)
        return out

if __name__ == "__main__":
    # 3. Command-Line Arguments
    parser = argparse.ArgumentParser(description="Consult the Predictive Physics Oracle.")
    parser.add_argument('-i', '--input', type=str, required=True, 
                        help="Path to the processed graph dataset to analyze.")
    parser.add_argument('-m', '--model', type=str, default="checkpoints/gnn_universe_model.pth", 
                        help="Path to the trained Oracle model weights.")
    
    args = parser.parse_args()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    # 4. Safety Checks
    if not os.path.exists(args.input):
        print(f"❌ Error: Input graph '{args.input}' not found.")
        exit(1)
    if not os.path.exists(args.model):
        print(f"❌ Error: Model weights '{args.model}' not found. Did you train BottNet yet?")
        exit(1)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("🔮 Waking up the Oracle...")

    # 5. Load Data and Model
    data = torch.load(args.input, weights_only=False)
    graph_dataset = data["graphs"]
    y_dim = data["y_dim"]

    model = UniverseGNN(node_features=6, output_dim=y_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.eval()

    print(f"👁️  Analyzing {len(graph_dataset)} target universes...")

    # 6. Inference Loop (Displaying the first 5 predictions)
    loader = DataLoader(graph_dataset, batch_size=1) 
    
    print("\n" + "="*50)
    print("✨ ORACLE PREDICTIONS ✨")
    print("="*50)

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= 5: 
                print("  ... (output truncated for screen)")
                break
            
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            
            actual = batch.y.cpu().numpy()[0]
            predicted = pred.cpu().numpy()[0]
            
            print(f"Universe #{idx+1} Geometry Scanned.")
            print(f"  > Predicted Physics: {np.round(predicted, 2)}")
            print(f"  > Actual Physics:    {np.round(actual, 2)}\n")
            
    print("="*50)