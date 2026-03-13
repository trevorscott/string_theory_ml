import torch
import torch.nn as nn
import math
from torch_geometric.nn import DenseGCNConv

class SinusoidalTimeEmbeddings(nn.Module):
    """
    Standard diffusion time embedding. 
    The network needs to know what timestep it is looking at so it knows 
    how aggressively it needs to denoise the graph.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DenseDenoisingGNN(nn.Module):
    def __init__(self, num_nodes, hidden_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        
        # 1. Time Context
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # 2. Initial Node Features
        # Since Calabi-Yau topological nodes don't have intrinsic "features" 
        # other than their structural ID, we give them a learnable starting embedding.
        self.node_emb = nn.Embedding(num_nodes, hidden_dim)
        
        # 3. Message Passing Layers
        # DenseGCNConv expects a dense adjacency matrix [batch, nodes, nodes]
        self.conv1 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGCNConv(hidden_dim, hidden_dim)
        
        # 4. Edge Predictor
        # Takes the embeddings of two nodes and predicts if an edge connects them
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Outputs 1 raw logit per edge pair
        )

    def forward(self, noisy_adj, t):
        batch_size = noisy_adj.shape[0]
        device = noisy_adj.device
        
        # --- Embed the Time ---
        t_emb = self.time_mlp(t) # [batch_size, hidden_dim]
        
        # --- Embed the Nodes ---
        # Give every node its initial structural ID
        node_ids = torch.arange(self.num_nodes, device=device).expand(batch_size, -1)
        x = self.node_emb(node_ids) # [batch_size, num_nodes, hidden_dim]
        
        # Inject the time context into the node features
        x = x + t_emb.unsqueeze(1) 
        
        # --- Message Passing ---
        # The GNN looks at the noisy structure to figure out the underlying geometry
        x = self.conv1(x, noisy_adj)
        x = torch.relu(x)
        x = self.conv2(x, noisy_adj)
        x = torch.relu(x)
        x = self.conv3(x, noisy_adj)
        
        # --- Edge Prediction ---
        # To predict an edge between node i and node j, we concatenate their features.
        # This expands x into a matrix of all possible node pairs.
        x_i = x.unsqueeze(2).expand(-1, -1, self.num_nodes, -1)
        x_j = x.unsqueeze(1).expand(-1, self.num_nodes, -1, -1)
        edge_features = torch.cat([x_i, x_j], dim=-1) # [batch, nodes, nodes, 2 * hidden_dim]
        
        # Predict the logits for every pair
        edge_logits = self.edge_mlp(edge_features).squeeze(-1) # [batch, nodes, nodes]
        
        # Calabi-Yau wireframes are undirected, so we force the matrix to be symmetric
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        return edge_logits


# ==========================================
# TEST BLOCK: Ensure the network can process the noise
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Let's simulate the garbage matrix we got from timestep 99 in the last script
    num_nodes = 10
    batch_size = 2
    noisy_garbage_adj = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float().to(device)
    
    # Ensure the input is symmetric (undirected graph)
    noisy_garbage_adj = (noisy_garbage_adj + noisy_garbage_adj.transpose(1, 2)) / 2
    noisy_garbage_adj = (noisy_garbage_adj > 0).float()
    
    # Simulate the timesteps (e.g., this batch is at t=99 and t=50)
    timesteps = torch.tensor([99, 50], device=device)
    
    # Initialize the model
    model = DenseDenoisingGNN(num_nodes=num_nodes, hidden_dim=64).to(device)
    
    print("--- MODEL ARCHITECTURE TEST ---\n")
    print(f"Input Noisy Graph Shape: {noisy_garbage_adj.shape}")
    print(f"Input Timesteps: {timesteps}")
    
    # Run the forward pass
    predicted_clean_logits = model(noisy_garbage_adj, timesteps)
    
    print(f"\nOutput Predicted Logits Shape: {predicted_clean_logits.shape}")
    print("\nPredicted Logits (Unnormalized probabilities) for Batch 0:")
    print(predicted_clean_logits[0].detach().cpu().numpy().round(2))
    print("\nSUCCESS: The model successfully ingested the noise and output symmetric structural predictions.")
